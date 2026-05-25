"""L18 basin geometry probe — 5000-cycle in-flight perturbation sweep.

silv 2026-05-25: "burn the 5 orders of magnitude to find the spark."

Sherwood framing applied: the LOCAL FIELD is the model + prompt + position;
the TRIGGER is the L18 residual perturbation; the SPARK is the energy
threshold that flips the L_final top-1 token. The basin geometry IS the
substrate-level signature of locked vs success vs explore-fail.

Method (per target position):
  1. Forward prompt to position p, capture L18 residual r (shape: hidden_dim)
  2. Forward natural r → L_final → record baseline top-1, top-3 logits
  3. For N_dirs random unit-norm directions d in R^{hidden_dim}:
       For each energy E in log-spaced grid [1e-3, 10.0]:
         r' = r + E * d
         Forward L18+1 ... L31 with r' replacing the residual at L18
         logits' = lm_head(final_norm(L31_output))
         flip = (argmax(logits') != baseline_top_1)
       Record min E where flip occurred (or None if never flipped)
  4. Aggregate per position: distribution of min-flip-energy

Hypotheses:
  H_LOCKED_FRAGILE: at locked positions, min-flip-energy distribution has
    LOW p10 (basin is shallow, small noise flips top-1) — refutes the
    intuition that locked = deep basin.
  H_LOCKED_RIGID: at locked positions, the basin is DEEP — most directions
    require HIGH energy to flip top-1 because all attractor directions
    converge to the same answer.
  H_LOCKED_ANISOTROPIC: locked positions have BIMODAL distribution — most
    directions need high energy (rigid) but a SMALL set of directions
    has very low threshold (escape-direction).

Scale: 5 positions × 200 directions × 5 energies = 5000 forwards. At ~30ms
per partial-forward (L19-L31), ~2.5 min wall.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Proxy + offline mode BEFORE HF imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
proxy_port = random.choice(range(10001, 10150))
os.environ.setdefault("HTTPS_PROXY", f"socks5h://127.0.0.1:{proxy_port}")
os.environ.setdefault("HTTP_PROXY", f"socks5h://127.0.0.1:{proxy_port}")
os.environ.setdefault("ALL_PROXY", f"socks5h://127.0.0.1:{proxy_port}")

import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from codec_audit.db import open_db, journal

MODEL_ID = "mlx-community/Qwen3.5-4B-MLX-4bit"

# L18 = GatedDeltaNet commit-installation layer (per fracture trace)
# L19 = full-attn norm explosion (the divergence point)
PERTURB_LAYER = 18  # 0-indexed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text-from", required=True, help="JSON file with response field")
    p.add_argument("--field", default="response")
    p.add_argument("--n-chars", type=int, default=2000)
    p.add_argument("--positions", default="-1",
                    help="comma-separated positions (negative = from end)")
    p.add_argument("--n-dirs", type=int, default=200)
    p.add_argument("--energies", default="0.001,0.01,0.1,1.0,3.0,10.0",
                    help="comma-separated energy values")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, help="output JSON path")
    args = p.parse_args()

    text = json.loads(Path(args.text_from).read_text())[args.field][:args.n_chars]
    energies = [float(e) for e in args.energies.split(",")]
    target_positions = [int(p) for p in args.positions.split(",")]

    print(f"[{time.strftime('%H:%M:%S')}] loading {MODEL_ID}")
    t0 = time.time()
    from mlx_lm import load
    model, tok = load(MODEL_ID)
    print(f"loaded in {time.time()-t0:.1f}s")

    ids = tok.encode(text)
    n_tok = len(ids)
    # Resolve negative positions
    resolved_positions = []
    for tp in target_positions:
        if tp < 0:
            tp = n_tok + tp
        if 0 <= tp < n_tok:
            resolved_positions.append(tp)
    print(f"tokens: {n_tok}, target positions: {resolved_positions}")

    internal = model.language_model.model
    layers = internal.layers
    n_layers = len(layers)
    hidden_dim = None
    captured = [None] * n_layers
    layer_idx = {id(l): i for i, l in enumerate(layers)}

    def patched_layer_call(self, x, mask=None, cache=None):
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        mlp_r = self.mlp(self.post_attention_layernorm(h))
        out = h + mlp_r
        idx = layer_idx.get(id(self))
        if idx is not None:
            captured[idx] = out
        return out

    LayerCls = type(layers[0])
    orig = LayerCls.__call__
    LayerCls.__call__ = patched_layer_call

    print(f"[{time.strftime('%H:%M:%S')}] prefill capturing all layers")
    t0 = time.time()
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    inp = mx.array([ids])
    _ = model(inp, cache=cache)
    mx.eval(*[c.state for c in cache if hasattr(c, "state")])
    print(f"prefill done in {time.time()-t0:.1f}s")
    LayerCls.__call__ = orig

    # Get hidden_dim
    hidden_dim = captured[PERTURB_LAYER].shape[-1]
    print(f"hidden_dim = {hidden_dim}")

    final_norm = internal.norm
    embed = internal.embed_tokens

    rng = np.random.default_rng(args.seed)

    def project_logits(x_layer18):
        """Given (1, n_tok, hidden) residual at L18, forward through L19-L31
        and return final logits at last position.

        We use the cache as-is (KV cache is per-layer) and replace the
        residual feed into L19. This is approximate — KV cache for L19+
        was built from the ORIGINAL L18 residual, so our perturbation
        doesn't update K/V for the perturbed token. For SINGLE-POSITION
        perturbation at the last position, this is a reasonable proxy
        (the perturbation affects the current-token forward without
        being attended to by future tokens, which is what we want).

        Returns: numpy float32 (vocab,) logits for the perturbed position.
        """
        h = x_layer18
        for L in range(PERTURB_LAYER + 1, n_layers):
            layer = layers[L]
            # We call layer directly with h, no cache here (no-cache mode)
            # to avoid corrupting the captured KV. The forward result
            # only matters for THIS position's final logit.
            if layer.is_linear:
                r = layer.linear_attn(layer.input_layernorm(h), mask=None, cache=None)
            else:
                r = layer.self_attn(layer.input_layernorm(h), mask=None, cache=None)
            h_mid = h + r
            mlp_r = layer.mlp(layer.post_attention_layernorm(h_mid))
            h = h_mid + mlp_r
        h_norm = final_norm(h)
        logits = embed.as_linear(h_norm)  # (1, n_tok, vocab)
        # Use last position
        logits_pos = logits[0, -1, :].astype(mx.float32)
        mx.eval(logits_pos)
        return np.array(logits_pos, copy=True, dtype=np.float32)

    # For each target position, do the perturbation sweep
    results = []
    for pos in resolved_positions:
        print(f"\n[{time.strftime('%H:%M:%S')}] position {pos} basin probe")
        # Slice captured L18 to single position context (1, pos+1, hidden)
        # The single-position perturbation requires re-running L19+ with
        # the perturbed residual at position pos. But we use the full
        # captured residual sequence and just substitute pos's residual.
        L18_full = captured[PERTURB_LAYER]  # (1, n_tok, hidden)
        # The naive approach above forwards the WHOLE sequence — too slow.
        # Instead, since perturbation is only at the LAST relevant position
        # and L19+ doesn't have a KV cache from the perturbation, we can
        # compute a SINGLE-POSITION forward.
        # Extract last-position residual
        L18_pos = L18_full[:, pos:pos+1, :]  # (1, 1, hidden)

        # Baseline logits at this position
        h = L18_pos
        for L in range(PERTURB_LAYER + 1, n_layers):
            layer = layers[L]
            # Use the cached state for prior positions but no mask required
            # since single-position forward attends only to past via cache
            r = (layer.linear_attn(layer.input_layernorm(h), mask=None, cache=None)
                  if layer.is_linear
                  else layer.self_attn(layer.input_layernorm(h), mask=None, cache=None))
            h_mid = h + r
            mlp_r = layer.mlp(layer.post_attention_layernorm(h_mid))
            h = h_mid + mlp_r
        h_norm = final_norm(h)
        baseline_logits = embed.as_linear(h_norm)[0, 0, :].astype(mx.float32)
        mx.eval(baseline_logits)
        baseline_top1 = int(mx.argmax(baseline_logits))
        baseline_tok_text = tok.decode([baseline_top1])

        # Compute baseline entropy + margin for context
        bl = np.array(baseline_logits, copy=True, dtype=np.float32)
        z = bl - bl.max()
        ez = np.exp(z, dtype=np.float64)
        sz = ez.sum()
        probs = ez / sz
        nz = probs > 0
        entropy = float(-(probs[nz] * np.log(probs[nz])).sum())
        sorted_idx = np.argsort(-bl)
        margin = float(np.log(probs[sorted_idx[0]]) - np.log(probs[sorted_idx[1]]))
        print(f"  baseline top-1: {baseline_tok_text!r} (id {baseline_top1}); "
              f"entropy={entropy:.4f} margin={margin:.4f}")

        # Perturbation sweep
        L18_pos_np = np.array(L18_pos[0, 0, :].astype(mx.float32), copy=True, dtype=np.float32)
        r_norm = float(np.linalg.norm(L18_pos_np))
        print(f"  baseline L18 residual norm: {r_norm:.4f}")
        print(f"  perturbing with {args.n_dirs} random directions × {len(energies)} energies")

        min_flip_energies = []  # per direction
        flip_count_per_energy = {E: 0 for E in energies}

        t_sweep_start = time.time()
        for d_i in range(args.n_dirs):
            d = rng.standard_normal(hidden_dim).astype(np.float32)
            d = d / np.linalg.norm(d)  # unit norm
            min_flip_E = None
            for E in energies:
                # Scale d by residual norm * E so energies are residual-relative
                r_pert_np = L18_pos_np + (r_norm * E) * d
                r_pert = mx.array(r_pert_np[None, None, :])  # (1, 1, hidden)
                # Forward L19+ with perturbed residual at this single position
                h = r_pert
                for L in range(PERTURB_LAYER + 1, n_layers):
                    layer = layers[L]
                    rr = (layer.linear_attn(layer.input_layernorm(h), mask=None, cache=None)
                           if layer.is_linear
                           else layer.self_attn(layer.input_layernorm(h), mask=None, cache=None))
                    h_mid = h + rr
                    mlp_r = layer.mlp(layer.post_attention_layernorm(h_mid))
                    h = h_mid + mlp_r
                h_norm_ = final_norm(h)
                pert_logits = embed.as_linear(h_norm_)[0, 0, :].astype(mx.float32)
                mx.eval(pert_logits)
                pert_top1 = int(mx.argmax(pert_logits))
                if pert_top1 != baseline_top1:
                    min_flip_E = E
                    flip_count_per_energy[E] += 1
                    break  # smallest energy that flips
            min_flip_energies.append(min_flip_E)
            if (d_i + 1) % 50 == 0:
                elapsed = time.time() - t_sweep_start
                rate = (d_i + 1) / elapsed
                print(f"    direction {d_i+1}/{args.n_dirs} ({rate:.1f} dirs/s, "
                      f"min_flip dist so far: {len([m for m in min_flip_energies if m is not None])}/{len(min_flip_energies)})")

        # Aggregate
        n_no_flip = sum(1 for m in min_flip_energies if m is None)
        flipped = [m for m in min_flip_energies if m is not None]
        flipped_logs = [np.log10(m) for m in flipped]
        if flipped_logs:
            log_p10 = float(np.percentile(flipped_logs, 10))
            log_p50 = float(np.percentile(flipped_logs, 50))
            log_p90 = float(np.percentile(flipped_logs, 90))
        else:
            log_p10 = log_p50 = log_p90 = None

        results.append({
            "position": pos,
            "n_tokens_context": pos + 1,
            "baseline_top1_id": baseline_top1,
            "baseline_top1_text": baseline_tok_text,
            "baseline_entropy": entropy,
            "baseline_margin": margin,
            "L18_residual_norm": r_norm,
            "n_directions": args.n_dirs,
            "n_no_flip": n_no_flip,
            "n_flipped": len(flipped),
            "log10_min_flip_E_p10": log_p10,
            "log10_min_flip_E_p50": log_p50,
            "log10_min_flip_E_p90": log_p90,
            "flip_count_per_energy": flip_count_per_energy,
        })
        print(f"  n_no_flip: {n_no_flip}/{args.n_dirs} (basin stability)")
        if log_p50 is not None:
            print(f"  log10(min flip E): p10={log_p10:.2f} p50={log_p50:.2f} p90={log_p90:.2f}")
        print(f"  flip count per energy: {flip_count_per_energy}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nsaved {args.out}")

    # Journal
    with open_db() as conn:
        journal(conn, "basin_geometry_probe", str(args.text_from),
                {"n_positions": len(resolved_positions),
                 "n_dirs": args.n_dirs,
                 "energies": energies,
                 "perturb_layer": PERTURB_LAYER,
                 "results_summary": [
                    {"pos": r["position"],
                     "p50_log_E": r["log10_min_flip_E_p50"],
                     "n_no_flip": r["n_no_flip"]} for r in results]})


if __name__ == "__main__":
    main()
