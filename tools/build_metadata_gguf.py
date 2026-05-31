#!/usr/bin/env python3
"""Build a metadata-only GGUF for DS4 V4 Flash, directly from the raw model dir.

silv 2026-05-28 directive: "generate your missing pieces directly from the
safetensors files of the original model" and "do NOT take anything from the
minimal-gguf, but only from the raw data from the original model".

What this emits:
  - All `deepseek4.*` KV the engine reads (35 scalars + 2 arrays)
  - `general.architecture` + `general.name`
  - `tokenizer.ggml.tokens` (129280 STRING array)
  - `tokenizer.ggml.merges` (127741 STRING array)
  - Zero tensors (tensor_count = 0)

The point: tokenizer + arch metadata extracted from raw sources only. Tensor
data comes from the two packs. The engine_open path needs surgery to consume
this bundle alongside the packs (#771 follow-up).

Sources read (all from the original HuggingFace model dir):
  - config.json                      → 35 scalar arch params
  - tokenizer.json                   → 129280 vocab tokens + 127741 BPE merges
  - tokenizer_config.json            → (chat template, optional)
  - NO minimal-GGUF read             → enforced by file-path scope

Output:
  ds4_metadata.gguf (~7-10 MB, no tensor data)

Run:
  python3 tools/build_metadata_gguf.py \
    --src-dir /Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash \
    --out ds4_metadata.gguf
"""

import argparse
import json
import struct
import sys
from pathlib import Path

# ============================================================================
# GGUF constants (subset; tensor types unused — we emit zero tensors)
# ============================================================================

GGUF_MAGIC = 0x46554747  # 'GGUF'
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12


# ============================================================================
# Tiny GGUF writer (KV-only — no tensor section)
# ============================================================================

class GGUFMetadataWriter:
    def __init__(self, path: str):
        self.f = open(path, "wb")
        self.kv = []  # list of (key, vtype, value)

    def add(self, key: str, vtype: int, value):
        self.kv.append((key, vtype, value))

    def _write_str(self, s: str):
        b = s.encode("utf-8")
        self.f.write(struct.pack("<Q", len(b)))
        self.f.write(b)

    def _write_value(self, vtype: int, value):
        if   vtype == GGUF_TYPE_UINT32:  self.f.write(struct.pack("<I", value))
        elif vtype == GGUF_TYPE_INT32:   self.f.write(struct.pack("<i", value))
        elif vtype == GGUF_TYPE_UINT64:  self.f.write(struct.pack("<Q", value))
        elif vtype == GGUF_TYPE_INT64:   self.f.write(struct.pack("<q", value))
        elif vtype == GGUF_TYPE_FLOAT32: self.f.write(struct.pack("<f", value))
        elif vtype == GGUF_TYPE_FLOAT64: self.f.write(struct.pack("<d", value))
        elif vtype == GGUF_TYPE_BOOL:    self.f.write(struct.pack("<B", 1 if value else 0))
        elif vtype == GGUF_TYPE_STRING:  self._write_str(value)
        elif vtype == GGUF_TYPE_UINT8:   self.f.write(struct.pack("<B", value))
        elif vtype == GGUF_TYPE_INT8:    self.f.write(struct.pack("<b", value))
        elif vtype == GGUF_TYPE_ARRAY:
            etype, values = value
            self.f.write(struct.pack("<I", etype))
            self.f.write(struct.pack("<Q", len(values)))
            for v in values:
                self._write_value(etype, v)
        else:
            raise ValueError(f"unsupported value type {vtype}")

    def write(self):
        # Header: magic, version, tensor_count=0, kv_count=N
        self.f.write(struct.pack("<I", GGUF_MAGIC))
        self.f.write(struct.pack("<I", GGUF_VERSION))
        self.f.write(struct.pack("<Q", 0))            # tensor_count
        self.f.write(struct.pack("<Q", len(self.kv))) # kv_count

        for key, vtype, value in self.kv:
            self._write_str(key)
            self.f.write(struct.pack("<I", vtype))
            self._write_value(vtype, value)

        # Pad to alignment (matches GGUF convention, though no data section)
        pos = self.f.tell()
        pad = (GGUF_DEFAULT_ALIGNMENT - (pos % GGUF_DEFAULT_ALIGNMENT)) % GGUF_DEFAULT_ALIGNMENT
        self.f.write(b"\x00" * pad)
        self.f.close()


# ============================================================================
# config.json → deepseek4.* KV mapping
# ============================================================================

def emit_arch_kv(w: GGUFMetadataWriter, cfg: dict):
    """Emit all 35 scalar deepseek4.* keys that ds4.c reads, derived from
    config.json directly. Engine validates these against hardcoded constants
    (DS4_VOCAB_SIZE, DS4_SWIGLU_CLAMP_EXP, etc.) — so values must match the
    real model config.
    """
    # general.*
    w.add("general.architecture", GGUF_TYPE_STRING, "deepseek4")
    w.add("general.name",         GGUF_TYPE_STRING, "DeepSeek-V4-Flash")

    # Core sizes
    w.add("deepseek4.block_count",            GGUF_TYPE_UINT32, cfg["num_hidden_layers"])      # 43
    w.add("deepseek4.context_length",         GGUF_TYPE_UINT64, cfg["max_position_embeddings"])# 1048576
    w.add("deepseek4.embedding_length",       GGUF_TYPE_UINT32, cfg["hidden_size"])            # 4096
    w.add("deepseek4.vocab_size",             GGUF_TYPE_UINT32, cfg["vocab_size"])             # 129280

    # Attention
    w.add("deepseek4.attention.head_count",       GGUF_TYPE_UINT32, cfg["num_attention_heads"])    # 64
    w.add("deepseek4.attention.head_count_kv",    GGUF_TYPE_UINT32, cfg["num_key_value_heads"])    # 1
    w.add("deepseek4.attention.key_length",       GGUF_TYPE_UINT32, cfg["head_dim"])               # 512
    # MLA has a separate value_length; in DS4 V4 Flash it equals head_dim per inference/config.json
    w.add("deepseek4.attention.value_length",     GGUF_TYPE_UINT32, cfg.get("v_head_dim", cfg["head_dim"]))  # 512
    w.add("deepseek4.attention.sliding_window",   GGUF_TYPE_UINT32, cfg["sliding_window"])         # 128
    w.add("deepseek4.attention.q_lora_rank",      GGUF_TYPE_UINT32, cfg["q_lora_rank"])            # 1024
    w.add("deepseek4.attention.output_lora_rank", GGUF_TYPE_UINT32, cfg["o_lora_rank"])            # 1024
    w.add("deepseek4.attention.output_group_count", GGUF_TYPE_UINT32, cfg["o_groups"])             # 8

    # Indexer (DS4 hierarchical attention)
    w.add("deepseek4.attention.indexer.head_count", GGUF_TYPE_UINT32, cfg["index_n_heads"])  # 64
    w.add("deepseek4.attention.indexer.key_length", GGUF_TYPE_UINT32, cfg["index_head_dim"]) # 128
    w.add("deepseek4.attention.indexer.top_k",      GGUF_TYPE_UINT32, cfg["index_topk"])     # 512

    # Expert (MoE)
    w.add("deepseek4.expert_count",             GGUF_TYPE_UINT32, cfg["n_routed_experts"])         # 256
    w.add("deepseek4.expert_used_count",        GGUF_TYPE_UINT32, cfg["num_experts_per_tok"])      # 6
    w.add("deepseek4.expert_feed_forward_length", GGUF_TYPE_UINT32, cfg["moe_intermediate_size"])  # 2048
    w.add("deepseek4.expert_shared_count",      GGUF_TYPE_UINT32, cfg["n_shared_experts"])         # 1
    w.add("deepseek4.expert_weights_scale",     GGUF_TYPE_FLOAT32, cfg["routed_scaling_factor"])   # 1.5
    w.add("deepseek4.expert_weights_norm",      GGUF_TYPE_BOOL, cfg["norm_topk_prob"])             # true
    # Expert group sizing — DS4 V4 Flash uses 8 groups of 32, picks 4 used
    w.add("deepseek4.expert_group_count",       GGUF_TYPE_UINT32, cfg.get("n_group", 8))           # 8
    w.add("deepseek4.expert_group_used_count",  GGUF_TYPE_UINT32, cfg.get("topk_group", 4))        # 4

    # Hash layers (DS4 hash-table specific)
    w.add("deepseek4.hash_layer_count",         GGUF_TYPE_UINT32, cfg["num_hash_layers"])          # 3

    # Hyper-connection
    w.add("deepseek4.hyper_connection.count",                GGUF_TYPE_UINT32, cfg["hc_mult"])           # 4
    w.add("deepseek4.hyper_connection.sinkhorn_iterations",  GGUF_TYPE_UINT32, cfg["hc_sinkhorn_iters"]) # 20
    w.add("deepseek4.hyper_connection.epsilon",              GGUF_TYPE_FLOAT32, cfg["hc_eps"])           # 1e-6

    # RoPE
    w.add("deepseek4.rope.dimension_count",                  GGUF_TYPE_UINT32, cfg["qk_rope_head_dim"])  # 64
    w.add("deepseek4.rope.scaling.original_context_length",  GGUF_TYPE_UINT64,
          cfg["rope_scaling"]["original_max_position_embeddings"])                                       # 65536
    w.add("deepseek4.rope.freq_base",                        GGUF_TYPE_FLOAT32, cfg["rope_theta"])       # 10000
    w.add("deepseek4.rope.scaling.factor",                   GGUF_TYPE_FLOAT32, cfg["rope_scaling"]["factor"])     # 16
    w.add("deepseek4.rope.scaling.yarn_beta_fast",           GGUF_TYPE_FLOAT32, cfg["rope_scaling"]["beta_fast"])  # 32
    w.add("deepseek4.rope.scaling.yarn_beta_slow",           GGUF_TYPE_FLOAT32, cfg["rope_scaling"]["beta_slow"])  # 1
    w.add("deepseek4.attention.compress_rope_freq_base",     GGUF_TYPE_FLOAT32, cfg["compress_rope_theta"])         # 160000

    # RMS norm
    w.add("deepseek4.attention.layer_norm_rms_epsilon",      GGUF_TYPE_FLOAT32, cfg["rms_norm_eps"])      # 1e-6

    # Per-layer arrays
    n_layer = cfg["num_hidden_layers"]
    compress_ratios = list(cfg["compress_ratios"])
    if len(compress_ratios) < n_layer:
        raise ValueError(f"compress_ratios shorter than num_hidden_layers ({len(compress_ratios)} < {n_layer})")
    w.add("deepseek4.attention.compress_ratios",
          GGUF_TYPE_ARRAY, (GGUF_TYPE_UINT32, [int(x) for x in compress_ratios]))

    # SwiGLU clamp_exp: scalar in config.json, but engine expects per-layer array
    swiglu_clamp = float(cfg["swiglu_limit"])
    w.add("deepseek4.swiglu_clamp_exp",
          GGUF_TYPE_ARRAY, (GGUF_TYPE_FLOAT32, [swiglu_clamp] * n_layer))


# ============================================================================
# tokenizer.json → tokens + merges
# ============================================================================

def build_token_array(tokenizer_json: dict, vocab_size: int) -> list[str]:
    """Build vocab_size-length array of token strings, indexed by token ID.

    Sources (in order, later overrides earlier — should not happen, but enforced):
      1. tokenizer.model.vocab: {token_string: id}  → 128000 BPE tokens
      2. tokenizer.added_tokens: [{id, content, ...}, ...] → 1283 special tokens

    Engine expects: tokens[id] = string for every id ∈ [0, vocab_size).
    """
    tokens = [None] * vocab_size

    # BPE vocab (id-keyed-by-value in HF format)
    bpe_vocab = tokenizer_json["model"]["vocab"]
    for tok_str, tok_id in bpe_vocab.items():
        if 0 <= tok_id < vocab_size:
            tokens[tok_id] = tok_str

    # Added/special tokens
    for entry in tokenizer_json.get("added_tokens", []):
        tok_id = entry["id"]
        tok_str = entry["content"]
        if 0 <= tok_id < vocab_size:
            tokens[tok_id] = tok_str

    # Find gaps — fill with placeholder. Some HF tokenizers leave gaps for
    # reserved IDs; for ds4.c lookups to work, every required special token
    # must be present at its expected ID.
    n_missing = sum(1 for t in tokens if t is None)
    if n_missing:
        # Don't silently fill with empty strings; the engine looks up specific
        # token strings (bos/eos/etc) and will fail loud if a special is missing.
        # But for general slots, a placeholder is fine — engine never indexes
        # gaps that don't appear in BPE merges or special lookups.
        for i, t in enumerate(tokens):
            if t is None:
                tokens[i] = f"<|reserved_{i}|>"
        print(f"  filled {n_missing} gap slots with placeholders", file=sys.stderr)

    # Sanity check: required ds4.c specials must resolve
    required_specials = [
        "<｜begin▁of▁sentence｜>",
        "<｜end▁of▁sentence｜>",
        "<｜User｜>",
        "<｜Assistant｜>",
        "<think>",
        "</think>",
        "｜DSML｜",
    ]
    by_str = {t: i for i, t in enumerate(tokens)}
    missing = [s for s in required_specials if s not in by_str]
    if missing:
        raise ValueError(f"required ds4.c special tokens not found in vocab: {missing}")

    return tokens


def build_merges_array(tokenizer_json: dict) -> list[str]:
    """HF tokenizer.json stores merges as list of [str, str] pairs in priority
    order. GGUF format stores them as single 'left right' strings, in the
    same priority order. Engine uses index as rank.
    """
    raw = tokenizer_json["model"]["merges"]
    merges = []
    for m in raw:
        if isinstance(m, list):
            # New HF format: ["left", "right"]
            merges.append(f"{m[0]} {m[1]}")
        elif isinstance(m, str):
            # Old HF format: "left right"
            merges.append(m)
        else:
            raise ValueError(f"unexpected merge format: {type(m).__name__}: {m!r}")
    return merges


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir",
                    default="/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash",
                    help="HuggingFace model dir (must contain config.json + tokenizer.json)")
    ap.add_argument("--out",
                    default="/Users/silv/cl/tlp/montyneg/ivp5_ds4/ds4_metadata.gguf",
                    help="Output GGUF path (KV-only, tensor_count=0)")
    args = ap.parse_args()

    src = Path(args.src_dir)
    out = Path(args.out)

    # Refuse to touch any minimal-GGUF — explicit silv directive.
    for p in [src, out]:
        if "minimal" in p.name.lower() and p.suffix == ".gguf":
            raise SystemExit(f"refusing to read/write minimal-GGUF: {p}")

    cfg_path = src / "config.json"
    tok_path = src / "tokenizer.json"
    if not cfg_path.exists() or not tok_path.exists():
        raise SystemExit(f"missing required raw files in {src}")

    print(f"src dir:   {src}")
    print(f"out file:  {out}")
    print()

    cfg = json.loads(cfg_path.read_text())
    tok = json.loads(tok_path.read_text())

    print(f"config.json:    {cfg.get('num_hidden_layers', '?')} layers, "
          f"vocab_size={cfg.get('vocab_size', '?')}, "
          f"experts={cfg.get('n_routed_experts', '?')}")
    print(f"tokenizer.json: vocab={len(tok['model']['vocab'])}, "
          f"merges={len(tok['model']['merges'])}, "
          f"added={len(tok.get('added_tokens', []))}")

    out.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFMetadataWriter(str(out))

    print("\n[1/3] Writing arch KV...")
    emit_arch_kv(w, cfg)

    print("[2/3] Building token table...")
    vocab_size = cfg["vocab_size"]
    tokens = build_token_array(tok, vocab_size)
    print(f"  tokens[{len(tokens)}] built (avg len {sum(len(t) for t in tokens)/len(tokens):.1f} bytes)")
    w.add("tokenizer.ggml.tokens", GGUF_TYPE_ARRAY, (GGUF_TYPE_STRING, tokens))

    print("[3/3] Building merges table...")
    merges = build_merges_array(tok)
    print(f"  merges[{len(merges)}] built")
    w.add("tokenizer.ggml.merges", GGUF_TYPE_ARRAY, (GGUF_TYPE_STRING, merges))

    # Optional: tokenizer.ggml.model (declares BPE)
    w.add("tokenizer.ggml.model", GGUF_TYPE_STRING, "gpt2")  # closest match; ds4.c does not validate this

    print(f"\nWriting {out}...")
    w.write()

    sz = out.stat().st_size
    print(f"  wrote {sz} bytes ({sz/1e6:.2f} MB)")
    print(f"\nKV count: {len(w.kv)}")
    print("Tensor count: 0 (metadata-only — tensor data must come from packs)")

    print(f"\nNext step (#771): engine_open surgery to consume this bundle")
    print(f"  alongside the two packs. The minimal-GGUF can then be retired.")


if __name__ == "__main__":
    main()
