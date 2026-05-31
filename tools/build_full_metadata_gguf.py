#!/usr/bin/env python3
"""Build a FULL metadata GGUF for DS4 V4 Flash — KV + tokenizer + tensor manifest
— directly from the raw HuggingFace model dir and the existing packs.

silv 2026-05-28: "generate your missing pieces directly from the safetensors
files of the original model" + "do NOT take anything from the minimal-gguf,
but only from the raw data from the original model".

Difference from build_metadata_gguf.py (4.73 MB, KV-only):
  This tool ALSO emits the tensor info section (names + dims + GGUF dtypes
  + zero-byte data offsets). Tensor count > 0 in the header; the GGUF
  reader's tensor table now mirrors what minimal-GGUF would have provided
  for shape/dtype validation, but ZERO actual tensor data bytes.

Engine consumption (next step, not yet wired):
  parse_tensors must accept `t->bytes != 0 && tensor_data_pos == file_size`
  (data section absent). Storage MUST be set for every tensor before any
  kernel dispatch, via override-fill from the two packs. The minimal-GGUF
  9 GB body of dead F16/Q8_0 fallback data drops entirely.

Sources read (ONLY from raw):
  - /Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash/config.json
  - /Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash/tokenizer.json
  - /Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash/tokenizer_config.json
  - /Users/silv/cl/tlp/montyneg/ds4/nonrouted/ds4v4_nonrouted.pack   (manifest only — for non-routed tensor list)
  - /Users/silv/cl/tlp/montyneg/ds4/vqb2/.../*.vqb2pack.index.csv    (for routed FFN coverage)

Output:
  ds4_metadata_full.gguf (~5-7 MB; tensor_count > 0; zero bytes of tensor data)

NB: this tool REFUSES to read any file with `minimal` in its name.
"""

import argparse
import json
import re
import struct
import sys
from pathlib import Path

# ============================================================================
# GGUF constants
# ============================================================================

GGUF_MAGIC = 0x46554747
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

# Tensor dtypes (ggml type codes — match the engine's enum in ds4.c line 913-929).
# silv 2026-05-28: declare UPSTREAM dtypes (what DeepSeek-V4-Flash actually ships:
# BF16 + F8_E4M3 + F32), NOT the engine's internal Q8_0/F16/IQ2_XXS compression.
# The pack carries upstream bytes verbatim; engine's storage-override + Cycle 5
# substitution dispatches BF16/FP8 kernels.
GGUF_TENSOR_F32      = 0
GGUF_TENSOR_F16      = 1
GGUF_TENSOR_Q8_0     = 8
GGUF_TENSOR_IQ2_XXS  = 16
GGUF_TENSOR_I8       = 24
GGUF_TENSOR_I32      = 26
GGUF_TENSOR_BF16     = 30
GGUF_TENSOR_FP8_E4M3 = 64  # ds4-local (above GGUF spec range)
GGUF_TENSOR_FP8_E8M0 = 65  # ds4-local — scale-pair, not standalone declared

# ============================================================================
# safetensors name → GGUF name (ported from pack_to_gguf.py, silv 2026-05-28)
# ============================================================================

TOP_NAMES = {
    "embed.weight":      "token_embd.weight",
    "head.weight":       "output.weight",
    "norm.weight":       "output_norm.weight",
    "hc_head_base":      "output_hc_base.weight",
    "hc_head_fn":        "output_hc_fn.weight",
    "hc_head_scale":     "output_hc_scale.weight",
}

LAYER_NAMES = {
    "layers.{X}.attn_norm.weight":      "blk.{X}.attn_norm.weight",
    "layers.{X}.ffn_norm.weight":       "blk.{X}.ffn_norm.weight",
    "layers.{X}.attn.wq_a.weight":      "blk.{X}.attn_q_a.weight",
    "layers.{X}.attn.q_norm.weight":    "blk.{X}.attn_q_a_norm.weight",
    "layers.{X}.attn.wq_b.weight":      "blk.{X}.attn_q_b.weight",
    "layers.{X}.attn.wkv.weight":       "blk.{X}.attn_kv.weight",
    "layers.{X}.attn.kv_norm.weight":   "blk.{X}.attn_kv_a_norm.weight",
    "layers.{X}.attn.attn_sink":        "blk.{X}.attn_sinks.weight",
    "layers.{X}.attn.wo_a.weight":      "blk.{X}.attn_output_a.weight",
    "layers.{X}.attn.wo_b.weight":      "blk.{X}.attn_output_b.weight",
    "layers.{X}.hc_attn_base":          "blk.{X}.hc_attn_base.weight",
    "layers.{X}.hc_attn_fn":            "blk.{X}.hc_attn_fn.weight",
    "layers.{X}.hc_attn_scale":         "blk.{X}.hc_attn_scale.weight",
    "layers.{X}.hc_ffn_base":           "blk.{X}.hc_ffn_base.weight",
    "layers.{X}.hc_ffn_fn":             "blk.{X}.hc_ffn_fn.weight",
    "layers.{X}.hc_ffn_scale":          "blk.{X}.hc_ffn_scale.weight",
    "layers.{X}.ffn.gate.weight":       "blk.{X}.ffn_gate_inp.weight",
    "layers.{X}.ffn.gate.bias":         "blk.{X}.exp_probs_b.bias",
    "layers.{X}.ffn.gate.tid2eid":      "blk.{X}.ffn_gate_tid2eid.weight",
    "layers.{X}.ffn.shared_experts.w1.weight": "blk.{X}.ffn_gate_shexp.weight",
    "layers.{X}.ffn.shared_experts.w2.weight": "blk.{X}.ffn_down_shexp.weight",
    "layers.{X}.ffn.shared_experts.w3.weight": "blk.{X}.ffn_up_shexp.weight",
    "layers.{X}.attn.compressor.wkv.weight":   "blk.{X}.attn_compressor_kv.weight",
    "layers.{X}.attn.compressor.wgate.weight": "blk.{X}.attn_compressor_gate.weight",
    "layers.{X}.attn.compressor.norm.weight":  "blk.{X}.attn_compressor_norm.weight",
    "layers.{X}.attn.compressor.ape":          "blk.{X}.attn_compressor_ape.weight",
    "layers.{X}.attn.indexer.wq_b.weight":            "blk.{X}.indexer.attn_q_b.weight",
    "layers.{X}.attn.indexer.weights_proj.weight":    "blk.{X}.indexer.proj.weight",
    "layers.{X}.attn.indexer.compressor.wkv.weight":  "blk.{X}.indexer_compressor_kv.weight",
    "layers.{X}.attn.indexer.compressor.wgate.weight":"blk.{X}.indexer_compressor_gate.weight",
    "layers.{X}.attn.indexer.compressor.norm.weight": "blk.{X}.indexer_compressor_norm.weight",
    "layers.{X}.attn.indexer.compressor.ape":         "blk.{X}.indexer_compressor_ape.weight",
}

MTP_NAMES = {
    "mtp.0.attn_norm.weight":       "mtp.0.attn_norm.weight",
    "mtp.0.ffn_norm.weight":        "mtp.0.ffn_norm.weight",
    "mtp.0.attn.wq_a.weight":       "mtp.0.attn_q_a.weight",
    "mtp.0.attn.q_norm.weight":     "mtp.0.attn_q_a_norm.weight",
    "mtp.0.attn.wq_b.weight":       "mtp.0.attn_q_b.weight",
    "mtp.0.attn.wkv.weight":        "mtp.0.attn_kv.weight",
    "mtp.0.attn.kv_norm.weight":    "mtp.0.attn_kv_a_norm.weight",
    "mtp.0.attn.attn_sink":         "mtp.0.attn_sinks.weight",
    "mtp.0.attn.wo_a.weight":       "mtp.0.attn_output_a.weight",
    "mtp.0.attn.wo_b.weight":       "mtp.0.attn_output_b.weight",
    "mtp.0.hc_attn_base":           "mtp.0.hc_attn_base.weight",
    "mtp.0.hc_attn_fn":             "mtp.0.hc_attn_fn.weight",
    "mtp.0.hc_attn_scale":          "mtp.0.hc_attn_scale.weight",
    "mtp.0.hc_ffn_base":            "mtp.0.hc_ffn_base.weight",
    "mtp.0.hc_ffn_fn":              "mtp.0.hc_ffn_fn.weight",
    "mtp.0.hc_ffn_scale":           "mtp.0.hc_ffn_scale.weight",
    "mtp.0.hc_head_base":           "mtp.0.hc_head_base.weight",
    "mtp.0.hc_head_fn":             "mtp.0.hc_head_fn.weight",
    "mtp.0.hc_head_scale":          "mtp.0.hc_head_scale.weight",
    "mtp.0.ffn.gate.weight":        "mtp.0.ffn_gate_inp.weight",
    "mtp.0.ffn.gate.bias":          "mtp.0.exp_probs_b.bias",
    "mtp.0.ffn.shared_experts.w1.weight": "mtp.0.ffn_gate_shexp.weight",
    "mtp.0.ffn.shared_experts.w2.weight": "mtp.0.ffn_down_shexp.weight",
    "mtp.0.ffn.shared_experts.w3.weight": "mtp.0.ffn_up_shexp.weight",
    "mtp.0.attn.compressor.wkv.weight":   "mtp.0.attn_compressor_kv.weight",
    "mtp.0.attn.compressor.wgate.weight": "mtp.0.attn_compressor_gate.weight",
    "mtp.0.attn.compressor.norm.weight":  "mtp.0.attn_compressor_norm.weight",
    "mtp.0.attn.compressor.ape":          "mtp.0.attn_compressor_ape.weight",
    "mtp.0.attn.indexer.wq_b.weight":            "mtp.0.indexer.attn_q_b.weight",
    "mtp.0.attn.indexer.weights_proj.weight":    "mtp.0.indexer.proj.weight",
    "mtp.0.attn.indexer.compressor.wkv.weight":  "mtp.0.indexer_compressor_kv.weight",
    "mtp.0.attn.indexer.compressor.wgate.weight":"mtp.0.indexer_compressor_gate.weight",
    "mtp.0.attn.indexer.compressor.norm.weight": "mtp.0.indexer_compressor_norm.weight",
    "mtp.0.attn.indexer.compressor.ape":         "mtp.0.indexer_compressor_ape.weight",
    "mtp.0.e_proj.weight":   "mtp.0.e_proj.weight",
    "mtp.0.h_proj.weight":   "mtp.0.h_proj.weight",
    "mtp.0.enorm.weight":    "mtp.0.enorm.weight",
    "mtp.0.hnorm.weight":    "mtp.0.hnorm.weight",
    "mtp.0.norm.weight":     "mtp.0.norm.weight",
}


def translate_name(st_name: str) -> str | None:
    """safetensors → GGUF tensor name. None = unmapped (routed expert, scale-pair, etc.)."""
    if st_name in TOP_NAMES:  return TOP_NAMES[st_name]
    if st_name in MTP_NAMES:  return MTP_NAMES[st_name]
    # .scale companions stay paired with their .weight in source-exact dispatch;
    # they're not standalone engine tensors.
    if st_name.endswith(".scale"):  return None
    if ".ffn.experts." in st_name:  return None  # routed FFN — handled via VQB2

    # Per-layer regex match
    m = re.match(r"^layers\.(\d+)\.(.+)$", st_name)
    if m:
        idx, rest = m.group(1), m.group(2)
        key = f"layers.{{X}}.{rest}"
        if key in LAYER_NAMES:  return LAYER_NAMES[key].format(X=idx)
    return None


# ============================================================================
# Engine-expected GGUF dtype map (extracted from ds4.c tensor_expect_layout +
# mtp_weights_validate_layout, silv 2026-05-28 #771)
# ============================================================================

# Per-layer tensors (suffix after `blk.X.` or `mtp.0.`). The engine validates
# t->type ≡ this dtype at model load. Pack source dtype is independent —
# storage override (Cycle 5) handles BF16/FP8 → engine-declared mismatches at
# runtime via tensor_effective_type().
LAYER_EXPECTED_DTYPE = {
    "attn_compressor_ape.weight":      GGUF_TENSOR_F16,
    "attn_compressor_gate.weight":     GGUF_TENSOR_F16,
    "attn_compressor_kv.weight":       GGUF_TENSOR_F16,
    "attn_compressor_norm.weight":     GGUF_TENSOR_F32,
    "attn_kv_a_norm.weight":           GGUF_TENSOR_F32,
    "attn_kv.weight":                  GGUF_TENSOR_Q8_0,
    "attn_norm.weight":                GGUF_TENSOR_F32,
    "attn_output_a.weight":            GGUF_TENSOR_Q8_0,
    "attn_output_b.weight":            GGUF_TENSOR_Q8_0,
    "attn_q_a_norm.weight":            GGUF_TENSOR_F32,
    "attn_q_a.weight":                 GGUF_TENSOR_Q8_0,
    "attn_q_b.weight":                 GGUF_TENSOR_Q8_0,
    "attn_sinks.weight":               GGUF_TENSOR_F32,
    "ffn_down_shexp.weight":           GGUF_TENSOR_Q8_0,
    "exp_probs_b.bias":                GGUF_TENSOR_F32,
    "ffn_gate_inp.weight":             GGUF_TENSOR_F16,
    "ffn_gate_shexp.weight":           GGUF_TENSOR_Q8_0,
    "ffn_gate_tid2eid.weight":         GGUF_TENSOR_I32,
    "ffn_norm.weight":                 GGUF_TENSOR_F32,
    "ffn_up_shexp.weight":             GGUF_TENSOR_Q8_0,
    "hc_attn_base.weight":             GGUF_TENSOR_F32,
    "hc_attn_fn.weight":               GGUF_TENSOR_F16,
    "hc_attn_scale.weight":            GGUF_TENSOR_F32,
    "hc_ffn_base.weight":              GGUF_TENSOR_F32,
    "hc_ffn_fn.weight":                GGUF_TENSOR_F16,
    "hc_ffn_scale.weight":             GGUF_TENSOR_F32,
    "indexer.attn_q_b.weight":         GGUF_TENSOR_F16,
    "indexer_compressor_ape.weight":   GGUF_TENSOR_F16,
    "indexer_compressor_gate.weight":  GGUF_TENSOR_F16,
    "indexer_compressor_kv.weight":    GGUF_TENSOR_F16,
    "indexer_compressor_norm.weight":  GGUF_TENSOR_F32,
    "indexer.proj.weight":             GGUF_TENSOR_F16,
}

# Top-level (no `blk.X.` or `mtp.0.` prefix).
TOP_EXPECTED_DTYPE = {
    "token_embd.weight":               GGUF_TENSOR_F16,
    "output.weight":                   GGUF_TENSOR_Q8_0,
    "output_norm.weight":              GGUF_TENSOR_F32,
    "output_hc_base.weight":           GGUF_TENSOR_F32,
    "output_hc_fn.weight":             GGUF_TENSOR_F16,
    "output_hc_scale.weight":          GGUF_TENSOR_F32,
}

# MTP-only fields not covered by LAYER_EXPECTED_DTYPE (MTP block reuses many
# layer-tensor names; only these are MTP-specific).
MTP_ONLY_EXPECTED_DTYPE = {
    "e_proj.weight":                   GGUF_TENSOR_Q8_0,
    "h_proj.weight":                   GGUF_TENSOR_Q8_0,
    "enorm.weight":                    GGUF_TENSOR_F32,
    "hnorm.weight":                    GGUF_TENSOR_F32,
    "norm.weight":                     GGUF_TENSOR_F32,
    "hc_head_base.weight":             GGUF_TENSOR_F32,
    "hc_head_fn.weight":               GGUF_TENSOR_F16,
    "hc_head_scale.weight":            GGUF_TENSOR_F32,
}


def lookup_expected_dtype(gguf_name: str) -> int | None:
    """Return the engine-expected GGUF dtype for a given tensor name, or None
    if the name is unknown (caller falls back to pack-dtype-derived guess).
    """
    # Top-level (no per-layer prefix)
    if gguf_name in TOP_EXPECTED_DTYPE:
        return TOP_EXPECTED_DTYPE[gguf_name]
    # Per-layer: strip `blk.<N>.` and look up the suffix
    m = re.match(r"^blk\.\d+\.(.+)$", gguf_name)
    if m:
        suffix = m.group(1)
        if suffix in LAYER_EXPECTED_DTYPE:
            return LAYER_EXPECTED_DTYPE[suffix]
    # MTP block: strip `mtp.0.` and check LAYER + MTP-only tables
    m = re.match(r"^mtp\.\d+\.(.+)$", gguf_name)
    if m:
        suffix = m.group(1)
        if suffix in LAYER_EXPECTED_DTYPE:
            return LAYER_EXPECTED_DTYPE[suffix]
        if suffix in MTP_ONLY_EXPECTED_DTYPE:
            return MTP_ONLY_EXPECTED_DTYPE[suffix]
    return None


# Pack source dtype → engine GGUF type. UPSTREAM-preserving (silv 2026-05-28):
# declare what's actually in the pack, not what minimal-GGUF would have
# compressed to.
def pack_dtype_to_gguf(pack_dtype: str, name: str) -> int:
    if pack_dtype == "F32":      return GGUF_TENSOR_F32
    if pack_dtype == "BF16":     return GGUF_TENSOR_BF16
    if pack_dtype == "F16":      return GGUF_TENSOR_F16
    if pack_dtype == "F8_E4M3":  return GGUF_TENSOR_FP8_E4M3
    if pack_dtype == "I64":      return GGUF_TENSOR_I32
    if pack_dtype == "I8":       return GGUF_TENSOR_I8
    raise ValueError(f"unknown pack dtype {pack_dtype!r} for {name!r}")


# ============================================================================
# GGUF metadata + tensor-manifest writer
# ============================================================================

class GGUFFullManifestWriter:
    def __init__(self, path: str):
        self.f = open(path, "wb")
        self.kv = []
        self.tensors = []  # list of dicts: name, dtype, dims, declared_bytes

    def add_kv(self, key, vtype, value):
        self.kv.append((key, vtype, value))

    def add_tensor(self, name: str, dtype: int, dims: list[int]):
        n_elem = 1
        for d in dims: n_elem *= d
        if dtype == GGUF_TENSOR_F32:   bytes_n = n_elem * 4
        elif dtype == GGUF_TENSOR_F16: bytes_n = n_elem * 2
        elif dtype == GGUF_TENSOR_BF16: bytes_n = n_elem * 2
        elif dtype == GGUF_TENSOR_FP8_E4M3: bytes_n = n_elem  # 1 byte per elem
        elif dtype == GGUF_TENSOR_FP8_E8M0: bytes_n = n_elem  # 1 byte per elem
        elif dtype == GGUF_TENSOR_Q8_0:
            assert n_elem % 32 == 0, f"{name}: Q8_0 needs n_elem%32==0, got {n_elem}"
            bytes_n = (n_elem // 32) * 34
        elif dtype == GGUF_TENSOR_IQ2_XXS:
            assert n_elem % 256 == 0, f"{name}: IQ2_XXS needs n_elem%256==0, got {n_elem}"
            bytes_n = (n_elem // 256) * 66
        elif dtype == GGUF_TENSOR_I32: bytes_n = n_elem * 4
        elif dtype == GGUF_TENSOR_I8:  bytes_n = n_elem * 1
        else:
            raise ValueError(f"unsupported gguf dtype {dtype} for {name}")
        self.tensors.append({"name": name, "dtype": dtype, "dims": dims, "bytes": bytes_n})

    def _w_str(self, s: str):
        b = s.encode("utf-8")
        self.f.write(struct.pack("<Q", len(b)))
        self.f.write(b)

    def _w_value(self, vtype, value):
        if   vtype == GGUF_TYPE_UINT32:  self.f.write(struct.pack("<I", value))
        elif vtype == GGUF_TYPE_INT32:   self.f.write(struct.pack("<i", value))
        elif vtype == GGUF_TYPE_UINT64:  self.f.write(struct.pack("<Q", value))
        elif vtype == GGUF_TYPE_INT64:   self.f.write(struct.pack("<q", value))
        elif vtype == GGUF_TYPE_FLOAT32: self.f.write(struct.pack("<f", value))
        elif vtype == GGUF_TYPE_FLOAT64: self.f.write(struct.pack("<d", value))
        elif vtype == GGUF_TYPE_BOOL:    self.f.write(struct.pack("<B", 1 if value else 0))
        elif vtype == GGUF_TYPE_STRING:  self._w_str(value)
        elif vtype == GGUF_TYPE_UINT8:   self.f.write(struct.pack("<B", value))
        elif vtype == GGUF_TYPE_INT8:    self.f.write(struct.pack("<b", value))
        elif vtype == GGUF_TYPE_ARRAY:
            etype, values = value
            self.f.write(struct.pack("<I", etype))
            self.f.write(struct.pack("<Q", len(values)))
            for v in values: self._w_value(etype, v)
        else:
            raise ValueError(f"unsupported value type {vtype}")

    def write(self):
        # Header
        self.f.write(struct.pack("<I", GGUF_MAGIC))
        self.f.write(struct.pack("<I", GGUF_VERSION))
        self.f.write(struct.pack("<Q", len(self.tensors)))
        self.f.write(struct.pack("<Q", len(self.kv)))

        # KV
        for key, vtype, value in self.kv:
            self._w_str(key)
            self.f.write(struct.pack("<I", vtype))
            self._w_value(vtype, value)

        # Compute aligned offsets (relative to data start)
        offset = 0
        for t in self.tensors:
            t["offset"] = offset
            offset += t["bytes"]
            offset = (offset + GGUF_DEFAULT_ALIGNMENT - 1) & ~(GGUF_DEFAULT_ALIGNMENT - 1)
        declared_data_bytes = offset

        # Tensor info entries
        for t in self.tensors:
            self._w_str(t["name"])
            self.f.write(struct.pack("<I", len(t["dims"])))
            for d in t["dims"]:
                self.f.write(struct.pack("<Q", d))
            self.f.write(struct.pack("<I", t["dtype"]))
            self.f.write(struct.pack("<Q", t["offset"]))

        # Padding to alignment, then file ends. NO actual tensor data section.
        pos = self.f.tell()
        pad = (GGUF_DEFAULT_ALIGNMENT - (pos % GGUF_DEFAULT_ALIGNMENT)) % GGUF_DEFAULT_ALIGNMENT
        self.f.write(b"\x00" * pad)

        self.f.close()
        return declared_data_bytes


# ============================================================================
# Arch KV (35 scalars + 2 arrays + 2 general — derived from config.json)
# ============================================================================

def emit_arch_kv(w, cfg):
    w.add_kv("general.architecture", GGUF_TYPE_STRING, "deepseek4")
    w.add_kv("general.name",         GGUF_TYPE_STRING, "DeepSeek-V4-Flash")

    w.add_kv("deepseek4.block_count",            GGUF_TYPE_UINT32, cfg["num_hidden_layers"])
    w.add_kv("deepseek4.context_length",         GGUF_TYPE_UINT64, cfg["max_position_embeddings"])
    w.add_kv("deepseek4.embedding_length",       GGUF_TYPE_UINT32, cfg["hidden_size"])
    w.add_kv("deepseek4.vocab_size",             GGUF_TYPE_UINT32, cfg["vocab_size"])
    w.add_kv("deepseek4.attention.head_count",   GGUF_TYPE_UINT32, cfg["num_attention_heads"])
    w.add_kv("deepseek4.attention.head_count_kv",GGUF_TYPE_UINT32, cfg["num_key_value_heads"])
    w.add_kv("deepseek4.attention.key_length",   GGUF_TYPE_UINT32, cfg["head_dim"])
    w.add_kv("deepseek4.attention.value_length", GGUF_TYPE_UINT32, cfg.get("v_head_dim", cfg["head_dim"]))
    w.add_kv("deepseek4.attention.sliding_window", GGUF_TYPE_UINT32, cfg["sliding_window"])
    w.add_kv("deepseek4.attention.q_lora_rank",      GGUF_TYPE_UINT32, cfg["q_lora_rank"])
    w.add_kv("deepseek4.attention.output_lora_rank", GGUF_TYPE_UINT32, cfg["o_lora_rank"])
    w.add_kv("deepseek4.attention.output_group_count", GGUF_TYPE_UINT32, cfg["o_groups"])
    w.add_kv("deepseek4.attention.indexer.head_count", GGUF_TYPE_UINT32, cfg["index_n_heads"])
    w.add_kv("deepseek4.attention.indexer.key_length", GGUF_TYPE_UINT32, cfg["index_head_dim"])
    w.add_kv("deepseek4.attention.indexer.top_k",      GGUF_TYPE_UINT32, cfg["index_topk"])
    w.add_kv("deepseek4.expert_count",             GGUF_TYPE_UINT32, cfg["n_routed_experts"])
    w.add_kv("deepseek4.expert_used_count",        GGUF_TYPE_UINT32, cfg["num_experts_per_tok"])
    w.add_kv("deepseek4.expert_feed_forward_length", GGUF_TYPE_UINT32, cfg["moe_intermediate_size"])
    w.add_kv("deepseek4.expert_shared_count",      GGUF_TYPE_UINT32, cfg["n_shared_experts"])
    w.add_kv("deepseek4.expert_weights_scale",     GGUF_TYPE_FLOAT32, cfg["routed_scaling_factor"])
    w.add_kv("deepseek4.expert_weights_norm",      GGUF_TYPE_BOOL, cfg["norm_topk_prob"])
    # NB: DS4 V4 Flash does NOT use expert grouping (engine expects 0/0). The
    # inference reference's `n_group=8`/`topk_group=4` describe a DS V3 variant.
    # Validated against ds4.c config_expect_u32 at line 3368-3369.
    w.add_kv("deepseek4.expert_group_count",       GGUF_TYPE_UINT32, 0)
    w.add_kv("deepseek4.expert_group_used_count",  GGUF_TYPE_UINT32, 0)
    w.add_kv("deepseek4.hash_layer_count",         GGUF_TYPE_UINT32, cfg["num_hash_layers"])
    w.add_kv("deepseek4.hyper_connection.count",   GGUF_TYPE_UINT32, cfg["hc_mult"])
    w.add_kv("deepseek4.hyper_connection.sinkhorn_iterations", GGUF_TYPE_UINT32, cfg["hc_sinkhorn_iters"])
    w.add_kv("deepseek4.hyper_connection.epsilon",  GGUF_TYPE_FLOAT32, cfg["hc_eps"])
    w.add_kv("deepseek4.rope.dimension_count",      GGUF_TYPE_UINT32, cfg["qk_rope_head_dim"])
    w.add_kv("deepseek4.rope.scaling.original_context_length", GGUF_TYPE_UINT64,
             cfg["rope_scaling"]["original_max_position_embeddings"])
    w.add_kv("deepseek4.rope.freq_base",            GGUF_TYPE_FLOAT32, cfg["rope_theta"])
    w.add_kv("deepseek4.rope.scaling.factor",       GGUF_TYPE_FLOAT32, cfg["rope_scaling"]["factor"])
    w.add_kv("deepseek4.rope.scaling.yarn_beta_fast", GGUF_TYPE_FLOAT32, cfg["rope_scaling"]["beta_fast"])
    w.add_kv("deepseek4.rope.scaling.yarn_beta_slow", GGUF_TYPE_FLOAT32, cfg["rope_scaling"]["beta_slow"])
    w.add_kv("deepseek4.attention.compress_rope_freq_base", GGUF_TYPE_FLOAT32, cfg["compress_rope_theta"])
    w.add_kv("deepseek4.attention.layer_norm_rms_epsilon",  GGUF_TYPE_FLOAT32, cfg["rms_norm_eps"])

    n_layer = cfg["num_hidden_layers"]
    cr = list(cfg["compress_ratios"])
    if len(cr) < n_layer:
        raise ValueError(f"compress_ratios shorter than {n_layer}")
    w.add_kv("deepseek4.attention.compress_ratios",
             GGUF_TYPE_ARRAY, (GGUF_TYPE_UINT32, [int(x) for x in cr]))
    sc = float(cfg["swiglu_limit"])
    w.add_kv("deepseek4.swiglu_clamp_exp",
             GGUF_TYPE_ARRAY, (GGUF_TYPE_FLOAT32, [sc] * n_layer))


def emit_tokenizer_kv(w, tok, vocab_size):
    # Tokens
    tokens = [None] * vocab_size
    for tok_str, tok_id in tok["model"]["vocab"].items():
        if 0 <= tok_id < vocab_size:
            tokens[tok_id] = tok_str
    for entry in tok.get("added_tokens", []):
        tok_id, tok_str = entry["id"], entry["content"]
        if 0 <= tok_id < vocab_size:
            tokens[tok_id] = tok_str
    n_missing = sum(1 for t in tokens if t is None)
    if n_missing:
        for i, t in enumerate(tokens):
            if t is None: tokens[i] = f"<|reserved_{i}|>"
        print(f"  filled {n_missing} reserved-slot placeholders", file=sys.stderr)

    required = ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>",
                "<｜User｜>", "<｜Assistant｜>",
                "<think>", "</think>", "｜DSML｜"]
    by_str = {t: i for i, t in enumerate(tokens)}
    missing = [s for s in required if s not in by_str]
    if missing:
        raise ValueError(f"missing engine specials in vocab: {missing}")

    # Merges
    raw = tok["model"]["merges"]
    merges = []
    for m in raw:
        if isinstance(m, list):  merges.append(f"{m[0]} {m[1]}")
        elif isinstance(m, str): merges.append(m)
        else:                    raise ValueError(f"bad merge: {m!r}")

    w.add_kv("tokenizer.ggml.model",  GGUF_TYPE_STRING, "gpt2")
    w.add_kv("tokenizer.ggml.tokens", GGUF_TYPE_ARRAY, (GGUF_TYPE_STRING, tokens))
    w.add_kv("tokenizer.ggml.merges", GGUF_TYPE_ARRAY, (GGUF_TYPE_STRING, merges))


# ============================================================================
# Tensor manifest emission — from pack manifests
# ============================================================================

def read_nrpk_manifest(pack_path: Path):
    with open(pack_path, "rb") as f:
        f.seek(16)
        manifest_off, manifest_bytes = struct.unpack("<QQ", f.read(16))
        f.seek(manifest_off)
        blob = f.read(manifest_bytes).rstrip(b"\x00")
    return json.loads(blob)


def emit_nonrouted_tensors(w, nrpk_entries):
    """For each non-routed pack tensor, emit the matching engine GGUF tensor info."""
    emitted = 0
    skipped_scale = 0
    skipped_unmapped = []

    for e in nrpk_entries:
        st_name = e["name"]
        if st_name.endswith(".scale"):
            skipped_scale += 1
            continue
        gguf_name = translate_name(st_name)
        if gguf_name is None:
            skipped_unmapped.append(st_name)
            continue
        pack_dtype = e["dtype"]
        # silv 2026-05-28 dual-source framing:
        #   Pack = upstream-exact weight bytes (BF16/FP8/F32, what HF ships).
        #   Metadata GGUF = engine-validation dtypes (F16/F32/Q8_0, what
        #     tensor_expect_layout enforces in ds4.c).
        # Producer uses engine-expected dtype when known (extracted from the
        # 62 tensor_expect_layout calls); falls back to pack-source dtype
        # when no engine expectation is set. Override-fill bridges the two:
        # pack BF16/F32 → engine F16 via inline conversion. Q8_0-target
        # tensors still use BF16/FP8 source-exact storage (Cycle 5 wired).
        gguf_dtype = lookup_expected_dtype(gguf_name)
        if gguf_dtype is None:
            gguf_dtype = pack_dtype_to_gguf(pack_dtype, st_name)
        # safetensors PyTorch convention is row-major (outermost-first).
        # GGUF is column-major (innermost-first). Reverse for storage.
        dims = list(reversed(e["shape"]))

        # Q8_0 requires n_elem % 32 == 0; if not, fall back to F32 (the engine
        # won't see this on the hot path for tid2eid).
        n_elem = 1
        for d in dims: n_elem *= d
        if gguf_dtype == GGUF_TENSOR_Q8_0 and n_elem % 32 != 0:
            gguf_dtype = GGUF_TENSOR_F32

        w.add_tensor(gguf_name, gguf_dtype, dims)
        emitted += 1

    return emitted, skipped_scale, skipped_unmapped


def emit_routed_tensors(w, cfg):
    """Emit routed FFN tensors with UPSTREAM dtype = FP8_E4M3.

    silv 2026-05-28: IQ2_XXS is legacy (the minimal-GGUF convention from
    silv's pre-VQB2 era). DS4 V4 Flash's routed FFN is FP8_E4M3 per the
    upstream config.json:
        "quantization_config": {
            "fmt": "e4m3", "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128], ...
        }
    The runtime path is VQB2-coded packs (#750-#761) which carry their own
    manifest; the GGUF declaration here only serves engine-side tensor-table
    validation. Storage override + the VQB2 pack provide the actual bytes.
    """
    n_layer  = cfg["num_hidden_layers"]
    n_expert = cfg["n_routed_experts"]
    n_ff     = cfg["moe_intermediate_size"]
    n_embd   = cfg["hidden_size"]

    for il in range(n_layer):
        w.add_tensor(f"blk.{il}.ffn_gate_exps.weight", GGUF_TENSOR_FP8_E4M3,
                     [n_embd, n_ff, n_expert])
        w.add_tensor(f"blk.{il}.ffn_up_exps.weight",   GGUF_TENSOR_FP8_E4M3,
                     [n_embd, n_ff, n_expert])
        w.add_tensor(f"blk.{il}.ffn_down_exps.weight", GGUF_TENSOR_FP8_E4M3,
                     [n_ff, n_embd, n_expert])

    w.add_tensor("mtp.0.ffn_gate_exps.weight", GGUF_TENSOR_FP8_E4M3, [n_embd, n_ff, n_expert])
    w.add_tensor("mtp.0.ffn_up_exps.weight",   GGUF_TENSOR_FP8_E4M3, [n_embd, n_ff, n_expert])
    w.add_tensor("mtp.0.ffn_down_exps.weight", GGUF_TENSOR_FP8_E4M3, [n_ff, n_embd, n_expert])

    return n_layer * 3 + 3


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir",
                    default="/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash")
    ap.add_argument("--nonrouted-pack",
                    default="/Users/silv/cl/tlp/montyneg/ds4/nonrouted/ds4v4_nonrouted.pack")
    ap.add_argument("--out",
                    default="/Users/silv/cl/tlp/montyneg/ivp5_ds4/ds4_metadata_full.gguf")
    args = ap.parse_args()

    src = Path(args.src_dir)
    pack = Path(args.nonrouted_pack)
    out = Path(args.out)

    for p in [src, pack, out]:
        if "minimal" in p.name.lower():
            raise SystemExit(f"refusing to touch minimal-* artifacts: {p}")

    cfg = json.loads((src / "config.json").read_text())
    tok = json.loads((src / "tokenizer.json").read_text())
    print(f"config: layers={cfg['num_hidden_layers']}, vocab={cfg['vocab_size']}, "
          f"experts={cfg['n_routed_experts']}")

    print(f"\nreading non-routed pack manifest: {pack.name}")
    nrpk_entries = read_nrpk_manifest(pack)
    print(f"  {len(nrpk_entries)} tensors declared")

    out.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFFullManifestWriter(str(out))

    print(f"\n[1/4] arch KV...")
    emit_arch_kv(w, cfg)
    print(f"  {len(w.kv)} KV entries")

    print(f"[2/4] tokenizer KV...")
    emit_tokenizer_kv(w, tok, cfg["vocab_size"])
    print(f"  {len(w.kv)} KV entries total")

    print(f"[3/4] non-routed tensor manifest...")
    n_nr, n_scale, unmapped = emit_nonrouted_tensors(w, nrpk_entries)
    print(f"  emitted={n_nr}, paired-scale-skipped={n_scale}, unmapped={len(unmapped)}")
    if unmapped:
        print(f"  unmapped sample: {unmapped[:5]}")

    print(f"[4/4] routed FFN tensor manifest (VQB2)...")
    n_r = emit_routed_tensors(w, cfg)
    print(f"  emitted={n_r}")

    print(f"\nwriting {out}...")
    declared_bytes = w.write()
    sz = out.stat().st_size
    print(f"  file size: {sz/1e6:.3f} MB (manifest only — no tensor data section)")
    print(f"  declared tensor data size: {declared_bytes/1e9:.3f} GB (will be supplied by packs)")
    print(f"  tensor count: {len(w.tensors)}")
    print(f"  KV count:     {len(w.kv)}")

    print(f"\nnext step (#771): patch parse_tensors to accept manifest-only mode,")
    print(f"then plumb both packs through override-fill (currently only the")
    print(f"non-routed pack is wired).")


if __name__ == "__main__":
    main()
