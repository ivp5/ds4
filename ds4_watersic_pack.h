/* ds4_watersic_pack.h — pack-backed WaterSIC view loader (silv 2026-05-28).
 *
 * Backs the WaterSIC weight-only codec port (QMM-II arxiv 2605.13768
 * Algorithm 3). Replaces 2D VQB2 vector quantization with scalar
 * quantization + per-row scale derived from Cholesky factor of the
 * activation covariance Σ_X. Theoretically within 2πe/12 ≈ 0.25 bit/entry
 * of the information-theoretic optimum; basis-free; simpler decode kernel
 * (no codebook, no LUT, coalesced byte reads).
 *
 * Pack layout (single mmap file, all little-endian):
 *
 *   [ ws_header_t                              ]  fixed 128 B header
 *   [ ws_lk_header_t × (n_layers × n_kinds)    ]  per (layer, kind) index
 *   [ alpha arena : n_lk × n_rows × fp32       ]  per-row scales (shared
 *                                                 across experts in a layer/kind)
 *   [ codes arena : n_lk × n_experts × n_rows
 *                     × n_cols × R bits        ]  scalar codes, R-bit packed
 *
 * Notes:
 *   - R is a single value per pack (fixed bit width). Per-layer adaptive R
 *     deferred to a future pack version.
 *   - α arena is small: 43 × 3 × n_rows × 4 bytes ≈ 2 MB total.
 *   - Codes arena dominates: at R=4 across all 256 experts × 43 × 3 matrices
 *     of [2048×4096] elements = ~131 GB. Production needs adaptive R.
 *   - The pack reconstructs Ŵ[i,j] = α[i] × Z[i,j]. Decoder kernel computes
 *     out[i] = α[i] × Σ_j Z[i,j] × X[j]. No LUT, just per-row scale.
 *
 * silv 2026-05-28: this header lands the format spec. The full encoder +
 * decoder kernel land in subsequent commits. See:
 *   /Users/silv/cl/tlp/montyneg/ds4/watersic/DESIGN.md
 *   /Users/silv/cl/tlp/montyneg/ds4/watersic/SONICMOE_ADDENDUM.md
 *   /Users/silv/cl/tlp/montyneg/ds4/watersic/TRITONMOE_ADDENDUM.md
 */
#ifndef DS4_WATERSIC_PACK_H
#define DS4_WATERSIC_PACK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_WATERSIC_MAGIC          0x57534943u /* 'WSIC' little-endian */
#define DS4_WATERSIC_VERSION        1u
#define DS4_WATERSIC_MAX_LAYERS     64u   /* generous; DS4 V4 uses 43 */
#define DS4_WATERSIC_N_KINDS        3u    /* gate=0, up=1, down=2 */

typedef struct ds4_watersic_header {
    uint32_t magic;             /* DS4_WATERSIC_MAGIC */
    uint32_t version;           /* DS4_WATERSIC_VERSION */
    uint32_t n_layers;          /* 43 for DS4 V4 */
    uint32_t n_experts;         /* 256 for DS4 V4 */
    uint32_t n_kinds;           /* 3 (gate, up, down) */
    uint32_t R;                 /* bits per code; codes are signed in [-2^(R-1), 2^(R-1)-1] */
    uint32_t d_model;           /* 4096 for DS4 V4 */
    uint32_t d_ffn;             /* 2048 for DS4 V4 */
    uint64_t lk_table_offset;   /* offset to first ws_lk_header_t */
    uint64_t alpha_arena_offset;
    uint64_t codes_arena_offset;
    uint64_t total_pack_bytes;  /* sanity check for mmap size */
    uint32_t pad[10];           /* reserved; zero */
} ds4_watersic_header;          /* 128 bytes */

/* Per (layer, kind) metadata. 43 × 3 = 129 entries, each 64 bytes. */
typedef struct ds4_watersic_lk_header {
    uint32_t layer;             /* 0..n_layers-1 */
    uint32_t kind;              /* 0=gate, 1=up, 2=down */
    uint32_t n_rows;            /* gate/up: d_ffn=2048; down: d_model=4096 */
    uint32_t n_cols;            /* gate/up: d_model=4096; down: d_ffn=2048 */
    uint64_t alpha_offset;      /* byte offset into alpha arena */
    uint64_t codes_offset;      /* byte offset into codes arena (start of expert 0) */
    uint64_t codes_stride;      /* per-expert stride in bytes within this lk's codes */
    uint64_t codes_bytes_per_lk;/* total codes bytes for all experts in this lk */
    uint32_t pad[6];            /* reserved */
} ds4_watersic_lk_header;       /* 64 bytes */

typedef struct ds4_watersic_pack {
    int      fd;                /* pack file fd; -1 when closed */
    void    *map;               /* mmap base */
    size_t   map_size;
    const ds4_watersic_header *hdr;          /* points into map */
    const ds4_watersic_lk_header *lk_table;  /* points into map */
    const uint8_t *alpha_arena;              /* points into map */
    const uint8_t *codes_arena;              /* points into map */
    /* O(1) lookup: lookup[layer * n_kinds + kind] = lk_table index, or -1. */
    int32_t *lookup;
    uint32_t n_lk;
    char     pack_path[1024];
} ds4_watersic_pack;

/* Open pack file + validate header + build lookup table.
 * Returns true on success, false on any validation failure. */
bool ds4_watersic_pack_open(const char *pack_path, ds4_watersic_pack *out);

/* Close pack + free lookup. Safe on partial / zero-init struct. */
void ds4_watersic_pack_close(ds4_watersic_pack *p);

/* O(1) lookup: returns lk_table index, or -1 if absent. */
int32_t ds4_watersic_pack_lookup_index(const ds4_watersic_pack *p,
                                       uint32_t layer, uint32_t kind);

/* Get pointers to α and codes for a specific (layer, kind, expert).
 *
 * On success:
 *   *out_alpha     = pointer to n_rows fp32 values (shared across experts
 *                    in this (layer, kind))
 *   *out_codes     = pointer to packed codes for this expert
 *                    (n_rows × n_cols × R / 8 bytes)
 *   *out_n_rows    = n_rows
 *   *out_n_cols    = n_cols
 *
 * Returns false on missing entry or invalid expert index. */
bool ds4_watersic_pack_get(const ds4_watersic_pack *p,
                           uint32_t layer, uint32_t kind, uint32_t expert,
                           const float  **out_alpha,
                           const uint8_t **out_codes,
                           uint32_t      *out_n_rows,
                           uint32_t      *out_n_cols);

/* CPU scalar reference dequantization: reconstruct one row of Ŵ.
 *
 * out[j] = α[row] * Z[row, j] for j in [0, n_cols).
 *
 * This is the deployable reconstruction matching the WaterSIC paper's
 * declared output: Ŵ = diag(α) Z. */
void ds4_watersic_pack_dequant_row(const ds4_watersic_pack *p,
                                   const float  *alpha,
                                   const uint8_t *codes,
                                   uint32_t n_cols,
                                   uint32_t row,
                                   float *out_w_row);

/* CPU scalar reference matvec: compute one output element.
 *
 * out[row] = α[row] * sum_j (Z[row, j] * X[j])
 *
 * Used by the Metal canary cross-check.
 * R must match pack's R; codes is the row's R-bit packed codes (n_cols values). */
float ds4_watersic_pack_matvec_row(const ds4_watersic_pack *p,
                                   const float  *alpha,
                                   const uint8_t *codes_row,
                                   const float  *x,
                                   uint32_t n_cols,
                                   uint32_t row);

/* Print summary to stderr: pack size, R, dims, n_layers, total bytes. */
void ds4_watersic_pack_print_summary(const ds4_watersic_pack *p);

/* Decode a single signed R-bit code at position `code_idx` from the
 * packed bytes. Used by the CPU reference. R must be one of {2, 4, 8}. */
int ds4_watersic_unpack_code(const uint8_t *codes_row, uint32_t code_idx, uint32_t R);

#ifdef __cplusplus
}
#endif

#endif /* DS4_WATERSIC_PACK_H */
