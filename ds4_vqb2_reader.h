/* ds4_vqb2_reader.h
 *
 * VQ-2D codec file reader, format v2 (bit-packed codes).
 *
 * VQB2 extends VQB1 with bit-packed code storage:
 *   K = 16  → 4-bit codes  (2 codes/byte,  0.50 byte/pair)
 *   K = 64  → 6-bit codes  (4 codes/3 bytes, 0.75 byte/pair)
 *   K = 256 → 8-bit codes  (1 code/byte,   1.00 byte/pair, same as VQB1)
 *
 * File layout (little-endian):
 *   header (40 bytes):
 *     0-3:   magic "VQB2"
 *     4-7:   version (=1)
 *     8-11:  n_experts
 *     12-15: n_rows
 *     16-19: n_pairs
 *     20-23: layer
 *     24-27: kind_id
 *     28-31: k                  (codebook size)
 *     32-35: bit_width          (4 | 6 | 8)
 *     36-39: n_codes            (== n_experts × n_rows × n_pairs, redundant
 *                                book-keeping field — verified in open())
 *   codebook:  k × 2 × float32  (re, im pairs)
 *   codes:     ceil(n_codes × bit_width / 8) bytes, little-bit-endian stream
 *
 * Decoding a pair (expert e, row r, pair p):
 *   1. linear_index = e × n_rows × n_pairs + r × n_pairs + p
 *   2. bit_off  = linear_index × bit_width
 *   3. byte_off = bit_off >> 3
 *   4. shift    = bit_off & 7
 *   5. fetch 16 bits at byte_off (little-endian), shift right by `shift`,
 *      mask with ((1<<bit_width)-1) → code in [0, k)
 *   6. lookup codebook[code*2 + {0,1}] → (re, im)
 *
 * One file per (layer, kind). Codebook shared across all experts within tile.
 *
 * Mirror of vq2d_encode_vqb2 output (codex H1824-H1826).
 */
#ifndef DS4_VQB2_READER_H
#define DS4_VQB2_READER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_VQB2_MAGIC "VQB2"
#define DS4_VQB2_HEADER_BYTES 40

/* Kind IDs identical to VQB1/PLR2 for cross-codec consistency. */
typedef enum {
    DS4_VQB2_KIND_GATE = 0,
    DS4_VQB2_KIND_UP   = 1,
    DS4_VQB2_KIND_DOWN = 2,
} ds4_vqb2_kind_t;

typedef struct ds4_vqb2_file {
    int fd;
    void *map;                 /* mmap base */
    size_t map_size;
    uint32_t version;
    uint32_t n_experts;
    uint32_t n_rows;
    uint32_t n_pairs;
    uint32_t layer;
    uint32_t kind_id;
    uint32_t k;
    uint32_t bit_width;        /* 4, 6, or 8 */
    uint64_t n_codes;          /* total pairs across all experts */
    uint8_t  code_mask;        /* (1<<bit_width)-1, precomputed */
    const float   *codebook;   /* points into map; k × 2 floats */
    const uint8_t *codes;      /* points into map; packed bit stream */
    size_t codes_bytes;        /* length of packed-codes region */
} ds4_vqb2_file;

/* Open VQB2 file: mmap + parse header + validate size. true on success. */
bool ds4_vqb2_open(const char *path, ds4_vqb2_file *out);

/* Close + munmap. Safe on partially-initialized struct. */
void ds4_vqb2_close(ds4_vqb2_file *f);

/* Decode one pair: returns (re, im) via out_re + out_im. False on bad index. */
bool ds4_vqb2_decode_pair(const ds4_vqb2_file *f,
                          uint32_t expert, uint32_t row, uint32_t pair,
                          float *out_re, float *out_im);

/* Raw code extraction (no codebook lookup). For diagnostics. */
uint32_t ds4_vqb2_get_code(const ds4_vqb2_file *f,
                           uint32_t expert, uint32_t row, uint32_t pair);

/* Print summary to stderr (header + a few sample decoded values). */
void ds4_vqb2_print_summary(const ds4_vqb2_file *f, uint32_t n_samples);

/* Computed bytes for the packed-codes region given n_codes + bit_width. */
static inline size_t ds4_vqb2_codes_bytes(uint64_t n_codes, uint32_t bit_width) {
    return (size_t)((n_codes * (uint64_t)bit_width + 7u) >> 3);
}

#ifdef __cplusplus
}
#endif

#endif /* DS4_VQB2_READER_H */
