/* ds4_vqb1_reader.h
 *
 * VQ-2D codec file reader (mirror of ds4_polar_reader for VQB1 format).
 *
 * VQB1 file: header (32 bytes) + codebook (K × 2 fp32) + codes (E × R × P uint8).
 * One file per (layer, kind). Codebook is shared across all experts within
 * a (layer, kind) tile.
 *
 * Layout matches analyzers/vq2d_encode_vqb1.py output.
 */
#ifndef DS4_VQB1_READER_H
#define DS4_VQB1_READER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_VQB1_MAGIC "VQB1"
#define DS4_VQB1_HEADER_BYTES 32

/* Match the kind IDs in PLR2 / polar_reader for cross-codec consistency. */
typedef enum {
    DS4_VQB1_KIND_GATE = 0,
    DS4_VQB1_KIND_UP   = 1,
    DS4_VQB1_KIND_DOWN = 2,
} ds4_vqb1_kind_t;

typedef struct ds4_vqb1_file {
    int fd;
    void *map;          /* mmap base */
    size_t map_size;
    uint32_t version;
    uint32_t n_experts;
    uint32_t n_rows;
    uint32_t n_pairs;
    uint32_t layer;
    uint32_t kind_id;
    uint32_t k;
    const float   *codebook;  /* points into map; k * 2 floats */
    const uint8_t *codes;     /* points into map; n_experts * n_rows * n_pairs uint8 */
} ds4_vqb1_file;

/* Open a VQB1 file at path, mmap, parse header. Returns true on success. */
bool ds4_vqb1_open(const char *path, ds4_vqb1_file *out);

/* Close + munmap. Safe to call on partially-initialized struct. */
void ds4_vqb1_close(ds4_vqb1_file *f);

/* Return pointer to expert's code matrix [n_rows × n_pairs] uint8. */
const uint8_t *ds4_vqb1_expert_codes(const ds4_vqb1_file *f, uint32_t expert);

/* Decode one pair: returns (re, im) via out_re + out_im pointers.
 * Returns false on bad indices. */
bool ds4_vqb1_decode_pair(const ds4_vqb1_file *f,
                          uint32_t expert, uint32_t row, uint32_t pair,
                          float *out_re, float *out_im);

/* Print summary to stderr (header + a few sample decoded values). */
void ds4_vqb1_print_summary(const ds4_vqb1_file *f, uint32_t n_samples);

#ifdef __cplusplus
}
#endif

#endif /* DS4_VQB1_READER_H */
