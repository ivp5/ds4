/* ds4_vqb2_reader.c — VQ-2D codec file reader, bit-packed format (VQB2). */

#include "ds4_vqb2_reader.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

bool ds4_vqb2_open(const char *path, ds4_vqb2_file *out) {
    if (!path || !out) return false;
    memset(out, 0, sizeof(*out));
    out->fd = -1;

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "ds4_vqb2: open(%s) failed: %s\n", path, strerror(errno));
        return false;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "ds4_vqb2: fstat(%s) failed: %s\n", path, strerror(errno));
        close(fd);
        return false;
    }
    if ((size_t)st.st_size < DS4_VQB2_HEADER_BYTES) {
        fprintf(stderr, "ds4_vqb2: %s too small (%lld bytes)\n",
                path, (long long)st.st_size);
        close(fd);
        return false;
    }

    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        fprintf(stderr, "ds4_vqb2: mmap(%s) failed: %s\n", path, strerror(errno));
        close(fd);
        return false;
    }

    const uint8_t *p = (const uint8_t *)map;
    if (memcmp(p, DS4_VQB2_MAGIC, 4) != 0) {
        fprintf(stderr, "ds4_vqb2: %s bad magic\n", path);
        munmap(map, (size_t)st.st_size);
        close(fd);
        return false;
    }

    out->fd = fd;
    out->map = map;
    out->map_size = (size_t)st.st_size;
    memcpy(&out->version,   p + 4,  4);
    memcpy(&out->n_experts, p + 8,  4);
    memcpy(&out->n_rows,    p + 12, 4);
    memcpy(&out->n_pairs,   p + 16, 4);
    memcpy(&out->layer,     p + 20, 4);
    memcpy(&out->kind_id,   p + 24, 4);
    memcpy(&out->k,         p + 28, 4);
    memcpy(&out->bit_width, p + 32, 4);
    uint32_t n_codes_lo = 0;
    memcpy(&n_codes_lo,     p + 36, 4);
    out->n_codes = (uint64_t)n_codes_lo;

    /* Validate bit-width vs k. */
    if (out->bit_width != 4 && out->bit_width != 6 && out->bit_width != 8) {
        fprintf(stderr, "ds4_vqb2: %s unsupported bit_width=%u (expected 4|6|8)\n",
                path, out->bit_width);
        munmap(map, out->map_size); close(fd);
        memset(out, 0, sizeof(*out)); out->fd = -1;
        return false;
    }
    const uint32_t k_expected = 1u << out->bit_width;
    if (out->k != k_expected) {
        fprintf(stderr, "ds4_vqb2: %s k=%u inconsistent with bit_width=%u (expected k=%u)\n",
                path, out->k, out->bit_width, k_expected);
        munmap(map, out->map_size); close(fd);
        memset(out, 0, sizeof(*out)); out->fd = -1;
        return false;
    }
    out->code_mask = (uint8_t)(k_expected - 1u);

    /* Validate n_codes redundant field matches dimensions. */
    const uint64_t n_codes_computed =
        (uint64_t)out->n_experts * (uint64_t)out->n_rows * (uint64_t)out->n_pairs;
    if (out->n_codes != n_codes_computed) {
        fprintf(stderr, "ds4_vqb2: %s n_codes header=%llu computed=%llu mismatch\n",
                path, (unsigned long long)out->n_codes,
                (unsigned long long)n_codes_computed);
        munmap(map, out->map_size); close(fd);
        memset(out, 0, sizeof(*out)); out->fd = -1;
        return false;
    }

    const size_t cb_bytes    = (size_t)out->k * 2u * sizeof(float);
    const size_t codes_bytes = ds4_vqb2_codes_bytes(out->n_codes, out->bit_width);
    const size_t expected    = DS4_VQB2_HEADER_BYTES + cb_bytes + codes_bytes;
    if (expected != out->map_size) {
        fprintf(stderr, "ds4_vqb2: %s size mismatch (have %zu, expected %zu) "
                        "header=%u cb=%zu codes=%zu n_codes=%llu bw=%u\n",
                path, out->map_size, expected,
                DS4_VQB2_HEADER_BYTES, cb_bytes, codes_bytes,
                (unsigned long long)out->n_codes, out->bit_width);
        munmap(map, out->map_size); close(fd);
        memset(out, 0, sizeof(*out)); out->fd = -1;
        return false;
    }

    out->codebook    = (const float   *)(p + DS4_VQB2_HEADER_BYTES);
    out->codes       = (const uint8_t *)(p + DS4_VQB2_HEADER_BYTES + cb_bytes);
    out->codes_bytes = codes_bytes;
    return true;
}

void ds4_vqb2_close(ds4_vqb2_file *f) {
    if (!f) return;
    if (f->map && f->map != MAP_FAILED) munmap(f->map, f->map_size);
    if (f->fd >= 0) close(f->fd);
    memset(f, 0, sizeof(*f));
    f->fd = -1;
}

/* Core bit-unpack: fetch up to 16 contiguous bits at linear pair index,
 * mask to bit_width, return the code in [0, k).
 *
 * Hot path. Branchless. Handles bit_width 4/6/8 uniformly.
 *
 * Endianness: little-bit-endian within bytes (LSB of byte 0 is bit 0 of
 * the first code). Matches Python `numpy.packbits(..., bitorder='little')`
 * convention used by codex encoder.
 */
static inline uint32_t vqb2_extract_code(const uint8_t *codes,
                                         uint64_t linear_pair,
                                         uint32_t bit_width,
                                         uint8_t code_mask) {
    const uint64_t bit_off  = linear_pair * (uint64_t)bit_width;
    const size_t   byte_off = (size_t)(bit_off >> 3);
    const uint32_t shift    = (uint32_t)(bit_off & 7ull);
    /* Load 16 bits LE; safe because: for 8-bit codes we only need 1 byte
     * read; for 4/6-bit codes the read may straddle a byte boundary so we
     * need 2 bytes. The packed-codes region is padded by ceil(n_bits/8),
     * and we ensure callers don't request a pair beyond n_codes.
     */
    const uint32_t lo = codes[byte_off];
    const uint32_t hi = (bit_width + shift > 8u) ? codes[byte_off + 1u] : 0u;
    const uint32_t window = lo | (hi << 8);
    return (window >> shift) & (uint32_t)code_mask;
}

uint32_t ds4_vqb2_get_code(const ds4_vqb2_file *f,
                           uint32_t expert, uint32_t row, uint32_t pair) {
    if (!f || !f->codes) return UINT32_MAX;
    if (expert >= f->n_experts || row >= f->n_rows || pair >= f->n_pairs) {
        return UINT32_MAX;
    }
    const uint64_t linear =
        (uint64_t)expert * f->n_rows * f->n_pairs
        + (uint64_t)row * f->n_pairs
        + (uint64_t)pair;
    return vqb2_extract_code(f->codes, linear, f->bit_width, f->code_mask);
}

bool ds4_vqb2_decode_pair(const ds4_vqb2_file *f,
                          uint32_t expert, uint32_t row, uint32_t pair,
                          float *out_re, float *out_im) {
    if (!f || !f->codebook || !f->codes) return false;
    if (expert >= f->n_experts || row >= f->n_rows || pair >= f->n_pairs) return false;
    const uint64_t linear =
        (uint64_t)expert * f->n_rows * f->n_pairs
        + (uint64_t)row * f->n_pairs
        + (uint64_t)pair;
    const uint32_t code =
        vqb2_extract_code(f->codes, linear, f->bit_width, f->code_mask);
    if (code >= f->k) return false; /* impossible given mask, but defensive */
    if (out_re) *out_re = f->codebook[(size_t)code * 2u + 0u];
    if (out_im) *out_im = f->codebook[(size_t)code * 2u + 1u];
    return true;
}

void ds4_vqb2_print_summary(const ds4_vqb2_file *f, uint32_t n_samples) {
    if (!f) { fprintf(stderr, "ds4_vqb2: (null file)\n"); return; }
    fprintf(stderr,
        "ds4_vqb2: version=%u  n_experts=%u  n_rows=%u  n_pairs=%u  layer=%u  "
        "kind_id=%u  k=%u  bit_width=%u  n_codes=%llu  map=%.1f MB  codes=%.1f MB\n",
        f->version, f->n_experts, f->n_rows, f->n_pairs, f->layer,
        f->kind_id, f->k, f->bit_width, (unsigned long long)f->n_codes,
        (double)f->map_size / 1e6, (double)f->codes_bytes / 1e6);

    if (n_samples > 0 && f->codebook) {
        const uint32_t n = n_samples > f->k ? f->k : n_samples;
        fprintf(stderr, "  codebook sample (first %u centroids):\n", n);
        for (uint32_t i = 0; i < n; i++) {
            fprintf(stderr, "    cb[%u] = (%+.4f, %+.4f)\n",
                    i, f->codebook[i*2], f->codebook[i*2+1]);
        }
    }
    if (n_samples > 0 && f->codes && f->n_experts > 0) {
        fprintf(stderr, "  decode samples (expert=0, row=0):\n");
        for (uint32_t p = 0; p < n_samples && p < f->n_pairs; p++) {
            float re = 0.0f, im = 0.0f;
            if (ds4_vqb2_decode_pair(f, 0, 0, p, &re, &im)) {
                const uint32_t c = ds4_vqb2_get_code(f, 0, 0, p);
                fprintf(stderr, "    pair[0,0,%u] code=%u → (%+.4f, %+.4f)\n",
                        p, c, re, im);
            }
        }
    }
}
