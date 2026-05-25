/* ds4_vqb1_reader.c — VQ-2D codec file reader (VQB1 format). */

#include "ds4_vqb1_reader.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

bool ds4_vqb1_open(const char *path, ds4_vqb1_file *out) {
    if (!path || !out) return false;
    memset(out, 0, sizeof(*out));
    out->fd = -1;

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "ds4_vqb1: open(%s) failed: %s\n", path, strerror(errno));
        return false;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "ds4_vqb1: fstat(%s) failed: %s\n", path, strerror(errno));
        close(fd);
        return false;
    }
    if ((size_t)st.st_size < DS4_VQB1_HEADER_BYTES) {
        fprintf(stderr, "ds4_vqb1: %s too small (%lld bytes)\n", path, (long long)st.st_size);
        close(fd);
        return false;
    }

    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        fprintf(stderr, "ds4_vqb1: mmap(%s) failed: %s\n", path, strerror(errno));
        close(fd);
        return false;
    }

    const uint8_t *p = (const uint8_t *)map;
    if (memcmp(p, DS4_VQB1_MAGIC, 4) != 0) {
        fprintf(stderr, "ds4_vqb1: %s bad magic\n", path);
        munmap(map, (size_t)st.st_size);
        close(fd);
        return false;
    }

    /* Header layout (little-endian uint32):
     *   4-7:  version
     *   8-11: n_experts
     *  12-15: n_rows
     *  16-19: n_pairs
     *  20-23: layer
     *  24-27: kind_id
     *  28-31: k
     */
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

    const size_t cb_bytes = (size_t)out->k * 2u * sizeof(float);
    const size_t codes_bytes = (size_t)out->n_experts * (size_t)out->n_rows * (size_t)out->n_pairs;
    const size_t expected = DS4_VQB1_HEADER_BYTES + cb_bytes + codes_bytes;
    if (expected != out->map_size) {
        fprintf(stderr, "ds4_vqb1: %s size mismatch (have %zu, expected %zu)\n",
                path, out->map_size, expected);
        munmap(map, out->map_size);
        close(fd);
        memset(out, 0, sizeof(*out));
        out->fd = -1;
        return false;
    }

    out->codebook = (const float   *)(p + DS4_VQB1_HEADER_BYTES);
    out->codes    = (const uint8_t *)(p + DS4_VQB1_HEADER_BYTES + cb_bytes);
    return true;
}

void ds4_vqb1_close(ds4_vqb1_file *f) {
    if (!f) return;
    if (f->map && f->map != MAP_FAILED) munmap(f->map, f->map_size);
    if (f->fd >= 0) close(f->fd);
    memset(f, 0, sizeof(*f));
    f->fd = -1;
}

const uint8_t *ds4_vqb1_expert_codes(const ds4_vqb1_file *f, uint32_t expert) {
    if (!f || !f->codes || expert >= f->n_experts) return NULL;
    return f->codes + (size_t)expert * (size_t)f->n_rows * (size_t)f->n_pairs;
}

bool ds4_vqb1_decode_pair(const ds4_vqb1_file *f,
                          uint32_t expert, uint32_t row, uint32_t pair,
                          float *out_re, float *out_im) {
    if (!f || !f->codebook || !f->codes) return false;
    if (expert >= f->n_experts || row >= f->n_rows || pair >= f->n_pairs) return false;
    const uint8_t *exp_codes = ds4_vqb1_expert_codes(f, expert);
    if (!exp_codes) return false;
    const uint8_t code = exp_codes[(size_t)row * f->n_pairs + pair];
    if (code >= f->k) return false;  /* corrupt code */
    if (out_re) *out_re = f->codebook[(size_t)code * 2u + 0u];
    if (out_im) *out_im = f->codebook[(size_t)code * 2u + 1u];
    return true;
}

void ds4_vqb1_print_summary(const ds4_vqb1_file *f, uint32_t n_samples) {
    if (!f) { fprintf(stderr, "ds4_vqb1: (null file)\n"); return; }
    fprintf(stderr, "ds4_vqb1: version=%u  n_experts=%u  n_rows=%u  n_pairs=%u  "
                    "layer=%u  kind_id=%u  k=%u  map=%.1f MB\n",
            f->version, f->n_experts, f->n_rows, f->n_pairs,
            f->layer, f->kind_id, f->k, (double)f->map_size / 1e6);

    if (n_samples > 0 && f->codebook) {
        fprintf(stderr, "  codebook sample (first %u centroids):\n",
                n_samples > f->k ? f->k : n_samples);
        for (uint32_t i = 0; i < n_samples && i < f->k; i++) {
            fprintf(stderr, "    cb[%u] = (%+.4f, %+.4f)\n",
                    i, f->codebook[i*2], f->codebook[i*2+1]);
        }
    }
    if (n_samples > 0 && f->codes && f->n_experts > 0) {
        fprintf(stderr, "  decode samples (expert=0, row=0):\n");
        for (uint32_t p = 0; p < n_samples && p < f->n_pairs; p++) {
            float re = 0.0f, im = 0.0f;
            if (ds4_vqb1_decode_pair(f, 0, 0, p, &re, &im)) {
                const uint8_t c = ds4_vqb1_expert_codes(f, 0)[p];
                fprintf(stderr, "    pair[0,0,%u] code=%u → (%+.4f, %+.4f)\n",
                        p, c, re, im);
            }
        }
    }
}
