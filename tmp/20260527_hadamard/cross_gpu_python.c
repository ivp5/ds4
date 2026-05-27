/* silv 2026-05-27 task #645 — Python-encoded ↔ GPU-kernel cross-validation.
 *
 * The strongest closure on the encode-side tool: read a Python-encoded
 * Hadamard-transformed tensor from disk, apply the GPU kernel to it, and
 * verify the result equals the ORIGINAL (because H × H = I). If the
 * Python encoder and GPU kernel agree on the transform, this round-trip
 * via two different code paths gives the original within FP16 precision.
 *
 * Workflow:
 *   1. encode.py writes Wh.fp16 = H(W)
 *   2. This program reads Wh.fp16 into a tensor
 *   3. Applies the GPU kernel: H(Wh) = H(H(W)) = W
 *   4. Compares against W (loaded from the original W.fp16)
 *
 * Build (from ivp5_ds4 dir):
 *   cc -O3 -fobjc-arc -I. tmp/20260527_hadamard/cross_gpu_python.c \
 *      ds4.o ds4_metal.o ds4_metal_vqb2_fp16.o ds4_inflight.o \
 *      ds4_expert_table.o ds4_polar_reader.o ds4_vqb1_reader.o \
 *      ds4_vqb2_reader.o ds4_moe_route_log.o ds4_neon_i8mm.o \
 *      -lm -pthread -framework Foundation -framework Metal \
 *      -o tmp/20260527_hadamard/cross_gpu_python
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "ds4_gpu.h"

static float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 0x1;
    const int32_t  exp  = (int32_t)((h >> 10) & 0x1f);
    const uint32_t mant = (uint32_t)h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        bits = sign << 31;
    } else if (exp == 31) {
        bits = (sign << 31) | 0x7f800000;
    } else {
        bits = (sign << 31) | ((uint32_t)(exp - 15 + 127) << 23) | (mant << 13);
    }
    float v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

static int read_fp16_file(const char *path, uint16_t **out, size_t *out_n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || (sz % 2) != 0) {
        fclose(f);
        fprintf(stderr, "bad size %ld\n", sz);
        return -1;
    }
    *out_n = (size_t)sz / 2;
    *out = malloc((size_t)sz);
    if (!*out) { fclose(f); return -1; }
    if (fread(*out, 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f); free(*out); return -1;
    }
    fclose(f);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <W.fp16> <Wh.fp16> <n_rows> <n_in>\n", argv[0]);
        return 1;
    }
    const char *w_path  = argv[1];
    const char *wh_path = argv[2];
    const uint32_t n_rows = (uint32_t)atoi(argv[3]);
    const uint32_t n_in   = (uint32_t)atoi(argv[4]);
    if (n_in % 16u != 0u) {
        fprintf(stderr, "n_in=%u not divisible by 16\n", n_in);
        return 1;
    }
    const uint32_t blocks_per_row = n_in / 16u;
    const size_t expected_n = (size_t)n_rows * (size_t)n_in;

    uint16_t *W = NULL, *Wh = NULL;
    size_t W_n = 0, Wh_n = 0;
    if (read_fp16_file(w_path, &W, &W_n) != 0) return 2;
    if (read_fp16_file(wh_path, &Wh, &Wh_n) != 0) { free(W); return 2; }
    if (W_n != expected_n || Wh_n != expected_n) {
        fprintf(stderr, "size mismatch: W_n=%zu Wh_n=%zu expected=%zu\n",
                W_n, Wh_n, expected_n);
        free(W); free(Wh); return 3;
    }
    printf("loaded W (%zu halves) and Wh (%zu halves)\n", W_n, Wh_n);

    if (!ds4_gpu_init()) { fprintf(stderr, "ds4_gpu_init failed\n"); return 4; }

    /* Allocate tensor, write Wh into it, dispatch Hadamard, read back. */
    const size_t bytes = expected_n * sizeof(uint16_t);
    ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(bytes);
    if (!tensor) { fprintf(stderr, "tensor alloc failed\n"); return 5; }
    if (!ds4_gpu_tensor_write(tensor, 0, Wh, bytes)) {
        fprintf(stderr, "tensor write failed\n"); return 6;
    }
    const uint64_t row_stride = (uint64_t)n_in * sizeof(uint16_t);
    if (!ds4_gpu_hadamard16_fp16_batched_tensor(tensor, n_rows, blocks_per_row, row_stride)) {
        fprintf(stderr, "GPU Hadamard dispatch failed\n"); return 7;
    }
    uint16_t *Whh = malloc(bytes);
    if (!Whh) { fprintf(stderr, "Whh alloc failed\n"); return 8; }
    if (!ds4_gpu_tensor_read(tensor, 0, Whh, bytes)) {
        fprintf(stderr, "tensor read failed\n"); return 9;
    }

    /* Compare H(Wh) ≈ W. Since H × H = I orthogonally, H(H(W)) = W. */
    double max_err = 0.0;
    double sum_sq_diff = 0.0;
    double sum_sq_w = 0.0;
    for (size_t i = 0; i < expected_n; i++) {
        const float a = fp16_to_fp32(W[i]);
        const float b = fp16_to_fp32(Whh[i]);
        const double d = (double)(b - a);
        sum_sq_diff += d * d;
        sum_sq_w    += (double)a * (double)a;
        const double abs_d = fabs(d);
        if (abs_d > max_err) max_err = abs_d;
    }
    const double rel_l2 = sqrt(sum_sq_diff / (sum_sq_w + 1e-12));
    printf("cross-validation: GPU(H(Python_H(W))) vs W\n");
    printf("  max|err| = %.4e\n", max_err);
    printf("  rel_L2   = %.4e\n", rel_l2);
    const int pass = (rel_l2 < 1.0e-2);
    printf("  %s\n", pass ? "PASS — Python encoder and GPU kernel agree within FP16 precision"
                          : "FAIL — encoder and kernel disagree");

    free(W); free(Wh); free(Whh);
    ds4_gpu_tensor_free(tensor);
    return pass ? 0 : 10;
}
