/* silv 2026-05-27 task #643 — Hadamard-16 batched FP16 smoke test.
 *
 * Verifies kernel_hadamard16_fp16_batched produces the correct 16-point
 * Walsh-Hadamard transform (orthogonally normalized by 1/sqrt(16)). Two
 * test cases:
 *
 * (1) Single block, known input [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].
 *     Expected output: [0.25, 0.25, ..., 0.25] (all 16 values).
 *     Because H_16 applied to e_0 gives the first column = all-1s, scaled
 *     by 1/sqrt(16) = 0.25.
 *
 * (2) Multi-block, randomized input. Apply Hadamard twice — must recover
 *     original (self-inverse). Within FP16 precision.
 *
 * Build: from ds4 dir,
 *   cc -O3 -fobjc-arc -I. tmp/20260527_hadamard/smoke.c \
 *      ds4_metal.o ds4_metal_vqb2_fp16.o ds4_inflight.o ds4_expert_table.o \
 *      ds4_polar_reader.o ds4_vqb1_reader.o ds4_vqb2_reader.o \
 *      -lm -framework Foundation -framework Metal \
 *      -o tmp/20260527_hadamard/smoke
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "ds4_gpu.h"

/* FP16 conversion helpers — minimal IEEE 754 binary16. */
static uint16_t fp32_to_fp16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    const uint32_t sign = (bits >> 31) & 0x1;
    const int32_t  exp  = (int32_t)((bits >> 23) & 0xff) - 127 + 15;
    const uint32_t mant = bits & 0x7fffff;
    uint16_t out;
    if (exp <= 0) {
        out = (uint16_t)(sign << 15);
    } else if (exp >= 31) {
        out = (uint16_t)((sign << 15) | 0x7c00);
    } else {
        out = (uint16_t)((sign << 15) | ((uint32_t)exp << 10) | (mant >> 13));
    }
    return out;
}

static float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 0x1;
    const int32_t  exp  = (int32_t)((h >> 10) & 0x1f);
    const uint32_t mant = (uint32_t)h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        bits = sign << 31;  /* zero or subnormal — treat as zero for our needs */
    } else if (exp == 31) {
        bits = (sign << 31) | 0x7f800000;
    } else {
        bits = (sign << 31) | ((uint32_t)(exp - 15 + 127) << 23) | (mant << 13);
    }
    float v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

/* CPU reference: 16-point Walsh-Hadamard butterfly (sequency unordered),
 * scaled by 1/sqrt(16) = 0.25. */
static void hadamard16_ref(float x[16]) {
    for (uint32_t stride = 1; stride < 16; stride <<= 1) {
        for (uint32_t i = 0; i < 16; i++) {
            if ((i & stride) == 0) {
                const uint32_t base = (i & ~(2 * stride - 1)) + (i & (stride - 1));
                const float a = x[base];
                const float b = x[base + stride];
                x[base]          = a + b;
                x[base + stride] = a - b;
            }
        }
    }
    for (uint32_t i = 0; i < 16; i++) x[i] *= 0.25f;
}

static int test_case_basis(void) {
    printf("=== test 1: H_16(e_0) should be [0.25, 0.25, ..., 0.25] ===\n");

    /* Allocate FP16 tensor: 1 row, 1 block, 16 halves = 32 bytes. */
    ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(32);
    if (!tensor) { fprintf(stderr, "  alloc failed\n"); return 1; }

    uint16_t input[16];
    memset(input, 0, sizeof(input));
    input[0] = fp32_to_fp16(1.0f);
    if (ds4_gpu_tensor_write(tensor, 0, input, sizeof(input)) == 0) {
        fprintf(stderr, "  write failed\n"); ds4_gpu_tensor_free(tensor); return 1;
    }

    const int rc = ds4_gpu_hadamard16_fp16_batched_tensor(
        tensor, 1, 1, 16 * sizeof(uint16_t));
    if (!rc) { fprintf(stderr, "  hadamard16 dispatch returned 0\n"); ds4_gpu_tensor_free(tensor); return 1; }

    uint16_t out[16];
    if (ds4_gpu_tensor_read(tensor, 0, out, sizeof(out)) == 0) {
        fprintf(stderr, "  read failed\n"); ds4_gpu_tensor_free(tensor); return 1;
    }

    int fail = 0;
    for (int i = 0; i < 16; i++) {
        const float v = fp16_to_fp32(out[i]);
        const float expected = 0.25f;
        if (fabsf(v - expected) > 1.0e-3f) {
            printf("  [%2d] %.4f != expected %.4f\n", i, (double)v, (double)expected);
            fail = 1;
        }
    }
    if (!fail) printf("  PASS: all 16 cells = 0.25 (within 1e-3)\n");

    ds4_gpu_tensor_free(tensor);
    return fail;
}

static int test_case_roundtrip(void) {
    printf("\n=== test 2: H(H(x)) = x within FP16 precision ===\n");

    const uint32_t n_rows = 4;
    const uint32_t blocks_per_row = 8;
    const uint64_t row_stride = blocks_per_row * 16 * sizeof(uint16_t);
    const size_t total_bytes = (size_t)n_rows * row_stride;
    ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(total_bytes);
    if (!tensor) { fprintf(stderr, "  alloc failed\n"); return 1; }

    /* Deterministic pseudo-random input. */
    const size_t n_halves = (size_t)n_rows * blocks_per_row * 16;
    uint16_t *input = malloc(n_halves * sizeof(uint16_t));
    float *original = malloc(n_halves * sizeof(float));
    if (!input || !original) {
        fprintf(stderr, "  malloc failed\n"); free(input); free(original);
        ds4_gpu_tensor_free(tensor); return 1;
    }
    uint32_t state = 0x12345678u;
    for (size_t i = 0; i < n_halves; i++) {
        state = state * 1664525u + 1013904223u;
        const float v = ((float)(int32_t)(state >> 8) / (float)(1u << 23)) * 2.0f - 1.0f;
        original[i] = v;
        input[i] = fp32_to_fp16(v);
    }
    if (ds4_gpu_tensor_write(tensor, 0, input, total_bytes) == 0) {
        fprintf(stderr, "  write failed\n");
        free(input); free(original); ds4_gpu_tensor_free(tensor); return 1;
    }

    /* Apply Hadamard twice — should recover original. */
    if (!ds4_gpu_hadamard16_fp16_batched_tensor(tensor, n_rows, blocks_per_row, row_stride)) {
        fprintf(stderr, "  hadamard16 1st dispatch failed\n");
        free(input); free(original); ds4_gpu_tensor_free(tensor); return 1;
    }
    if (!ds4_gpu_hadamard16_fp16_batched_tensor(tensor, n_rows, blocks_per_row, row_stride)) {
        fprintf(stderr, "  hadamard16 2nd dispatch failed\n");
        free(input); free(original); ds4_gpu_tensor_free(tensor); return 1;
    }

    uint16_t *output = malloc(n_halves * sizeof(uint16_t));
    if (!output) {
        fprintf(stderr, "  malloc failed\n");
        free(input); free(original); ds4_gpu_tensor_free(tensor); return 1;
    }
    if (ds4_gpu_tensor_read(tensor, 0, output, total_bytes) == 0) {
        fprintf(stderr, "  read failed\n");
        free(input); free(original); free(output); ds4_gpu_tensor_free(tensor); return 1;
    }

    int fail = 0;
    float max_err = 0.0f;
    float sum_sq_err = 0.0f;
    float sum_sq_orig = 0.0f;
    for (size_t i = 0; i < n_halves; i++) {
        const float v = fp16_to_fp32(output[i]);
        const float err = fabsf(v - original[i]);
        if (err > max_err) max_err = err;
        sum_sq_err  += err * err;
        sum_sq_orig += original[i] * original[i];
    }
    const float rel_l2 = sqrtf(sum_sq_err / (sum_sq_orig + 1.0e-12f));
    printf("  n_halves=%zu, max|err|=%.4e, rel_L2=%.4e\n",
           n_halves, (double)max_err, (double)rel_l2);
    /* FP16 has ~3 decimal digits of precision; allow some accumulation. */
    if (rel_l2 > 1.0e-2f) {
        printf("  FAIL: rel_L2 > 1e-2\n");
        fail = 1;
    } else {
        printf("  PASS: rel_L2 within FP16 noise floor\n");
    }

    free(input); free(original); free(output);
    ds4_gpu_tensor_free(tensor);
    return fail;
}

static int test_case_vs_cpu_ref(void) {
    printf("\n=== test 3: GPU H(x) matches CPU reference ===\n");

    const uint32_t n_rows = 2;
    const uint32_t blocks_per_row = 4;
    const uint64_t row_stride = blocks_per_row * 16 * sizeof(uint16_t);
    const size_t total_bytes = (size_t)n_rows * row_stride;
    ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(total_bytes);
    if (!tensor) { fprintf(stderr, "  alloc failed\n"); return 1; }

    const size_t n_halves = (size_t)n_rows * blocks_per_row * 16;
    uint16_t *input = malloc(n_halves * sizeof(uint16_t));
    float *cpu_ref = malloc(n_halves * sizeof(float));
    if (!input || !cpu_ref) {
        fprintf(stderr, "  malloc failed\n");
        free(input); free(cpu_ref); ds4_gpu_tensor_free(tensor); return 1;
    }

    /* Fill with deterministic pseudo-random values. */
    uint32_t state = 0xdeadbeefu;
    for (size_t i = 0; i < n_halves; i++) {
        state = state * 1103515245u + 12345u;
        const float v = ((float)(int32_t)(state >> 8) / (float)(1u << 23)) * 2.0f - 1.0f;
        input[i] = fp32_to_fp16(v);
        cpu_ref[i] = fp16_to_fp32(input[i]);  /* match FP16 round-trip */
    }

    /* CPU reference: apply Hadamard-16 to each 16-block. */
    for (uint32_t row = 0; row < n_rows; row++) {
        for (uint32_t block = 0; block < blocks_per_row; block++) {
            float blk[16];
            for (int i = 0; i < 16; i++) {
                blk[i] = cpu_ref[(size_t)row * blocks_per_row * 16 + block * 16 + i];
            }
            hadamard16_ref(blk);
            for (int i = 0; i < 16; i++) {
                cpu_ref[(size_t)row * blocks_per_row * 16 + block * 16 + i] = blk[i];
            }
        }
    }

    if (ds4_gpu_tensor_write(tensor, 0, input, total_bytes) == 0) {
        fprintf(stderr, "  write failed\n");
        free(input); free(cpu_ref); ds4_gpu_tensor_free(tensor); return 1;
    }
    if (!ds4_gpu_hadamard16_fp16_batched_tensor(tensor, n_rows, blocks_per_row, row_stride)) {
        fprintf(stderr, "  hadamard16 dispatch failed\n");
        free(input); free(cpu_ref); ds4_gpu_tensor_free(tensor); return 1;
    }
    uint16_t *output = malloc(n_halves * sizeof(uint16_t));
    if (!output) {
        fprintf(stderr, "  malloc failed\n");
        free(input); free(cpu_ref); ds4_gpu_tensor_free(tensor); return 1;
    }
    if (ds4_gpu_tensor_read(tensor, 0, output, total_bytes) == 0) {
        fprintf(stderr, "  read failed\n");
        free(input); free(cpu_ref); free(output); ds4_gpu_tensor_free(tensor); return 1;
    }

    int fail = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < n_halves; i++) {
        const float gpu = fp16_to_fp32(output[i]);
        const float err = fabsf(gpu - cpu_ref[i]);
        if (err > max_err) max_err = err;
        if (err > 5.0e-3f) {
            printf("  [%zu] gpu=%.4f cpu=%.4f err=%.4e\n",
                   i, (double)gpu, (double)cpu_ref[i], (double)err);
            fail = 1;
        }
    }
    printf("  n_halves=%zu, max|err|=%.4e (FP16 precision floor ~5e-3)\n",
           n_halves, (double)max_err);
    if (!fail) printf("  PASS: GPU matches CPU reference within FP16 precision\n");

    free(input); free(cpu_ref); free(output);
    ds4_gpu_tensor_free(tensor);
    return fail;
}

int main(void) {
    if (!ds4_gpu_init()) {
        fprintf(stderr, "ds4_gpu_init failed\n");
        return 1;
    }
    int fail = 0;
    fail |= test_case_basis();
    fail |= test_case_roundtrip();
    fail |= test_case_vs_cpu_ref();
    printf("\n%s\n", fail ? "FAIL" : "ALL PASS");
    return fail;
}
