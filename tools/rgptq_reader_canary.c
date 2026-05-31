/* Build:
 *   cc -O2 -Wall -Wextra -std=c99 -I. -o tools/rgptq_reader_canary \
 *     tools/rgptq_reader_canary.c ds4_ridgegptq_reader.c -lm
 *
 * Usage:
 *   tools/rgptq_reader_canary tmp/20260531_rgptq_canary/sample.rgq2
 */
#include "../ds4_ridgegptq_reader.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void put_u32(uint8_t *bytes, uint32_t value) {
    bytes[0] = (uint8_t)(value & 0xffu);
    bytes[1] = (uint8_t)((value >> 8) & 0xffu);
    bytes[2] = (uint8_t)((value >> 16) & 0xffu);
    bytes[3] = (uint8_t)((value >> 24) & 0xffu);
}

static void put_u64(uint8_t *bytes, uint64_t value) {
    put_u32(bytes, (uint32_t)(value & 0xffffffffu));
    put_u32(bytes + 4, (uint32_t)(value >> 32));
}

static void pack_code(uint8_t *codes, uint64_t index, uint32_t bit_width, uint32_t code) {
    const uint64_t bit_off = index * bit_width;
    const size_t byte_off = (size_t)(bit_off >> 3);
    const uint32_t shift = (uint32_t)(bit_off & 7u);
    codes[byte_off] |= (uint8_t)(code << shift);
    if (shift + bit_width > 8u) codes[byte_off + 1u] |= (uint8_t)(code >> (8u - shift));
}

static int write_sample(const char *path) {
    const uint32_t n_experts = 2;
    const uint32_t n_rows = 3;
    const uint32_t n_cols = 5;
    const uint32_t bit_width = 2;
    const uint32_t q_levels = 4;
    const uint32_t scale_group_cols = 2;
    const uint32_t scale_groups = 3;
    const uint64_t n_codes = (uint64_t)n_experts * n_rows * n_cols;
    const size_t values_bytes = q_levels * sizeof(float);
    const size_t scales_bytes = n_experts * n_rows * scale_groups * sizeof(float);
    const size_t codes_bytes = ds4_rgptq_codes_bytes(n_codes, bit_width);
    const size_t total = DS4_RGPTQ_HEADER_BYTES + values_bytes + scales_bytes + codes_bytes;
    uint8_t *bytes = (uint8_t *)calloc(total, 1);
    if (!bytes) return 2;

    memcpy(bytes, DS4_RGPTQ_MAGIC, 4);
    put_u32(bytes + 4, DS4_RGPTQ_VERSION);
    put_u32(bytes + 8, n_experts);
    put_u32(bytes + 12, n_rows);
    put_u32(bytes + 16, n_cols);
    put_u32(bytes + 20, 7);
    put_u32(bytes + 24, DS4_RGPTQ_KIND_GATE);
    put_u32(bytes + 28, 128);
    put_u32(bytes + 32, bit_width);
    put_u32(bytes + 36, q_levels);
    put_u32(bytes + 40, scale_group_cols);
    put_u32(bytes + 44, 0);
    put_u64(bytes + 48, n_codes);
    put_u64(bytes + 56, 0);

    float *values = (float *)(bytes + DS4_RGPTQ_HEADER_BYTES);
    values[0] = -3.0f;
    values[1] = -1.0f;
    values[2] = 1.0f;
    values[3] = 3.0f;

    float *scales = (float *)(bytes + DS4_RGPTQ_HEADER_BYTES + values_bytes);
    for (uint32_t expert = 0; expert < n_experts; expert++) {
        for (uint32_t row = 0; row < n_rows; row++) {
            for (uint32_t group = 0; group < scale_groups; group++) {
                const uint64_t scale_index = ((uint64_t)expert * n_rows + row) * scale_groups + group;
                scales[scale_index] = 0.125f * (float)(1u + expert + row + group);
            }
        }
    }

    uint8_t *codes = bytes + DS4_RGPTQ_HEADER_BYTES + values_bytes + scales_bytes;
    for (uint32_t expert = 0; expert < n_experts; expert++) {
        for (uint32_t row = 0; row < n_rows; row++) {
            for (uint32_t col = 0; col < n_cols; col++) {
                const uint64_t index = ((uint64_t)expert * n_rows + row) * n_cols + col;
                pack_code(codes, index, bit_width, (expert + row + col) & 3u);
            }
        }
    }

    FILE *file = fopen(path, "wb");
    if (!file) {
        free(bytes);
        return 3;
    }
    const size_t wrote = fwrite(bytes, 1, total, file);
    fclose(file);
    free(bytes);
    return wrote == total ? 0 : 4;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s OUT.rgq2\n", argv[0]);
        return 2;
    }
    int rc = write_sample(argv[1]);
    if (rc != 0) {
        fprintf(stderr, "write_sample failed rc=%d\n", rc);
        return rc;
    }

    ds4_rgptq_file file;
    if (!ds4_rgptq_open(argv[1], &file)) return 5;
    ds4_rgptq_print_summary(&file);

    float input[5] = {1.0f, -2.0f, 0.5f, 3.0f, -1.5f};
    float row[5] = {0};
    if (!ds4_rgptq_dequant_row(&file, 1, 2, row, 5)) return 6;
    float ref = 0.0f;
    for (uint32_t col = 0; col < 5; col++) ref += row[col] * input[col];
    const float got = ds4_rgptq_matvec_row(&file, 1, 2, input, 5);
    if (fabsf(ref - got) > 1.0e-6f) {
        fprintf(stderr, "matvec mismatch ref=%g got=%g\n", ref, got);
        ds4_rgptq_close(&file);
        return 7;
    }

    float output[3] = {0};
    if (!ds4_rgptq_matvec_expert(&file, 1, input, 5, output, 3)) return 8;
    if (fabsf(output[2] - got) > 1.0e-6f) {
        fprintf(stderr, "expert row mismatch output=%g got=%g\n", output[2], got);
        ds4_rgptq_close(&file);
        return 9;
    }

    printf("rgptq_canary ok ref=%g got=%g output2=%g\\n", ref, got, output[2]);
    ds4_rgptq_close(&file);
    return 0;
}
