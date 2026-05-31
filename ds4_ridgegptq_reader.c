#include "ds4_ridgegptq_reader.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static uint32_t rgptq_read_u32(const uint8_t *bytes) {
    return (uint32_t)bytes[0]
        | ((uint32_t)bytes[1] << 8)
        | ((uint32_t)bytes[2] << 16)
        | ((uint32_t)bytes[3] << 24);
}

static uint64_t rgptq_read_u64(const uint8_t *bytes) {
    return (uint64_t)rgptq_read_u32(bytes)
        | ((uint64_t)rgptq_read_u32(bytes + 4) << 32);
}

static bool rgptq_mul_overflow_u64(uint64_t left, uint64_t right, uint64_t *out) {
    if (left != 0 && right > UINT64_MAX / left) return true;
    *out = left * right;
    return false;
}

static bool rgptq_validate_size(const ds4_rgptq_file *file) {
    uint64_t values_bytes = (uint64_t)file->q_levels * sizeof(float);
    uint64_t scale_count = 0;
    uint64_t scale_rows = 0;
    if (rgptq_mul_overflow_u64(file->n_experts, file->n_rows, &scale_rows)) return false;
    if (rgptq_mul_overflow_u64(scale_rows, file->scale_groups, &scale_count)) return false;
    uint64_t scales_bytes = 0;
    if (rgptq_mul_overflow_u64(scale_count, sizeof(float), &scales_bytes)) return false;
    uint64_t total_without_codes = DS4_RGPTQ_HEADER_BYTES + values_bytes + scales_bytes;
    if (total_without_codes < values_bytes || total_without_codes < scales_bytes) return false;
    uint64_t total = total_without_codes + file->codes_bytes;
    if (total < total_without_codes) return false;
    return total == (uint64_t)file->map_size;
}

bool ds4_rgptq_open(const char *path, ds4_rgptq_file *out) {
    if (!path || !out) return false;
    memset(out, 0, sizeof(*out));
    out->fd = -1;

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "ds4_rgptq: open(%s) failed: %s\n", path, strerror(errno));
        return false;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "ds4_rgptq: fstat(%s) failed: %s\n", path, strerror(errno));
        close(fd);
        return false;
    }
    if (st.st_size < (off_t)DS4_RGPTQ_HEADER_BYTES) {
        fprintf(stderr, "ds4_rgptq: %s too small (%lld bytes)\n", path, (long long)st.st_size);
        close(fd);
        return false;
    }

    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        fprintf(stderr, "ds4_rgptq: mmap(%s) failed: %s\n", path, strerror(errno));
        close(fd);
        return false;
    }

    out->fd = fd;
    out->map = map;
    out->map_size = (size_t)st.st_size;

    const uint8_t *header = (const uint8_t *)map;
    if (memcmp(header, DS4_RGPTQ_MAGIC, 4) != 0) {
        fprintf(stderr, "ds4_rgptq: %s bad magic\n", path);
        ds4_rgptq_close(out);
        return false;
    }

    out->version = rgptq_read_u32(header + 4);
    out->n_experts = rgptq_read_u32(header + 8);
    out->n_rows = rgptq_read_u32(header + 12);
    out->n_cols = rgptq_read_u32(header + 16);
    out->layer = rgptq_read_u32(header + 20);
    out->kind_id = rgptq_read_u32(header + 24);
    out->row_start = rgptq_read_u32(header + 28);
    out->bit_width = rgptq_read_u32(header + 32);
    out->q_levels = rgptq_read_u32(header + 36);
    out->scale_group_cols = rgptq_read_u32(header + 40);
    out->flags = rgptq_read_u32(header + 44);
    out->n_codes = rgptq_read_u64(header + 48);
    const uint64_t reserved = rgptq_read_u64(header + 56);

    if (out->version != DS4_RGPTQ_VERSION || out->flags != 0 || reserved != 0) {
        fprintf(stderr, "ds4_rgptq: %s unsupported header version/flags/reserved\n", path);
        ds4_rgptq_close(out);
        return false;
    }
    if (out->bit_width == 0 || out->bit_width > 2 || out->q_levels == 0 ||
        out->q_levels > (1u << out->bit_width) || out->scale_group_cols == 0 ||
        out->kind_id > DS4_RGPTQ_KIND_DOWN) {
        fprintf(stderr, "ds4_rgptq: %s invalid dimensions/quant fields\n", path);
        ds4_rgptq_close(out);
        return false;
    }

    uint64_t expected_codes = 0;
    uint64_t row_cols = 0;
    if (rgptq_mul_overflow_u64(out->n_rows, out->n_cols, &row_cols) ||
        rgptq_mul_overflow_u64(out->n_experts, row_cols, &expected_codes) ||
        expected_codes != out->n_codes) {
        fprintf(stderr, "ds4_rgptq: %s n_codes mismatch\n", path);
        ds4_rgptq_close(out);
        return false;
    }

    out->scale_groups = (out->n_cols + out->scale_group_cols - 1u) / out->scale_group_cols;
    out->code_mask = (uint8_t)((1u << out->bit_width) - 1u);
    out->values_bytes = (size_t)out->q_levels * sizeof(float);
    out->scales_bytes = (size_t)out->n_experts * out->n_rows * out->scale_groups * sizeof(float);
    out->codes_bytes = ds4_rgptq_codes_bytes(out->n_codes, out->bit_width);

    if (!rgptq_validate_size(out)) {
        fprintf(stderr, "ds4_rgptq: %s size mismatch (have %zu, expected header+values+scales+codes)\n",
                path, out->map_size);
        ds4_rgptq_close(out);
        return false;
    }

    const uint8_t *cursor = (const uint8_t *)out->map + DS4_RGPTQ_HEADER_BYTES;
    out->values = (const float *)cursor;
    cursor += out->values_bytes;
    out->scales = (const float *)cursor;
    cursor += out->scales_bytes;
    out->codes = cursor;
    return true;
}

void ds4_rgptq_close(ds4_rgptq_file *file) {
    if (!file) return;
    if (file->map && file->map != MAP_FAILED) munmap(file->map, file->map_size);
    if (file->fd >= 0) close(file->fd);
    memset(file, 0, sizeof(*file));
    file->fd = -1;
}

uint32_t ds4_rgptq_get_code(const ds4_rgptq_file *file,
                            uint32_t expert, uint32_t row, uint32_t col) {
    if (!file || expert >= file->n_experts || row >= file->n_rows || col >= file->n_cols) return 0;
    const uint64_t linear = ((uint64_t)expert * file->n_rows + row) * file->n_cols + col;
    const uint64_t bit_off = linear * file->bit_width;
    const size_t byte_off = (size_t)(bit_off >> 3);
    const uint32_t shift = (uint32_t)(bit_off & 7u);
    const uint32_t low = file->codes[byte_off];
    const uint32_t high = (shift + file->bit_width > 8u) ? file->codes[byte_off + 1u] : 0u;
    return (low | (high << 8)) >> shift & file->code_mask;
}

float ds4_rgptq_get_weight(const ds4_rgptq_file *file,
                           uint32_t expert, uint32_t row, uint32_t col) {
    if (!file || expert >= file->n_experts || row >= file->n_rows || col >= file->n_cols) return 0.0f;
    const uint32_t code = ds4_rgptq_get_code(file, expert, row, col);
    if (code >= file->q_levels) return 0.0f;
    const uint32_t group = col / file->scale_group_cols;
    const uint64_t scale_index = ((uint64_t)expert * file->n_rows + row) * file->scale_groups + group;
    return file->scales[scale_index] * file->values[code];
}

bool ds4_rgptq_dequant_row(const ds4_rgptq_file *file,
                           uint32_t expert, uint32_t row,
                           float *out_row, uint32_t out_cols) {
    if (!file || !out_row || expert >= file->n_experts || row >= file->n_rows) return false;
    if (out_cols < file->n_cols) return false;
    for (uint32_t col = 0; col < file->n_cols; col++) {
        out_row[col] = ds4_rgptq_get_weight(file, expert, row, col);
    }
    return true;
}

float ds4_rgptq_matvec_row(const ds4_rgptq_file *file,
                           uint32_t expert, uint32_t row,
                           const float *input, uint32_t input_cols) {
    if (!file || !input || expert >= file->n_experts || row >= file->n_rows ||
        input_cols < file->n_cols) {
        return 0.0f;
    }
    float sum = 0.0f;
    const uint64_t row_code_base = ((uint64_t)expert * file->n_rows + row) * file->n_cols;
    const uint64_t row_scale_base = ((uint64_t)expert * file->n_rows + row) * file->scale_groups;
    for (uint32_t group = 0; group < file->scale_groups; group++) {
        const uint32_t col_begin = group * file->scale_group_cols;
        uint32_t col_end = col_begin + file->scale_group_cols;
        if (col_end > file->n_cols) col_end = file->n_cols;
        const float scale = file->scales[row_scale_base + group];
        float group_sum = 0.0f;
        for (uint32_t col = col_begin; col < col_end; col++) {
            const uint64_t bit_off = (row_code_base + col) * file->bit_width;
            const size_t byte_off = (size_t)(bit_off >> 3);
            const uint32_t shift = (uint32_t)(bit_off & 7u);
            const uint32_t low = file->codes[byte_off];
            const uint32_t high = (shift + file->bit_width > 8u) ? file->codes[byte_off + 1u] : 0u;
            const uint32_t code = ((low | (high << 8)) >> shift) & file->code_mask;
            if (code < file->q_levels) group_sum += file->values[code] * input[col];
        }
        sum += scale * group_sum;
    }
    return sum;
}

bool ds4_rgptq_matvec_expert(const ds4_rgptq_file *file,
                             uint32_t expert,
                             const float *input, uint32_t input_cols,
                             float *output, uint32_t output_rows) {
    if (!file || !input || !output || expert >= file->n_experts ||
        input_cols < file->n_cols || output_rows < file->n_rows) {
        return false;
    }
    for (uint32_t row = 0; row < file->n_rows; row++) {
        output[row] = ds4_rgptq_matvec_row(file, expert, row, input, input_cols);
    }
    return true;
}

void ds4_rgptq_print_summary(const ds4_rgptq_file *file) {
    if (!file) {
        fprintf(stderr, "ds4_rgptq: (null)\n");
        return;
    }
    fprintf(stderr,
            "ds4_rgptq: v=%u L=%u kind=%u row_start=%u experts=%u rows=%u cols=%u "
            "bit_width=%u q_levels=%u scale_group_cols=%u scale_groups=%u bytes=%zu\n",
            file->version, file->layer, file->kind_id, file->row_start,
            file->n_experts, file->n_rows, file->n_cols, file->bit_width,
            file->q_levels, file->scale_group_cols, file->scale_groups,
            file->map_size);
}
