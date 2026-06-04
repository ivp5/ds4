#include "ds4_d8fs_reader.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    DS4_D8FS_HEADER_BYTES = 4096,
    DS4_D8FS_RECORD_BYTES = 80,
    DS4_D8FS_RECORDS = 256,
    DS4_D8FS_FLAG_LIVE = 1u,
};

static uint32_t d8fs_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint16_t d8fs_u16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint64_t d8fs_u64(const uint8_t *p) {
    return (uint64_t)d8fs_u32(p) | ((uint64_t)d8fs_u32(p + 4) << 32);
}

static bool d8fs_span_ok(const ds4_d8fs_file *file, uint64_t offset, uint64_t bytes) {
    if (!file || bytes == 0u) return false;
    if (offset < DS4_D8FS_HEADER_BYTES + (uint64_t)DS4_D8FS_RECORDS * DS4_D8FS_RECORD_BYTES) return false;
    if (offset + bytes < offset) return false;
    return offset + bytes <= (uint64_t)file->size;
}

static bool d8fs_header_u64(const uint8_t *json, uint32_t bytes, const char *name, uint64_t *out) {
    if (!json || !name || !out) return false;
    char key[96];
    snprintf(key, sizeof(key), "\"%s\"", name);
    const size_t key_len = strlen(key);
    if (key_len == 0 || key_len >= sizeof(key) || bytes < key_len) return false;
    for (uint32_t i = 0; i + key_len <= bytes; i++) {
        if (memcmp(json + i, key, key_len) != 0) continue;
        uint32_t j = i + (uint32_t)key_len;
        while (j < bytes && (json[j] == ' ' || json[j] == '\t' || json[j] == '\n' || json[j] == '\r')) j++;
        if (j >= bytes || json[j] != ':') continue;
        j++;
        while (j < bytes && (json[j] == ' ' || json[j] == '\t' || json[j] == '\n' || json[j] == '\r')) j++;
        uint64_t value = 0;
        uint32_t digits = 0;
        while (j < bytes && json[j] >= '0' && json[j] <= '9') {
            value = value * 10u + (uint64_t)(json[j] - '0');
            digits++;
            j++;
        }
        if (!digits) return false;
        *out = value;
        return true;
    }
    return false;
}

static uint32_t d8fs_header_u32_or(const uint8_t *json, uint32_t bytes, const char *name, uint32_t default_value) {
    uint64_t value = 0;
    if (!d8fs_header_u64(json, bytes, name, &value) || value > UINT32_MAX) return default_value;
    return (uint32_t)value;
}

static uint64_t d8fs_header_u64_or(const uint8_t *json, uint32_t bytes, const char *name, uint64_t default_value) {
    uint64_t value = 0;
    return d8fs_header_u64(json, bytes, name, &value) ? value : default_value;
}

static uint32_t d8fs_ceil_log2_u32(uint32_t value) {
    if (value <= 1u) return 1u;
    uint32_t bits = 0;
    uint32_t v = value - 1u;
    while (v) {
        bits++;
        v >>= 1;
    }
    return bits ? bits : 1u;
}

static uint32_t d8fs_bitpack_get(const uint8_t *data, uint32_t bit_offset, uint32_t bits) {
    uint32_t value = 0;
    for (uint32_t b = 0; b < bits; b++) {
        const uint32_t pos = bit_offset + b;
        const uint32_t byte_index = pos >> 3;
        const uint32_t bit_index = pos & 7u;
        value |= (uint32_t)((data[byte_index] >> bit_index) & 1u) << b;
    }
    return value;
}

static void d8fs_record_read(const uint8_t *p, ds4_d8fs_record *out) {
    out->expert = d8fs_u32(p + 0);
    out->rows = d8fs_u32(p + 4);
    out->groups = d8fs_u32(p + 8);
    out->k = d8fs_u32(p + 12);
    out->unique_count = d8fs_u32(p + 16);
    out->max_group_unique = d8fs_u32(p + 20);
    out->group_prefix_offset = d8fs_u64(p + 24);
    out->inverse_offset_table_offset = d8fs_u64(p + 32);
    out->unique_code_offset = d8fs_u64(p + 40);
    out->inverse_bits_offset = d8fs_u64(p + 48);
    out->group_prefix_bytes = d8fs_u32(p + 56);
    out->inverse_offset_table_bytes = d8fs_u32(p + 60);
    out->unique_code_bytes = d8fs_u32(p + 64);
    out->inverse_bits_bytes = d8fs_u32(p + 68);
    out->inverse_bits_bits = d8fs_u32(p + 72);
    out->flags = d8fs_u32(p + 76);
}

static bool d8fs_record_valid(const ds4_d8fs_file *file, const ds4_d8fs_record *rec, uint32_t slot) {
    if (!file || !rec) return false;
    if ((rec->flags & DS4_D8FS_FLAG_LIVE) == 0u) {
        return rec->rows == 0u && rec->groups == 0u && rec->k == 0u &&
               rec->unique_count == 0u && rec->max_group_unique == 0u;
    }
    if (rec->expert != slot ||
        rec->rows != 4096u ||
        rec->groups != 256u ||
        rec->k == 0u ||
        rec->unique_count == 0u ||
        rec->max_group_unique == 0u ||
        rec->max_group_unique > rec->rows ||
        rec->unique_code_bytes != rec->unique_count * sizeof(uint16_t) ||
        rec->group_prefix_bytes != (rec->groups + 1u) * sizeof(uint32_t) ||
        rec->inverse_offset_table_bytes != (rec->groups + 1u) * sizeof(uint32_t) ||
        (rec->flags & ~DS4_D8FS_FLAG_LIVE) != 0u ||
        !d8fs_span_ok(file, rec->group_prefix_offset, rec->group_prefix_bytes) ||
        !d8fs_span_ok(file, rec->inverse_offset_table_offset, rec->inverse_offset_table_bytes) ||
        !d8fs_span_ok(file, rec->unique_code_offset, rec->unique_code_bytes) ||
        !d8fs_span_ok(file, rec->inverse_bits_offset, rec->inverse_bits_bytes)) {
        return false;
    }
    return true;
}

bool ds4_d8fs_open(const char *path, ds4_d8fs_file *file) {
    if (!path || !file) return false;
    memset(file, 0, sizeof(*file));
    file->fd = -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "ds4_d8fs: open(%s) failed: %s\n", path, strerror(errno));
        return false;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < DS4_D8FS_HEADER_BYTES + DS4_D8FS_RECORDS * DS4_D8FS_RECORD_BYTES) {
        fprintf(stderr, "ds4_d8fs: bad size for %s\n", path);
        close(fd);
        return false;
    }
    uint8_t *base = (uint8_t *)mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        fprintf(stderr, "ds4_d8fs: mmap(%s) failed: %s\n", path, strerror(errno));
        close(fd);
        return false;
    }
    if (memcmp(base, "DS4D8FS\0", 8) != 0) {
        fprintf(stderr, "ds4_d8fs: bad magic in %s\n", path);
        munmap(base, (size_t)st.st_size);
        close(fd);
        return false;
    }
    file->fd = fd;
    file->map = base;
    file->size = (size_t)st.st_size;
    file->version = d8fs_u32(base + 8);
    file->header_json_bytes = d8fs_u32(base + 12);
    file->record_bytes = DS4_D8FS_RECORD_BYTES;
    file->records_count = DS4_D8FS_RECORDS;
    if (file->version != 1u || 16u + file->header_json_bytes > DS4_D8FS_HEADER_BYTES) {
        fprintf(stderr, "ds4_d8fs: unsupported header version=%u json=%u\n",
                file->version, file->header_json_bytes);
        ds4_d8fs_close(file);
        return false;
    }
    const uint8_t *json = base + 16;
    file->layer = d8fs_header_u32_or(json, file->header_json_bytes, "layer", UINT32_MAX);
    file->unique_total = d8fs_header_u64_or(json, file->header_json_bytes, "unique_total", 0u);
    file->sparse_bytes = d8fs_header_u64_or(json, file->header_json_bytes, "sparse_bytes", 0u);
    file->native_equivalent_bytes =
        d8fs_header_u64_or(json, file->header_json_bytes, "native_equivalent_bytes", 0u);

    const uint8_t *record_table = base + DS4_D8FS_HEADER_BYTES;
    uint64_t unique_sum = 0;
    uint64_t sparse_sum = 0;
    for (uint32_t expert = 0; expert < DS4_D8FS_RECORDS; expert++) {
        ds4_d8fs_record rec;
        d8fs_record_read(record_table + (size_t)expert * DS4_D8FS_RECORD_BYTES, &rec);
        if (!d8fs_record_valid(file, &rec, expert)) {
            fprintf(stderr,
                    "ds4_d8fs: invalid sparse record expert=%u flags=%u rows=%u groups=%u unique=%u\n",
                    expert, rec.flags, rec.rows, rec.groups, rec.unique_count);
            ds4_d8fs_close(file);
            return false;
        }
        file->records[expert] = rec;
        if (rec.flags & DS4_D8FS_FLAG_LIVE) {
            file->live_records++;
            unique_sum += rec.unique_count;
            sparse_sum += (uint64_t)rec.group_prefix_bytes +
                          (uint64_t)rec.inverse_offset_table_bytes +
                          (uint64_t)rec.unique_code_bytes +
                          (uint64_t)rec.inverse_bits_bytes;
        }
    }
    if ((file->unique_total && file->unique_total != unique_sum) ||
        (file->sparse_bytes && file->sparse_bytes != sparse_sum)) {
        fprintf(stderr,
                "ds4_d8fs: header totals mismatch unique=%llu/%llu sparse=%llu/%llu\n",
                (unsigned long long)file->unique_total,
                (unsigned long long)unique_sum,
                (unsigned long long)file->sparse_bytes,
                (unsigned long long)sparse_sum);
        ds4_d8fs_close(file);
        return false;
    }
    if (!file->unique_total) file->unique_total = unique_sum;
    if (!file->sparse_bytes) file->sparse_bytes = sparse_sum;
    return true;
}

void ds4_d8fs_close(ds4_d8fs_file *file) {
    if (!file) return;
    if (file->map && file->map != MAP_FAILED) munmap((void *)file->map, file->size);
    if (file->fd >= 0) close(file->fd);
    memset(file, 0, sizeof(*file));
    file->fd = -1;
}

bool ds4_d8fs_get_record(const ds4_d8fs_file *file, uint32_t expert, ds4_d8fs_record *out) {
    if (!file || !out || expert >= DS4_D8FS_RECORDS) return false;
    ds4_d8fs_record rec = file->records[expert];
    if ((rec.flags & DS4_D8FS_FLAG_LIVE) == 0u) return false;
    *out = rec;
    return true;
}

bool ds4_d8fs_code_at(const ds4_d8fs_file *file,
                      const ds4_d8fs_record *record,
                      uint32_t row,
                      uint32_t group,
                      uint16_t *out_code) {
    if (!file || !record || !out_code) return false;
    if (row >= record->rows || group >= record->groups) return false;
    if ((record->flags & DS4_D8FS_FLAG_LIVE) == 0u) return false;
    const uint8_t *group_prefix_base = file->map + record->group_prefix_offset;
    const uint8_t *inverse_offset_base = file->map + record->inverse_offset_table_offset;
    const uint32_t unique_start = d8fs_u32(group_prefix_base + (uint64_t)group * sizeof(uint32_t));
    const uint32_t unique_end = d8fs_u32(group_prefix_base + (uint64_t)(group + 1u) * sizeof(uint32_t));
    const uint32_t inverse_start = d8fs_u32(inverse_offset_base + (uint64_t)group * sizeof(uint32_t));
    const uint32_t inverse_end = d8fs_u32(inverse_offset_base + (uint64_t)(group + 1u) * sizeof(uint32_t));
    if (unique_start >= unique_end ||
        unique_end > record->unique_count ||
        inverse_start > inverse_end ||
        inverse_end > record->inverse_bits_bytes) {
        return false;
    }
    const uint32_t group_unique = unique_end - unique_start;
    const uint32_t bits = d8fs_ceil_log2_u32(group_unique);
    const uint64_t needed_bits = (uint64_t)record->rows * bits;
    const uint64_t available_bits = (uint64_t)(inverse_end - inverse_start) * 8ull;
    if (needed_bits > available_bits || bits > 16u) return false;
    const uint8_t *inverse = file->map + record->inverse_bits_offset + inverse_start;
    const uint32_t local_index = d8fs_bitpack_get(inverse, row * bits, bits);
    if (local_index >= group_unique) return false;
    const uint8_t *unique = file->map + record->unique_code_offset;
    *out_code = d8fs_u16(unique + (uint64_t)(unique_start + local_index) * sizeof(uint16_t));
    return *out_code < record->k;
}

void ds4_d8fs_print_summary(const ds4_d8fs_file *file) {
    if (!file) return;
    fprintf(stderr,
            "DS4D8FS sparse sidecar: layer=%u records=%u live=%u unique=%llu sparse=%.3f MiB native_equiv=%.3f MiB file=%.3f MiB\n",
            file->layer,
            file->records_count,
            file->live_records,
            (unsigned long long)file->unique_total,
            (double)file->sparse_bytes / 1048576.0,
            (double)file->native_equivalent_bytes / 1048576.0,
            (double)file->size / 1048576.0);
}
