#include "ds4_cdx3_reader.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static uint16_t cdx3_u16(const uint8_t *p) {
    return (uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8);
}

static uint32_t cdx3_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t cdx3_u64(const uint8_t *p) {
    return (uint64_t)cdx3_u32(p) | ((uint64_t)cdx3_u32(p + 4) << 32);
}

static float cdx3_f32(const uint8_t *p) {
    uint32_t bits = cdx3_u32(p);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float cdx3_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x03ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            out = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float value;
    memcpy(&value, &out, sizeof(value));
    return value;
}

static bool cdx3_mmap_readonly(const char *path, void **map, size_t *size, int *fd_out) {
    *map = NULL;
    *size = 0;
    *fd_out = -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "ds4_cdx3: open(%s) failed: %s\n", path, strerror(errno));
        return false;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "ds4_cdx3: fstat(%s) failed or empty: %s\n", path, strerror(errno));
        close(fd);
        return false;
    }
    void *mapped = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "ds4_cdx3: mmap(%s) failed: %s\n", path, strerror(errno));
        close(fd);
        return false;
    }
    *map = mapped;
    *size = (size_t)st.st_size;
    *fd_out = fd;
    return true;
}

static ds4_cdx3_record cdx3_record_at(const ds4_cdx3_file *file, uint32_t record_index) {
    const uint8_t *p = file->records + (size_t)record_index * DS4_CDX3I_RECORD_BYTES;
    ds4_cdx3_record r;
    r.layer = cdx3_u16(p + 0);
    r.expert = cdx3_u16(p + 2);
    r.kind = cdx3_u16(p + 4);
    r.bits = cdx3_u16(p + 6);
    r.out_dim = cdx3_u32(p + 8);
    r.in_dim = cdx3_u32(p + 12);
    r.n_indices = cdx3_u32(p + 16);
    r.scale_count = cdx3_u32(p + 20);
    r.payload_offset = cdx3_u64(p + 24);
    r.framed_bytes = cdx3_u64(p + 32);
    r.scale_offset = cdx3_u64(p + 40);
    r.index_offset = cdx3_u64(p + 48);
    r.scale_log_min = cdx3_f32(p + 56);
    r.scale_log_step = cdx3_f32(p + 60);
    return r;
}

bool ds4_cdx3_open(const char *pack_path, const char *index_path, ds4_cdx3_file *out) {
    if (!pack_path || !index_path || !out) return false;
    memset(out, 0, sizeof(*out));
    out->pack_fd = -1;
    out->index_fd = -1;
    if (!cdx3_mmap_readonly(pack_path, &out->pack_map, &out->pack_size, &out->pack_fd)) return false;
    if (!cdx3_mmap_readonly(index_path, &out->index_map, &out->index_size, &out->index_fd)) {
        ds4_cdx3_close(out);
        return false;
    }
    if (out->pack_size < 8 || memcmp(out->pack_map, "CDX3", 4) != 0) {
        fprintf(stderr, "ds4_cdx3: bad CDX3 pack magic\n");
        ds4_cdx3_close(out);
        return false;
    }
    const uint8_t *idx = (const uint8_t *)out->index_map;
    if (out->index_size < DS4_CDX3I_HEADER_BYTES || memcmp(idx, DS4_CDX3I_MAGIC, 7) != 0) {
        fprintf(stderr, "ds4_cdx3: bad CDX3I index magic/size\n");
        ds4_cdx3_close(out);
        return false;
    }
    out->version = cdx3_u32(idx + 8);
    out->n_records = cdx3_u32(idx + 12);
    out->records_total = cdx3_u32(idx + 16);
    out->complete = cdx3_u32(idx + 20);
    out->d = cdx3_u32(idx + 24);
    out->group = cdx3_u32(idx + 28);
    out->first_record_offset = cdx3_u64(idx + 32);
    out->indexed_pack_size = cdx3_u64(idx + 40);
    out->created_epoch = cdx3_u64(idx + 48);
    if (out->version != DS4_CDX3I_VERSION || out->d != 8u || out->group != 128u) {
        fprintf(stderr, "ds4_cdx3: unsupported index version/d/group: v=%u d=%u group=%u\n", out->version, out->d, out->group);
        ds4_cdx3_close(out);
        return false;
    }
    const size_t expected_min = DS4_CDX3I_HEADER_BYTES + (size_t)out->n_records * DS4_CDX3I_RECORD_BYTES;
    if (out->index_size < expected_min) {
        fprintf(stderr, "ds4_cdx3: index truncated: have=%zu expected>=%zu\n", out->index_size, expected_min);
        ds4_cdx3_close(out);
        return false;
    }
    out->k_by_layer = (const uint32_t *)(idx + 56);
    out->codebook_offsets = (const uint64_t *)(idx + 56 + 43u * sizeof(uint32_t));
    out->records = idx + DS4_CDX3I_HEADER_BYTES;
    return true;
}

void ds4_cdx3_close(ds4_cdx3_file *file) {
    if (!file) return;
    if (file->pack_map && file->pack_map != MAP_FAILED) munmap(file->pack_map, file->pack_size);
    if (file->index_map && file->index_map != MAP_FAILED) munmap(file->index_map, file->index_size);
    if (file->pack_fd >= 0) close(file->pack_fd);
    if (file->index_fd >= 0) close(file->index_fd);
    memset(file, 0, sizeof(*file));
    file->pack_fd = -1;
    file->index_fd = -1;
}

bool ds4_cdx3_get_record(const ds4_cdx3_file *file, uint32_t layer, uint32_t expert, uint32_t kind, ds4_cdx3_record *out) {
    if (!file || !out || layer >= 43u || expert >= 256u || kind > DS4_CDX3_KIND_DOWN) return false;
    const uint32_t direct = (layer * 256u + expert) * 3u + kind;
    if (direct < file->n_records) {
        ds4_cdx3_record r = cdx3_record_at(file, direct);
        if (r.layer == layer && r.expert == expert && r.kind == kind) {
            *out = r;
            return true;
        }
    }
    for (uint32_t i = 0; i < file->n_records; i++) {
        ds4_cdx3_record r = cdx3_record_at(file, i);
        if (r.layer == layer && r.expert == expert && r.kind == kind) {
            *out = r;
            return true;
        }
    }
    return false;
}

uint32_t ds4_cdx3_get_code(const ds4_cdx3_file *file, const ds4_cdx3_record *record, uint32_t block_index) {
    if (!file || !record || block_index >= record->n_indices) return UINT32_MAX;
    const uint64_t bit_off = (uint64_t)block_index * record->bits;
    const size_t byte_off = (size_t)(bit_off >> 3);
    const uint32_t shift = (uint32_t)(bit_off & 7u);
    const size_t index_bytes = (size_t)(((uint64_t)record->n_indices * record->bits + 7u) >> 3);
    if (record->index_offset + byte_off >= file->pack_size) return UINT32_MAX;
    const uint8_t *codes = (const uint8_t *)file->pack_map + record->index_offset;
    uint32_t window = 0;
    for (uint32_t i = 0; i < 4u && byte_off + i < index_bytes; i++) {
        window |= (uint32_t)codes[byte_off + i] << (8u * i);
    }
    const uint32_t mask = (1u << record->bits) - 1u;
    return (window >> shift) & mask;
}

float ds4_cdx3_get_scale(const ds4_cdx3_file *file, const ds4_cdx3_record *record, uint32_t block_index) {
    if (!file || !record || block_index >= record->n_indices || record->in_dim == 0) return 0.0f;
    const uint32_t blocks_per_row = record->in_dim / 8u;
    const uint32_t row = block_index / blocks_per_row;
    const uint32_t block_in_row = block_index - row * blocks_per_row;
    const uint32_t col = block_in_row * 8u;
    const uint32_t scale_groups = record->in_dim / 128u;
    const uint64_t scale_index = (uint64_t)row * scale_groups + (col / 128u);
    if (scale_index >= record->scale_count || record->scale_offset + scale_index >= file->pack_size) return 0.0f;
    const uint8_t code = *((const uint8_t *)file->pack_map + record->scale_offset + scale_index);
    return expf(record->scale_log_min + (float)code * record->scale_log_step);
}

bool ds4_cdx3_decode_block(const ds4_cdx3_file *file, const ds4_cdx3_record *record, uint32_t block_index, float out_values[8]) {
    if (!file || !record || !out_values || record->layer >= 43u || record->kind > DS4_CDX3_KIND_DOWN) return false;
    const uint32_t code = ds4_cdx3_get_code(file, record, block_index);
    const uint32_t k = file->k_by_layer[record->layer];
    if (code >= k) return false;
    const uint64_t cb_offset = file->codebook_offsets[(uint64_t)record->layer * 3u + record->kind];
    const uint64_t block_offset = cb_offset + (uint64_t)code * 8u * 2u;
    if (block_offset + 16u > file->pack_size) return false;
    const uint8_t *half = (const uint8_t *)file->pack_map + block_offset;
    const float scale = ds4_cdx3_get_scale(file, record, block_index);
    for (uint32_t i = 0; i < 8u; i++) {
        out_values[i] = scale * cdx3_f16_to_f32(cdx3_u16(half + i * 2u));
    }
    return true;
}

void ds4_cdx3_print_summary(const ds4_cdx3_file *file) {
    if (!file) {
        fprintf(stderr, "ds4_cdx3: (null)\n");
        return;
    }
    fprintf(stderr,
            "ds4_cdx3: version=%u records=%u/%u complete=%u d=%u group=%u pack=%.2f GiB index=%.2f MiB first_record=%llu indexed_pack=%llu\n",
            file->version, file->n_records, file->records_total, file->complete,
            file->d, file->group, (double)file->pack_size / 1073741824.0,
            (double)file->index_size / 1048576.0,
            (unsigned long long)file->first_record_offset,
            (unsigned long long)file->indexed_pack_size);
}
