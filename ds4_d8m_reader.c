#include "ds4_d8m_reader.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    DS4_D8M_HEADER_BYTES = 4096,
    DS4_D8M_RECORD_COUNT = 256,
    DS4_D8M_RECORD_BYTES = 40,
    DS4_D8A_RECORD_BYTES = 56,
    DS4_D8M_FLAG_ACT_SCALE = 1u,
};

static bool d8m_supported_k(uint32_t k) {
    return k == 512u || k == 1024u || k == 2048u || k == 4096u;
}

static bool d8m_supported_bits(uint32_t bits) {
    return bits == 9u || bits == 10u || bits == 11u || bits == 12u;
}

static uint32_t d8m_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t d8m_u64(const uint8_t *p) {
    return (uint64_t)d8m_u32(p) | ((uint64_t)d8m_u32(p + 4) << 32);
}

static void d8m_record_read(const uint8_t *p, uint32_t record_bytes, ds4_d8m_record *out) {
    out->expert = d8m_u32(p + 0);
    out->k = d8m_u32(p + 4);
    out->bits = d8m_u32(p + 8);
    out->block = d8m_u32(p + 12);
    out->codebook_offset = d8m_u64(p + 16);
    out->index_offset = d8m_u64(p + 24);
    out->codebook_bytes = d8m_u32(p + 32);
    out->index_bytes = d8m_u32(p + 36);
    out->scale_offset = 0;
    out->scale_bytes = 0;
    out->flags = 0;
    if (record_bytes >= DS4_D8A_RECORD_BYTES) {
        out->scale_offset = d8m_u64(p + 40);
        out->scale_bytes = d8m_u32(p + 48);
        out->flags = d8m_u32(p + 52);
    }
}

bool ds4_d8m_open(const char *path, ds4_d8m_file *file) {
    if (!path || !file) return false;
    memset(file, 0, sizeof(*file));
    file->fd = -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "ds4_d8m: open(%s) failed: %s\n", path, strerror(errno));
        return false;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < DS4_D8M_HEADER_BYTES + DS4_D8M_RECORD_COUNT * DS4_D8M_RECORD_BYTES) {
        fprintf(stderr, "ds4_d8m: bad size for %s\n", path);
        close(fd);
        return false;
    }
    void *mapped = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "ds4_d8m: mmap(%s) failed: %s\n", path, strerror(errno));
        close(fd);
        return false;
    }
    const uint8_t *base = (const uint8_t *)mapped;
    uint32_t record_bytes = 0;
    if (memcmp(base, "DS4D8M1\0", 8) == 0) {
        record_bytes = DS4_D8M_RECORD_BYTES;
    } else if (memcmp(base, "DS4D8A1\0", 8) == 0) {
        record_bytes = DS4_D8A_RECORD_BYTES;
    } else {
        fprintf(stderr, "ds4_d8m: bad magic in %s\n", path);
        munmap(mapped, (size_t)st.st_size);
        close(fd);
        return false;
    }
    if ((uint64_t)DS4_D8M_HEADER_BYTES + (uint64_t)DS4_D8M_RECORD_COUNT * record_bytes > (uint64_t)st.st_size) {
        fprintf(stderr, "ds4_d8m: bad table size for %s record_bytes=%u\n", path, record_bytes);
        munmap(mapped, (size_t)st.st_size);
        close(fd);
        return false;
    }
    file->version = d8m_u32(base + 8);
    file->header_json_bytes = d8m_u32(base + 12);
    if (file->version != 1 || 16u + file->header_json_bytes > DS4_D8M_HEADER_BYTES) {
        fprintf(stderr, "ds4_d8m: unsupported header version=%u json=%u\n", file->version, file->header_json_bytes);
        munmap(mapped, (size_t)st.st_size);
        close(fd);
        return false;
    }
    file->map = base;
    file->size = (size_t)st.st_size;
    file->fd = fd;
    file->record_bytes = record_bytes;
    file->table_offset = DS4_D8M_HEADER_BYTES;
    const uint8_t *table = base + DS4_D8M_HEADER_BYTES;
    for (uint32_t i = 0; i < DS4_D8M_RECORD_COUNT; i++) {
        ds4_d8m_record rec;
        d8m_record_read(table + (size_t)i * record_bytes, record_bytes, &rec);
        if (rec.k != 0) {
            if (rec.expert != i || rec.block != 8u ||
                !d8m_supported_k(rec.k) ||
                !d8m_supported_bits(rec.bits) ||
                rec.codebook_offset + rec.codebook_bytes > file->size ||
                rec.index_offset + rec.index_bytes > file->size ||
                (rec.flags & ~DS4_D8M_FLAG_ACT_SCALE) != 0u ||
                ((rec.flags & DS4_D8M_FLAG_ACT_SCALE) != 0u &&
                 (rec.scale_bytes == 0u || rec.scale_offset + rec.scale_bytes > file->size))) {
                fprintf(stderr, "ds4_d8m: invalid record expert=%u k=%u bits=%u flags=%u\n", i, rec.k, rec.bits, rec.flags);
                ds4_d8m_close(file);
                return false;
            }
        }
        file->records[i] = rec;
    }
    return true;
}

void ds4_d8m_close(ds4_d8m_file *file) {
    if (!file) return;
    if (file->map && file->size) munmap((void *)file->map, file->size);
    if (file->fd >= 0) close(file->fd);
    memset(file, 0, sizeof(*file));
    file->fd = -1;
}

bool ds4_d8m_get_record(const ds4_d8m_file *file, uint32_t expert, ds4_d8m_record *out) {
    if (!file || !out || expert >= DS4_D8M_RECORD_COUNT) return false;
    ds4_d8m_record rec = file->records[expert];
    if (rec.k == 0) return false;
    *out = rec;
    return true;
}

uint32_t ds4_d8m_code_at(const ds4_d8m_file *file, const ds4_d8m_record *record, uint64_t block_index) {
    if (!file || !record || !record->bits) return UINT32_MAX;
    const uint64_t bit_off = block_index * (uint64_t)record->bits;
    const uint64_t byte_off = bit_off >> 3;
    const uint32_t shift = (uint32_t)(bit_off & 7u);
    if (byte_off >= record->index_bytes) return UINT32_MAX;
    const uint8_t *ix = file->map + record->index_offset;
    uint32_t w = 0;
    if (byte_off + 0u < record->index_bytes) w |= (uint32_t)ix[byte_off + 0u] << 0;
    if (byte_off + 1u < record->index_bytes) w |= (uint32_t)ix[byte_off + 1u] << 8;
    if (byte_off + 2u < record->index_bytes) w |= (uint32_t)ix[byte_off + 2u] << 16;
    if (byte_off + 3u < record->index_bytes) w |= (uint32_t)ix[byte_off + 3u] << 24;
    return (w >> shift) & ((1u << record->bits) - 1u);
}

void ds4_d8m_print_summary(const ds4_d8m_file *file) {
    if (!file) return;
    uint32_t live = 0, k512 = 0, k1024 = 0, k2048 = 0, k4096 = 0, act_scaled = 0;
    uint64_t codebook_bytes = 0, index_bytes = 0, scale_bytes = 0;
    for (uint32_t i = 0; i < DS4_D8M_RECORD_COUNT; i++) {
        const ds4_d8m_record *rec = &file->records[i];
        if (!rec->k) continue;
        live++;
        if (rec->k == 512u) k512++;
        if (rec->k == 1024u) k1024++;
        if (rec->k == 2048u) k2048++;
        if (rec->k == 4096u) k4096++;
        if (rec->flags & DS4_D8M_FLAG_ACT_SCALE) act_scaled++;
        codebook_bytes += rec->codebook_bytes;
        index_bytes += rec->index_bytes;
        scale_bytes += rec->scale_bytes;
    }
    fprintf(stderr,
            "ds4_d8m: version=%u record_bytes=%u size=%.3f MiB live=%u act_scaled=%u k512=%u k1024=%u k2048=%u k4096=%u codebook=%.3f MiB index=%.3f MiB scale=%.3f MiB\n",
            file->version, file->record_bytes, (double)file->size / 1048576.0, live, act_scaled, k512, k1024, k2048, k4096,
            (double)codebook_bytes / 1048576.0, (double)index_bytes / 1048576.0, (double)scale_bytes / 1048576.0);
}
