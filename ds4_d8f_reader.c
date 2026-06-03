#include "ds4_d8f_reader.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    DS4_D8F_HEADER_BYTES = 4096,
    DS4_D8F_EXPERTS = 256,
    DS4_D8F_RECORD_BYTES = 64,
    DS4_D8F_SIDECAR_RECORD_BYTES = 64,
    DS4_D8F_NATIVE_CODE_RECORD_BYTES = 32,
    DS4_D8F_NATIVE_CODE_DTYPE_U16 = 1,
    DS4_D8F_FLAG_ACT_SCALE = 1u,
    DS4_D8F_GATEUP_OVERLAY_PROJECTIONS = 2,
    DS4_D8F_GATEUP_ROW_BLOCKS = 16,
    DS4_D8F_GATEUP_OVERLAY_SENTINEL = 0xffffffffu,
    DS4_D8F_GATEUP_OVERLAY_MAX_RECORDS = 8192,
};

static uint32_t d8f_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t d8f_u64(const uint8_t *p) {
    return (uint64_t)d8f_u32(p) | ((uint64_t)d8f_u32(p + 4) << 32);
}

static float d8f_f32(const uint8_t *p) {
    float value = 0.0f;
    memcpy(&value, p, sizeof(value));
    return value;
}

static bool d8f_supported_k(uint32_t k) {
    return k == 256u || k == 512u || k == 1024u || k == 2048u || k == 4096u;
}

static bool d8f_supported_bits(uint32_t bits) {
    return bits == 8u || bits == 9u || bits == 10u || bits == 11u || bits == 12u;
}

static uint32_t d8f_header_layer(const uint8_t *json, uint32_t bytes) {
    static const char key[] = "\"layer\"";
    if (!json || bytes < sizeof(key)) return UINT32_MAX;
    for (uint32_t i = 0; i + sizeof(key) - 1u <= bytes; i++) {
        if (memcmp(json + i, key, sizeof(key) - 1u) != 0) continue;
        uint32_t j = i + (uint32_t)sizeof(key) - 1u;
        while (j < bytes && (json[j] == ' ' || json[j] == '\t' || json[j] == '\n' || json[j] == '\r')) j++;
        if (j >= bytes || json[j] != ':') continue;
        j++;
        while (j < bytes && (json[j] == ' ' || json[j] == '\t' || json[j] == '\n' || json[j] == '\r')) j++;
        uint32_t value = 0;
        uint32_t digits = 0;
        while (j < bytes && json[j] >= '0' && json[j] <= '9') {
            value = value * 10u + (uint32_t)(json[j] - '0');
            digits++;
            j++;
        }
        return digits ? value : UINT32_MAX;
    }
    return UINT32_MAX;
}

static bool d8f_header_u64(const uint8_t *json, uint32_t bytes, const char *name, uint64_t *out) {
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

static uint32_t d8f_header_u32_or(const uint8_t *json, uint32_t bytes, const char *name, uint32_t default_value) {
    uint64_t value = 0;
    if (!d8f_header_u64(json, bytes, name, &value) || value > UINT32_MAX) return default_value;
    return (uint32_t)value;
}

static bool d8f_header_string_value(const uint8_t *json,
                                    uint32_t bytes,
                                    const char *name,
                                    const uint8_t **value,
                                    uint32_t *value_bytes) {
    if (!json || !name || !value || !value_bytes) return false;
    char key[128];
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
        if (j >= bytes || json[j] != '"') return false;
        j++;
        const uint32_t start = j;
        while (j < bytes && json[j] != '"') {
            if (json[j] == '\\') return false;
            j++;
        }
        if (j >= bytes || json[j] != '"') return false;
        *value = json + start;
        *value_bytes = j - start;
        return true;
    }
    return false;
}

static bool d8f_header_string_equals(const uint8_t *json, uint32_t bytes, const char *name, const char *expected) {
    const uint8_t *value = NULL;
    uint32_t value_bytes = 0;
    const size_t expected_len = expected ? strlen(expected) : 0;
    return expected &&
           expected_len <= UINT32_MAX &&
           d8f_header_string_value(json, bytes, name, &value, &value_bytes) &&
           value_bytes == (uint32_t)expected_len &&
           memcmp(value, expected, expected_len) == 0;
}

static bool d8f_header_string_any2(const uint8_t *json,
                                   uint32_t bytes,
                                   const char *name,
                                   const char *a,
                                   const char *b) {
    return d8f_header_string_equals(json, bytes, name, a) ||
           d8f_header_string_equals(json, bytes, name, b);
}

static void d8f_sidecar_read(const uint8_t *p, ds4_d8f_sidecar_record *out) {
    out->expert = d8f_u32(p + 0);
    out->rank = d8f_u32(p + 4);
    out->in_dim = d8f_u32(p + 8);
    out->out_dim = d8f_u32(p + 12);
    out->u_offset = d8f_u64(p + 16);
    out->a_offset = d8f_u64(p + 24);
    out->u_bytes = d8f_u32(p + 32);
    out->a_bytes = d8f_u32(p + 36);
    out->flags = d8f_u32(p + 40);
    out->reserved = d8f_u32(p + 44);
    out->eff_rank = d8f_f32(p + 48);
    out->gain = d8f_f32(p + 52);
    out->aa_frac = d8f_f32(p + 56);
    out->reserved_f32 = d8f_f32(p + 60);
}

static void d8f_record_read(const uint8_t *p, ds4_d8f_record *out) {
    out->projection = d8f_u32(p + 0);
    out->expert = d8f_u32(p + 4);
    out->k = d8f_u32(p + 8);
    out->bits = d8f_u32(p + 12);
    out->block = d8f_u32(p + 16);
    out->row_block = d8f_u32(p + 20);
    out->codebook_offset = d8f_u64(p + 24);
    out->index_offset = d8f_u64(p + 32);
    out->scale_offset = d8f_u64(p + 40);
    out->codebook_bytes = d8f_u32(p + 48);
    out->index_bytes = d8f_u32(p + 52);
    out->scale_bytes = d8f_u32(p + 56);
    out->flags = d8f_u32(p + 60);
}

static void d8f_native_code_read(const uint8_t *p, ds4_d8f_native_code_record *out) {
    out->expert = d8f_u32(p + 0);
    out->rows = d8f_u32(p + 4);
    out->groups = d8f_u32(p + 8);
    out->dtype = d8f_u32(p + 12);
    out->offset = d8f_u64(p + 16);
    out->bytes = d8f_u32(p + 24);
    out->flags = d8f_u32(p + 28);
}

static bool d8f_span_ok(const ds4_d8f_file *file, uint64_t offset, uint64_t bytes) {
    if (bytes == 0u) return false;
    if (offset < DS4_D8F_HEADER_BYTES + (uint64_t)DS4_D8F_PROJECTION_COUNT * DS4_D8F_EXPERTS * DS4_D8F_RECORD_BYTES) return false;
    if (offset + bytes < offset) return false;
    return offset + bytes <= (uint64_t)file->size;
}

static bool d8f_payload_span_ok(const ds4_d8f_file *file, uint64_t offset, uint64_t bytes) {
    if (bytes == 0u) return false;
    if (offset + bytes < offset) return false;
    return offset + bytes <= (uint64_t)file->size;
}

static bool d8f_sidecar_span_ok(const ds4_d8f_file *file, uint64_t offset, uint64_t bytes) {
    if (bytes == 0u) return false;
    if (offset + bytes < offset) return false;
    if (file->sidecar_table_offset &&
        offset < file->sidecar_table_offset + (uint64_t)file->sidecar_records * file->sidecar_record_bytes) return false;
    return offset + bytes <= (uint64_t)file->size;
}

bool ds4_d8f_open(const char *path, ds4_d8f_file *file) {
    if (!path || !file) return false;
    memset(file, 0, sizeof(*file));
    file->fd = -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "ds4_d8f: open(%s) failed: %s\n", path, strerror(errno));
        return false;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < DS4_D8F_HEADER_BYTES + DS4_D8F_PROJECTION_COUNT * DS4_D8F_EXPERTS * DS4_D8F_RECORD_BYTES) {
        fprintf(stderr, "ds4_d8f: bad size for %s\n", path);
        close(fd);
        return false;
    }
    uint8_t *base = (uint8_t *)mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        fprintf(stderr, "ds4_d8f: mmap(%s) failed: %s\n", path, strerror(errno));
        close(fd);
        return false;
    }
    if (memcmp(base, "DS4D8F1\0", 8) != 0) {
        fprintf(stderr, "ds4_d8f: bad magic in %s\n", path);
        munmap(base, (size_t)st.st_size);
        close(fd);
        return false;
    }
    file->fd = fd;
    file->map = base;
    file->size = (size_t)st.st_size;
    file->version = d8f_u32(base + 8);
    file->header_json_bytes = d8f_u32(base + 12);
    file->record_bytes = DS4_D8F_RECORD_BYTES;
    if (file->version != 1u || 16u + file->header_json_bytes > DS4_D8F_HEADER_BYTES) {
        fprintf(stderr, "ds4_d8f: unsupported header version=%u json=%u\n", file->version, file->header_json_bytes);
        ds4_d8f_close(file);
        return false;
    }
    file->layer = d8f_header_layer(base + 16, file->header_json_bytes);
    const uint8_t *header_json = base + 16;
    file->sidecar_required =
        d8f_header_string_equals(header_json, file->header_json_bytes, "target", "rank1_down_sidecar_required") ||
        d8f_header_string_equals(header_json, file->header_json_bytes, "target", "rank1_down_sidecar_direct");
    file->sidecar_none_selected =
        d8f_header_string_equals(header_json, file->header_json_bytes, "target", "rank1_down_sidecar_none_selected") ||
        d8f_header_string_equals(header_json, file->header_json_bytes, "target", "d8f_direct_no_sidecar");
    file->rank1_residual_sidecars =
        d8f_header_string_any2(header_json, file->header_json_bytes, "codec",
                               "raw-FP4->VQ-D8-rank1-prefactor-residual",
                               "direct-FP4-to-VQ-D8+rank1-down-sidecar") &&
        d8f_header_string_any2(header_json, file->header_json_bytes, "sidecar_format",
                               "DS4D8F_DOWN_RANK1_PREFACTOR_RESIDUAL",
                               "DS4D8F_DOWN_RANK1");
    file->declared_sidecar_count = d8f_header_u32_or(header_json, file->header_json_bytes, "sidecar_count", 0u);
    uint64_t overlay_slot_offset = 0;
    if (d8f_header_u64(header_json, file->header_json_bytes, "gateup_overlay_slot_table_offset", &overlay_slot_offset)) {
        uint64_t overlay_record_offset = 0;
        file->gateup_overlay_slot_table_offset = overlay_slot_offset;
        if (!d8f_header_u64(header_json, file->header_json_bytes, "gateup_overlay_record_offset", &overlay_record_offset)) {
            fprintf(stderr, "ds4_d8f: gate/up overlay slot table present without record offset\n");
            ds4_d8f_close(file);
            return false;
        }
        file->gateup_overlay_record_offset = overlay_record_offset;
        file->gateup_overlay_slot_entries =
            d8f_header_u32_or(header_json, file->header_json_bytes, "gateup_overlay_slot_entries",
                              DS4_D8F_GATEUP_OVERLAY_PROJECTIONS * DS4_D8F_EXPERTS * DS4_D8F_GATEUP_ROW_BLOCKS);
        file->gateup_overlay_record_bytes =
            d8f_header_u32_or(header_json, file->header_json_bytes, "gateup_overlay_record_bytes", DS4_D8F_RECORD_BYTES);
        file->gateup_overlay_records =
            d8f_header_u32_or(header_json, file->header_json_bytes, "gateup_overlay_records", 0u);
        file->gateup_overlay_sentinel =
            d8f_header_u32_or(header_json, file->header_json_bytes, "gateup_overlay_sentinel",
                              DS4_D8F_GATEUP_OVERLAY_SENTINEL);
        const uint64_t slot_bytes = (uint64_t)file->gateup_overlay_slot_entries * sizeof(uint32_t);
        const uint64_t record_bytes =
            (uint64_t)file->gateup_overlay_records * file->gateup_overlay_record_bytes;
        if (file->gateup_overlay_slot_entries !=
                DS4_D8F_GATEUP_OVERLAY_PROJECTIONS * DS4_D8F_EXPERTS * DS4_D8F_GATEUP_ROW_BLOCKS ||
            file->gateup_overlay_record_bytes != DS4_D8F_RECORD_BYTES ||
            file->gateup_overlay_records > DS4_D8F_GATEUP_OVERLAY_MAX_RECORDS ||
            file->gateup_overlay_sentinel != DS4_D8F_GATEUP_OVERLAY_SENTINEL ||
            !d8f_payload_span_ok(file, file->gateup_overlay_slot_table_offset, slot_bytes) ||
            !d8f_payload_span_ok(file, file->gateup_overlay_record_offset, record_bytes)) {
            fprintf(stderr,
                    "ds4_d8f: invalid gate/up overlay slot_off=%llu entries=%u record_off=%llu records=%u record_bytes=%u sentinel=%u\n",
                    (unsigned long long)file->gateup_overlay_slot_table_offset,
                    file->gateup_overlay_slot_entries,
                    (unsigned long long)file->gateup_overlay_record_offset,
                    file->gateup_overlay_records,
                    file->gateup_overlay_record_bytes,
                    file->gateup_overlay_sentinel);
            ds4_d8f_close(file);
            return false;
        }
        file->gateup_rowblock_overlay = true;
    }
    uint64_t sidecar_offset = 0;
    if (d8f_header_u64(header_json, file->header_json_bytes, "sidecar_table_offset", &sidecar_offset)) {
        file->sidecar_table_offset = sidecar_offset;
        file->sidecar_record_bytes = d8f_header_u32_or(header_json, file->header_json_bytes, "sidecar_record_bytes", DS4_D8F_SIDECAR_RECORD_BYTES);
        file->sidecar_records = d8f_header_u32_or(header_json, file->header_json_bytes, "sidecar_records", DS4_D8F_EXPERTS);
        if (file->sidecar_record_bytes != DS4_D8F_SIDECAR_RECORD_BYTES ||
            file->sidecar_records > DS4_D8F_EXPERTS ||
            file->sidecar_table_offset < DS4_D8F_HEADER_BYTES + (uint64_t)DS4_D8F_PROJECTION_COUNT * DS4_D8F_EXPERTS * DS4_D8F_RECORD_BYTES ||
            file->sidecar_table_offset + (uint64_t)file->sidecar_records * file->sidecar_record_bytes > (uint64_t)file->size) {
            fprintf(stderr, "ds4_d8f: invalid sidecar table off=%llu records=%u record_bytes=%u\n",
                    (unsigned long long)file->sidecar_table_offset, file->sidecar_records, file->sidecar_record_bytes);
            ds4_d8f_close(file);
            return false;
        }
    }
    uint64_t native_code_offset = 0;
    if (d8f_header_u64(header_json, file->header_json_bytes, "down_native_code_sidecar_table_offset", &native_code_offset)) {
        file->down_native_code_sidecar_table_offset = native_code_offset;
        file->down_native_code_sidecar_record_bytes =
            d8f_header_u32_or(header_json, file->header_json_bytes,
                              "down_native_code_sidecar_record_bytes", DS4_D8F_NATIVE_CODE_RECORD_BYTES);
        file->down_native_code_sidecar_records =
            d8f_header_u32_or(header_json, file->header_json_bytes,
                              "down_native_code_sidecar_records", DS4_D8F_EXPERTS);
        if (file->down_native_code_sidecar_record_bytes != DS4_D8F_NATIVE_CODE_RECORD_BYTES ||
            file->down_native_code_sidecar_records > DS4_D8F_EXPERTS ||
            file->down_native_code_sidecar_table_offset <
                DS4_D8F_HEADER_BYTES + (uint64_t)DS4_D8F_PROJECTION_COUNT * DS4_D8F_EXPERTS * DS4_D8F_RECORD_BYTES ||
            file->down_native_code_sidecar_table_offset +
                (uint64_t)file->down_native_code_sidecar_records * file->down_native_code_sidecar_record_bytes >
                (uint64_t)file->size) {
            fprintf(stderr, "ds4_d8f: invalid down native-code sidecar table off=%llu records=%u record_bytes=%u\n",
                    (unsigned long long)file->down_native_code_sidecar_table_offset,
                    file->down_native_code_sidecar_records,
                    file->down_native_code_sidecar_record_bytes);
            ds4_d8f_close(file);
            return false;
        }
    }
    const uint8_t *table = base + DS4_D8F_HEADER_BYTES;
    for (uint32_t projection = 0; projection < DS4_D8F_PROJECTION_COUNT; projection++) {
        for (uint32_t expert = 0; expert < DS4_D8F_EXPERTS; expert++) {
            const uint32_t index = projection * DS4_D8F_EXPERTS + expert;
            ds4_d8f_record rec;
            d8f_record_read(table + (size_t)index * DS4_D8F_RECORD_BYTES, &rec);
            if (rec.projection != projection || rec.expert != expert) {
                fprintf(stderr, "ds4_d8f: invalid record order projection=%u/%u expert=%u/%u\n",
                        rec.projection, projection, rec.expert, expert);
                ds4_d8f_close(file);
                return false;
            }
            if (rec.k != 0u) {
                if (!d8f_supported_k(rec.k) ||
                    !d8f_supported_bits(rec.bits) ||
                    rec.block != 8u ||
                    rec.row_block != 0u ||
                    !d8f_span_ok(file, rec.codebook_offset, rec.codebook_bytes) ||
                    !d8f_span_ok(file, rec.index_offset, rec.index_bytes) ||
                    (rec.flags & ~DS4_D8F_FLAG_ACT_SCALE) != 0u ||
                    ((rec.flags & DS4_D8F_FLAG_ACT_SCALE) != 0u &&
                     (projection != DS4_D8F_DOWN || !d8f_span_ok(file, rec.scale_offset, rec.scale_bytes))) ||
                    ((rec.flags & DS4_D8F_FLAG_ACT_SCALE) == 0u &&
                     (rec.scale_offset != 0u || rec.scale_bytes != 0u))) {
                    fprintf(stderr, "ds4_d8f: invalid record projection=%u expert=%u k=%u bits=%u flags=%u\n",
                            projection, expert, rec.k, rec.bits, rec.flags);
                    ds4_d8f_close(file);
                    return false;
                }
            }
            file->records[projection][expert] = rec;
        }
    }
    if (file->gateup_rowblock_overlay) {
        const uint8_t *slot_bytes = base + file->gateup_overlay_slot_table_offset;
        const uint8_t *record_bytes = base + file->gateup_overlay_record_offset;
        uint8_t seen[DS4_D8F_GATEUP_OVERLAY_MAX_RECORDS] = {0};
        for (uint32_t slot_index = 0; slot_index < file->gateup_overlay_slot_entries; slot_index++) {
            const uint32_t slot = d8f_u32(slot_bytes + (size_t)slot_index * sizeof(uint32_t));
            if (slot == file->gateup_overlay_sentinel) continue;
            file->gateup_overlay_live_slots++;
            if (slot >= file->gateup_overlay_records) {
                fprintf(stderr, "ds4_d8f: gate/up overlay slot out of range slot_index=%u slot=%u records=%u\n",
                        slot_index, slot, file->gateup_overlay_records);
                ds4_d8f_close(file);
                return false;
            }
            const uint32_t projection = slot_index / (DS4_D8F_EXPERTS * DS4_D8F_GATEUP_ROW_BLOCKS);
            const uint32_t within_projection = slot_index % (DS4_D8F_EXPERTS * DS4_D8F_GATEUP_ROW_BLOCKS);
            const uint32_t expert = within_projection / DS4_D8F_GATEUP_ROW_BLOCKS;
            const uint32_t row_block = within_projection % DS4_D8F_GATEUP_ROW_BLOCKS;
            ds4_d8f_record rec;
            d8f_record_read(record_bytes + (size_t)slot * DS4_D8F_RECORD_BYTES, &rec);
            if (rec.projection != projection ||
                rec.expert != expert ||
                rec.block != 8u ||
                rec.flags != 0u ||
                rec.scale_offset != 0u ||
                rec.scale_bytes != 0u ||
                rec.k == 0u ||
                !d8f_supported_k(rec.k) ||
                !d8f_supported_bits(rec.bits) ||
                !d8f_span_ok(file, rec.codebook_offset, rec.codebook_bytes) ||
                !d8f_span_ok(file, rec.index_offset, rec.index_bytes) ||
                rec.row_block != row_block) {
                fprintf(stderr,
                        "ds4_d8f: invalid gate/up overlay record slot=%u expected p=%u e=%u row_block=%u got p=%u e=%u k=%u bits=%u block=%u row_block=%u flags=%u\n",
                        slot, projection, expert, row_block, rec.projection, rec.expert, rec.k, rec.bits, rec.block, rec.row_block, rec.flags);
                ds4_d8f_close(file);
                return false;
            }
            seen[slot] = 1u;
        }
        for (uint32_t slot = 0; slot < file->gateup_overlay_records; slot++) {
            if (!seen[slot]) {
                fprintf(stderr, "ds4_d8f: unreachable gate/up overlay record slot=%u records=%u\n",
                        slot, file->gateup_overlay_records);
                ds4_d8f_close(file);
                return false;
            }
        }
        if (file->gateup_overlay_live_slots != file->gateup_overlay_records) {
            fprintf(stderr, "ds4_d8f: duplicate gate/up overlay slots live=%u records=%u\n",
                    file->gateup_overlay_live_slots, file->gateup_overlay_records);
            ds4_d8f_close(file);
            return false;
        }
    }
    if (file->sidecar_table_offset) {
        const uint8_t *sidecar_table = base + file->sidecar_table_offset;
        for (uint32_t i = 0; i < file->sidecar_records; i++) {
            ds4_d8f_sidecar_record sidecar;
            d8f_sidecar_read(sidecar_table + (size_t)i * DS4_D8F_SIDECAR_RECORD_BYTES, &sidecar);
            if (sidecar.rank == 0u) continue;
            if (sidecar.expert >= DS4_D8F_EXPERTS ||
                sidecar.rank > 8u ||
                sidecar.in_dim != 2048u ||
                sidecar.out_dim != 4096u ||
                sidecar.reserved != 0u ||
                sidecar.reserved_f32 != 0.0f ||
                !d8f_sidecar_span_ok(file, sidecar.u_offset, sidecar.u_bytes) ||
                !d8f_sidecar_span_ok(file, sidecar.a_offset, sidecar.a_bytes) ||
                sidecar.u_bytes != sidecar.rank * 2048u * 2u ||
                sidecar.a_bytes != sidecar.rank * 4096u * 2u) {
                fprintf(stderr, "ds4_d8f: invalid down sidecar row=%u expert=%u rank=%u in=%u out=%u\n",
                        i, sidecar.expert, sidecar.rank, sidecar.in_dim, sidecar.out_dim);
                ds4_d8f_close(file);
                return false;
            }
            file->down_sidecars[sidecar.expert] = sidecar;
            file->sidecar_count++;
        }
    }
    if (file->down_native_code_sidecar_table_offset) {
        const uint8_t *native_table = base + file->down_native_code_sidecar_table_offset;
        for (uint32_t i = 0; i < file->down_native_code_sidecar_records; i++) {
            ds4_d8f_native_code_record rec;
            d8f_native_code_read(native_table + (size_t)i * DS4_D8F_NATIVE_CODE_RECORD_BYTES, &rec);
            if (rec.bytes == 0u) continue;
            if (rec.expert >= DS4_D8F_EXPERTS ||
                rec.rows != 4096u ||
                rec.groups != 256u ||
                rec.dtype != DS4_D8F_NATIVE_CODE_DTYPE_U16 ||
                rec.flags != 0u ||
                rec.bytes != rec.rows * rec.groups * 2u ||
                !d8f_payload_span_ok(file, rec.offset, rec.bytes)) {
                fprintf(stderr, "ds4_d8f: invalid down native-code sidecar row=%u expert=%u rows=%u groups=%u dtype=%u bytes=%u\n",
                        i, rec.expert, rec.rows, rec.groups, rec.dtype, rec.bytes);
                ds4_d8f_close(file);
                return false;
            }
            file->down_native_codes[rec.expert] = rec;
            file->down_native_code_sidecar_count++;
        }
    }
    if (file->sidecar_count > 0u && !file->rank1_residual_sidecars) {
        fprintf(stderr,
                "ds4_d8f: sidecar table present without rank1-residual codec contract "
                "target_required=%u declared=%u live=%u\n",
                file->sidecar_required ? 1u : 0u, file->declared_sidecar_count, file->sidecar_count);
        ds4_d8f_close(file);
        return false;
    }
    if (file->sidecar_required && file->sidecar_count == 0u) {
        fprintf(stderr, "ds4_d8f: rank1 sidecar target has no live sidecars declared=%u\n",
                file->declared_sidecar_count);
        ds4_d8f_close(file);
        return false;
    }
    if (file->sidecar_none_selected && file->sidecar_count != 0u) {
        fprintf(stderr, "ds4_d8f: none-selected sidecar target has live sidecars=%u\n", file->sidecar_count);
        ds4_d8f_close(file);
        return false;
    }
    if ((file->sidecar_required || file->sidecar_none_selected || file->sidecar_table_offset) &&
        file->declared_sidecar_count != file->sidecar_count) {
        fprintf(stderr, "ds4_d8f: sidecar count mismatch declared=%u live=%u\n",
                file->declared_sidecar_count, file->sidecar_count);
        ds4_d8f_close(file);
        return false;
    }
    return true;
}

void ds4_d8f_close(ds4_d8f_file *file) {
    if (!file) return;
    if (file->map && file->map != MAP_FAILED) munmap((void *)file->map, file->size);
    if (file->fd >= 0) close(file->fd);
    memset(file, 0, sizeof(*file));
    file->fd = -1;
}

bool ds4_d8f_get_record(const ds4_d8f_file *file, ds4_d8f_projection projection, uint32_t expert, ds4_d8f_record *out) {
    if (!file || !out || projection >= DS4_D8F_PROJECTION_COUNT || expert >= DS4_D8F_EXPERTS) return false;
    ds4_d8f_record rec = file->records[projection][expert];
    if (rec.k == 0u) return false;
    *out = rec;
    return true;
}

bool ds4_d8f_get_gateup_overlay_record(const ds4_d8f_file *file, ds4_d8f_projection projection, uint32_t expert, uint32_t row_block, ds4_d8f_record *out) {
    if (!file || !out || !file->gateup_rowblock_overlay) return false;
    if (projection != DS4_D8F_GATE && projection != DS4_D8F_UP) return false;
    if (expert >= DS4_D8F_EXPERTS || row_block >= DS4_D8F_GATEUP_ROW_BLOCKS) return false;
    const uint32_t slot_index =
        ((uint32_t)projection * DS4_D8F_EXPERTS + expert) * DS4_D8F_GATEUP_ROW_BLOCKS + row_block;
    if (slot_index >= file->gateup_overlay_slot_entries) return false;
    const uint8_t *slot_base = file->map + file->gateup_overlay_slot_table_offset;
    const uint32_t slot = d8f_u32(slot_base + (size_t)slot_index * sizeof(uint32_t));
    if (slot == file->gateup_overlay_sentinel || slot >= file->gateup_overlay_records) return false;
    ds4_d8f_record rec;
    d8f_record_read(file->map + file->gateup_overlay_record_offset + (uint64_t)slot * file->gateup_overlay_record_bytes, &rec);
    if (rec.projection != projection ||
        rec.expert != expert ||
        rec.row_block != row_block ||
        rec.k == 0u) return false;
    *out = rec;
    return true;
}

bool ds4_d8f_get_down_sidecar(const ds4_d8f_file *file, uint32_t expert, ds4_d8f_sidecar_record *out) {
    if (!file || !out || expert >= DS4_D8F_EXPERTS) return false;
    ds4_d8f_sidecar_record rec = file->down_sidecars[expert];
    if (rec.rank == 0u) return false;
    *out = rec;
    return true;
}

bool ds4_d8f_get_down_native_codes(const ds4_d8f_file *file, uint32_t expert, ds4_d8f_native_code_record *out) {
    if (!file || !out || expert >= DS4_D8F_EXPERTS) return false;
    ds4_d8f_native_code_record rec = file->down_native_codes[expert];
    if (rec.bytes == 0u) return false;
    *out = rec;
    return true;
}

uint32_t ds4_d8f_down_sidecar_count(const ds4_d8f_file *file) {
    return file ? file->sidecar_count : 0u;
}

uint32_t ds4_d8f_down_native_code_sidecar_count(const ds4_d8f_file *file) {
    return file ? file->down_native_code_sidecar_count : 0u;
}

uint32_t ds4_d8f_gateup_overlay_count(const ds4_d8f_file *file) {
    return file ? file->gateup_overlay_live_slots : 0u;
}

uint32_t ds4_d8f_code_at(const ds4_d8f_file *file, const ds4_d8f_record *record, uint64_t block_index) {
    if (!file || !record || record->bits == 0u) return 0;
    const uint64_t bit = block_index * (uint64_t)record->bits;
    const uint64_t byte = bit >> 3;
    const uint32_t shift = (uint32_t)(bit & 7u);
    if (byte + 4u > record->index_bytes) return 0;
    const uint8_t *p = file->map + record->index_offset + byte;
    const uint32_t word = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    const uint32_t mask = record->bits >= 32u ? UINT32_MAX : ((1u << record->bits) - 1u);
    return (word >> shift) & mask;
}

void ds4_d8f_print_summary(const ds4_d8f_file *file) {
    if (!file) return;
    uint32_t live[DS4_D8F_PROJECTION_COUNT] = {0};
    uint32_t act_scaled = 0;
    uint64_t codebook_bytes = 0, index_bytes = 0, scale_bytes = 0, sidecar_bytes = 0;
    uint64_t overlay_codebook_bytes = 0, overlay_index_bytes = 0;
    for (uint32_t projection = 0; projection < DS4_D8F_PROJECTION_COUNT; projection++) {
        for (uint32_t expert = 0; expert < DS4_D8F_EXPERTS; expert++) {
            const ds4_d8f_record *rec = &file->records[projection][expert];
            if (rec->k == 0u) continue;
            live[projection]++;
            if (rec->flags & DS4_D8F_FLAG_ACT_SCALE) act_scaled++;
            codebook_bytes += rec->codebook_bytes;
            index_bytes += rec->index_bytes;
            scale_bytes += rec->scale_bytes;
        }
    }
    for (uint32_t expert = 0; expert < DS4_D8F_EXPERTS; expert++) {
        const ds4_d8f_sidecar_record *rec = &file->down_sidecars[expert];
        if (rec->rank == 0u) continue;
        sidecar_bytes += rec->u_bytes + rec->a_bytes;
    }
    if (file->gateup_rowblock_overlay) {
        const uint8_t *record_base = file->map + file->gateup_overlay_record_offset;
        for (uint32_t slot = 0; slot < file->gateup_overlay_records; slot++) {
            ds4_d8f_record rec;
            d8f_record_read(record_base + (uint64_t)slot * file->gateup_overlay_record_bytes, &rec);
            if (rec.k == 0u) continue;
            overlay_codebook_bytes += rec.codebook_bytes;
            overlay_index_bytes += rec.index_bytes;
        }
    }
    fprintf(stderr,
            "ds4_d8f: version=%u record_bytes=%u size=%.3f MiB live_gate=%u live_up=%u live_down=%u act_scaled=%u sidecars=%u gateup_overlay=%u rank1_residual=%u codebook=%.3f MiB index=%.3f MiB scale=%.3f MiB sidecar=%.3f MiB overlay_codebook=%.3f MiB overlay_index=%.3f MiB\n",
            file->version, file->record_bytes, (double)file->size / 1048576.0,
            live[DS4_D8F_GATE], live[DS4_D8F_UP], live[DS4_D8F_DOWN], act_scaled, file->sidecar_count,
            file->gateup_overlay_live_slots,
            file->rank1_residual_sidecars ? 1u : 0u,
            (double)codebook_bytes / 1048576.0,
            (double)index_bytes / 1048576.0,
            (double)scale_bytes / 1048576.0,
            (double)sidecar_bytes / 1048576.0,
            (double)overlay_codebook_bytes / 1048576.0,
            (double)overlay_index_bytes / 1048576.0);
}
