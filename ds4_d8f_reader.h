#ifndef DS4_D8F_READER_H
#define DS4_D8F_READER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum ds4_d8f_projection {
    DS4_D8F_GATE = 0,
    DS4_D8F_UP = 1,
    DS4_D8F_DOWN = 2,
    DS4_D8F_PROJECTION_COUNT = 3,
} ds4_d8f_projection;

typedef struct ds4_d8f_record {
    uint32_t projection;
    uint32_t expert;
    uint32_t k;
    uint32_t bits;
    uint32_t block;
    uint32_t row_block;
    uint64_t codebook_offset;
    uint64_t index_offset;
    uint64_t scale_offset;
    uint32_t codebook_bytes;
    uint32_t index_bytes;
    uint32_t scale_bytes;
    uint32_t flags;
} ds4_d8f_record;

typedef struct ds4_d8f_sidecar_record {
    uint32_t expert;
    uint32_t rank;
    uint32_t in_dim;
    uint32_t out_dim;
    uint64_t u_offset;
    uint64_t a_offset;
    uint32_t u_bytes;
    uint32_t a_bytes;
    uint32_t flags;
    uint32_t reserved;
    float eff_rank;
    float gain;
    float aa_frac;
    float reserved_f32;
} ds4_d8f_sidecar_record;

typedef struct ds4_d8f_native_code_record {
    uint32_t expert;
    uint32_t rows;
    uint32_t groups;
    uint32_t dtype;
    uint64_t offset;
    uint32_t bytes;
    uint32_t flags;
} ds4_d8f_native_code_record;

typedef struct ds4_d8f_file {
    int fd;
    const uint8_t *map;
    size_t size;
    uint32_t version;
    uint32_t header_json_bytes;
    uint32_t record_bytes;
    uint32_t layer;
    uint64_t sidecar_table_offset;
    uint32_t sidecar_record_bytes;
    uint32_t sidecar_records;
    uint32_t sidecar_count;
    uint32_t declared_sidecar_count;
    uint64_t gateup_overlay_slot_table_offset;
    uint64_t gateup_overlay_record_offset;
    uint32_t gateup_overlay_slot_entries;
    uint32_t gateup_overlay_record_bytes;
    uint32_t gateup_overlay_records;
    uint32_t gateup_overlay_live_slots;
    uint32_t gateup_overlay_sentinel;
    uint64_t down_native_code_sidecar_table_offset;
    uint32_t down_native_code_sidecar_record_bytes;
    uint32_t down_native_code_sidecar_records;
    uint32_t down_native_code_sidecar_count;
    bool gateup_rowblock_overlay;
    bool rank1_residual_sidecars;
    bool sidecar_required;
    bool sidecar_none_selected;
    ds4_d8f_record records[DS4_D8F_PROJECTION_COUNT][256];
    ds4_d8f_sidecar_record down_sidecars[256];
    ds4_d8f_native_code_record down_native_codes[256];
} ds4_d8f_file;

bool ds4_d8f_open(const char *path, ds4_d8f_file *file);
void ds4_d8f_close(ds4_d8f_file *file);
bool ds4_d8f_get_record(const ds4_d8f_file *file, ds4_d8f_projection projection, uint32_t expert, ds4_d8f_record *out);
bool ds4_d8f_get_gateup_overlay_record(const ds4_d8f_file *file, ds4_d8f_projection projection, uint32_t expert, uint32_t row_block, ds4_d8f_record *out);
bool ds4_d8f_get_down_sidecar(const ds4_d8f_file *file, uint32_t expert, ds4_d8f_sidecar_record *out);
bool ds4_d8f_get_down_native_codes(const ds4_d8f_file *file, uint32_t expert, ds4_d8f_native_code_record *out);
uint32_t ds4_d8f_down_sidecar_count(const ds4_d8f_file *file);
uint32_t ds4_d8f_down_native_code_sidecar_count(const ds4_d8f_file *file);
uint32_t ds4_d8f_gateup_overlay_count(const ds4_d8f_file *file);
uint32_t ds4_d8f_code_at(const ds4_d8f_file *file, const ds4_d8f_record *record, uint64_t block_index);
void ds4_d8f_print_summary(const ds4_d8f_file *file);

#endif
