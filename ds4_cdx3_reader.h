#ifndef DS4_CDX3_READER_H
#define DS4_CDX3_READER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_CDX3I_MAGIC "CDX3IDX"
#define DS4_CDX3I_VERSION 2u
#define DS4_CDX3I_HEADER_BYTES 1260u
#define DS4_CDX3I_RECORD_BYTES 64u

typedef enum {
    DS4_CDX3_KIND_GATE = 0,
    DS4_CDX3_KIND_UP   = 1,
    DS4_CDX3_KIND_DOWN = 2,
} ds4_cdx3_kind_t;

typedef struct ds4_cdx3_record {
    uint16_t layer;
    uint16_t expert;
    uint16_t kind;
    uint16_t bits;
    uint32_t out_dim;
    uint32_t in_dim;
    uint32_t n_indices;
    uint32_t scale_count;
    uint64_t payload_offset;
    uint64_t framed_bytes;
    uint64_t scale_offset;
    uint64_t index_offset;
    float scale_log_min;
    float scale_log_step;
} ds4_cdx3_record;

typedef struct ds4_cdx3_file {
    int pack_fd;
    int index_fd;
    void *pack_map;
    void *index_map;
    size_t pack_size;
    size_t index_size;
    uint32_t version;
    uint32_t n_records;
    uint32_t records_total;
    uint32_t complete;
    uint32_t d;
    uint32_t group;
    uint64_t first_record_offset;
    uint64_t indexed_pack_size;
    uint64_t created_epoch;
    const uint32_t *k_by_layer;
    const uint64_t *codebook_offsets;
    const uint8_t *records;
} ds4_cdx3_file;

bool ds4_cdx3_open(const char *pack_path, const char *index_path, ds4_cdx3_file *out);
void ds4_cdx3_close(ds4_cdx3_file *file);
bool ds4_cdx3_get_record(const ds4_cdx3_file *file, uint32_t layer, uint32_t expert, uint32_t kind, ds4_cdx3_record *out);
uint32_t ds4_cdx3_get_code(const ds4_cdx3_file *file, const ds4_cdx3_record *record, uint32_t block_index);
float ds4_cdx3_get_scale(const ds4_cdx3_file *file, const ds4_cdx3_record *record, uint32_t block_index);
bool ds4_cdx3_decode_block(const ds4_cdx3_file *file, const ds4_cdx3_record *record, uint32_t block_index, float out_values[8]);
void ds4_cdx3_print_summary(const ds4_cdx3_file *file);

#ifdef __cplusplus
}
#endif

#endif
