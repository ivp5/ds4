#ifndef DS4_D8FS_READER_H
#define DS4_D8FS_READER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct ds4_d8fs_record {
    uint32_t expert;
    uint32_t rows;
    uint32_t groups;
    uint32_t k;
    uint32_t unique_count;
    uint32_t max_group_unique;
    uint64_t group_prefix_offset;
    uint64_t inverse_offset_table_offset;
    uint64_t unique_code_offset;
    uint64_t inverse_bits_offset;
    uint32_t group_prefix_bytes;
    uint32_t inverse_offset_table_bytes;
    uint32_t unique_code_bytes;
    uint32_t inverse_bits_bytes;
    uint32_t inverse_bits_bits;
    uint32_t flags;
} ds4_d8fs_record;

typedef struct ds4_d8fs_file {
    int fd;
    const uint8_t *map;
    size_t size;
    uint32_t version;
    uint32_t header_json_bytes;
    uint32_t record_bytes;
    uint32_t records_count;
    uint32_t layer;
    uint32_t live_records;
    uint64_t unique_total;
    uint64_t sparse_bytes;
    uint64_t native_equivalent_bytes;
    ds4_d8fs_record records[256];
} ds4_d8fs_file;

bool ds4_d8fs_open(const char *path, ds4_d8fs_file *file);
void ds4_d8fs_close(ds4_d8fs_file *file);
bool ds4_d8fs_get_record(const ds4_d8fs_file *file, uint32_t expert, ds4_d8fs_record *out);
bool ds4_d8fs_code_at(const ds4_d8fs_file *file,
                      const ds4_d8fs_record *record,
                      uint32_t row,
                      uint32_t group,
                      uint16_t *out_code);
void ds4_d8fs_print_summary(const ds4_d8fs_file *file);

#endif
