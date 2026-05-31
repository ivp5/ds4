#ifndef DS4_D8M_READER_H
#define DS4_D8M_READER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t expert;
    uint32_t k;
    uint32_t bits;
    uint32_t block;
    uint64_t codebook_offset;
    uint64_t index_offset;
    uint32_t codebook_bytes;
    uint32_t index_bytes;
} ds4_d8m_record;

typedef struct {
    const uint8_t *map;
    size_t size;
    int fd;
    uint32_t version;
    uint32_t header_json_bytes;
    ds4_d8m_record records[256];
} ds4_d8m_file;

bool ds4_d8m_open(const char *path, ds4_d8m_file *file);
void ds4_d8m_close(ds4_d8m_file *file);
bool ds4_d8m_get_record(const ds4_d8m_file *file, uint32_t expert, ds4_d8m_record *out);
uint32_t ds4_d8m_code_at(const ds4_d8m_file *file, const ds4_d8m_record *record, uint64_t block_index);
void ds4_d8m_print_summary(const ds4_d8m_file *file);

#endif
