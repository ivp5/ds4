/* ds4_ridgegptq_reader.h — scalar ridge-GPTQ routed expert packet reader.
 *
 * RGQ2 stores activation-aware scalar GPTQ weights without a VQ codebook:
 *   weight(expert,row,col) = scale(expert,row,col_group) * value[code]
 *
 * File layout, little-endian:
 *   header, 64 bytes:
 *     0-3    magic "RGQ2"
 *     4-7    version (=1)
 *     8-11   n_experts
 *     12-15  n_rows
 *     16-19  n_cols
 *     20-23  layer
 *     24-27  kind_id       0=gate, 1=up, 2=down
 *     28-31  row_start
 *     32-35  bit_width     1 or 2
 *     36-39  q_levels      <= 1<<bit_width
 *     40-43  scale_group_cols
 *     44-47  flags         reserved, must be 0 for v1
 *     48-55  n_codes       n_experts * n_rows * n_cols
 *     56-63  reserved      must be 0
 *   values:  q_levels float32 values
 *   scales:  n_experts * n_rows * ceil(n_cols/scale_group_cols) float32
 *   codes:   ceil(n_codes * bit_width / 8) bytes, little-bit-endian stream
 *
 * The format is deliberately scalar and group-scaled so it can host the
 * current champion family: binary, ternary-in-2bit, and int2 ridge-GPTQ,
 * plus later entropy coding outside the runtime-critical matvec path.
 */
#ifndef DS4_RIDGEGPTQ_READER_H
#define DS4_RIDGEGPTQ_READER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_RGPTQ_MAGIC "RGQ2"
#define DS4_RGPTQ_VERSION 1u
#define DS4_RGPTQ_HEADER_BYTES 64u

typedef enum {
    DS4_RGPTQ_KIND_GATE = 0,
    DS4_RGPTQ_KIND_UP   = 1,
    DS4_RGPTQ_KIND_DOWN = 2,
} ds4_rgptq_kind_t;

typedef struct ds4_rgptq_file {
    int fd;
    void *map;
    size_t map_size;
    uint32_t version;
    uint32_t n_experts;
    uint32_t n_rows;
    uint32_t n_cols;
    uint32_t layer;
    uint32_t kind_id;
    uint32_t row_start;
    uint32_t bit_width;
    uint32_t q_levels;
    uint32_t scale_group_cols;
    uint32_t scale_groups;
    uint32_t flags;
    uint64_t n_codes;
    uint8_t code_mask;
    const float *values;
    const float *scales;
    const uint8_t *codes;
    size_t values_bytes;
    size_t scales_bytes;
    size_t codes_bytes;
} ds4_rgptq_file;

bool ds4_rgptq_open(const char *path, ds4_rgptq_file *out);
void ds4_rgptq_close(ds4_rgptq_file *file);

uint32_t ds4_rgptq_get_code(const ds4_rgptq_file *file,
                            uint32_t expert, uint32_t row, uint32_t col);

float ds4_rgptq_get_weight(const ds4_rgptq_file *file,
                           uint32_t expert, uint32_t row, uint32_t col);

bool ds4_rgptq_dequant_row(const ds4_rgptq_file *file,
                           uint32_t expert, uint32_t row,
                           float *out_row, uint32_t out_cols);

float ds4_rgptq_matvec_row(const ds4_rgptq_file *file,
                           uint32_t expert, uint32_t row,
                           const float *input, uint32_t input_cols);

bool ds4_rgptq_matvec_expert(const ds4_rgptq_file *file,
                             uint32_t expert,
                             const float *input, uint32_t input_cols,
                             float *output, uint32_t output_rows);

void ds4_rgptq_print_summary(const ds4_rgptq_file *file);

static inline size_t ds4_rgptq_codes_bytes(uint64_t n_codes, uint32_t bit_width) {
    return (size_t)((n_codes * (uint64_t)bit_width + 7u) >> 3);
}

#ifdef __cplusplus
}
#endif

#endif
