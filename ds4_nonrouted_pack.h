/* ds4_nonrouted_pack.h — DS4 V4 non-routed weights pack reader (silv 2026-05-28).
 *
 * Companion to the existing VQB2 routed-FFN pack. The non-routed pack
 * contains everything else: attention (Q/K/V/O), shared expert,
 * embedding, head, layer norms, router gate weights, hadamard rotation
 * factors (hc_*), and the MTP block.
 *
 * Source: derived from DS4 V4 bf16 safetensors via
 *   /Users/silv/cl/tlp/montyneg/ds4/nonrouted/pack_nonrouted.py
 *
 * Format (DS4NRPK1):
 *   [128 B] header
 *   [manifest_bytes] JSON manifest:
 *     [{"name": str, "dtype": str, "shape": [int...],
 *       "data_off": int, "data_bytes": int}, ...]
 *   [data_bytes] tensor data, 64-byte aligned per tensor
 *
 * dtype strings: "BF16", "F16", "F32", "I8", "F8_E4M3", "F8_E8M0"
 */
#ifndef DS4_NONROUTED_PACK_H
#define DS4_NONROUTED_PACK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_NRPK_MAGIC           "DS4NRPK1"
#define DS4_NRPK_VERSION         1u
#define DS4_NRPK_MAX_NAME_LEN    160u
#define DS4_NRPK_MAX_DIMS        8u

typedef enum ds4_nrpk_dtype {
    DS4_NRPK_DTYPE_UNKNOWN  = 0,
    DS4_NRPK_DTYPE_F32      = 1,
    DS4_NRPK_DTYPE_F16      = 2,
    DS4_NRPK_DTYPE_BF16     = 3,
    DS4_NRPK_DTYPE_I8       = 4,
    DS4_NRPK_DTYPE_F8_E4M3  = 5,  /* float8 e4m3fn */
    DS4_NRPK_DTYPE_F8_E8M0  = 6,  /* float8 e8m0fnu (scale exponent) */
} ds4_nrpk_dtype;

typedef struct ds4_nrpk_header {
    char     magic[8];          /* "DS4NRPK1" */
    uint32_t version;
    uint32_t n_tensors;
    uint64_t manifest_offset;
    uint64_t manifest_bytes;
    uint64_t data_offset;
    uint64_t data_bytes;
    uint64_t total_bytes;
    uint32_t pad[18];
} ds4_nrpk_header;               /* 128 bytes */

typedef struct ds4_nrpk_entry {
    char           name[DS4_NRPK_MAX_NAME_LEN];
    ds4_nrpk_dtype dtype;
    uint32_t       n_dims;
    uint32_t       dims[DS4_NRPK_MAX_DIMS];
    uint64_t       data_off;     /* relative to data_offset */
    uint64_t       data_bytes;
} ds4_nrpk_entry;

typedef struct ds4_nrpk {
    int      fd;
    void    *map;
    size_t   map_size;
    const ds4_nrpk_header *hdr;
    const uint8_t *data_arena;   /* points to data section in mmap */

    ds4_nrpk_entry *entries;     /* sorted by name for binary lookup */
    uint32_t        n_entries;
    char            pack_path[1024];
} ds4_nrpk;

/* Open pack file + parse manifest JSON + build sorted entries. */
bool ds4_nrpk_open(const char *pack_path, ds4_nrpk *out);

/* Close pack + free entries. */
void ds4_nrpk_close(ds4_nrpk *p);

/* O(log n) name lookup. Returns NULL if missing. */
const ds4_nrpk_entry *ds4_nrpk_lookup(const ds4_nrpk *p, const char *name);

/* Get raw byte pointer to a tensor's data (mmapped, read-only). */
const void *ds4_nrpk_get_data(const ds4_nrpk *p, const ds4_nrpk_entry *e);

/* dtype string → enum (for parsing). */
ds4_nrpk_dtype ds4_nrpk_dtype_from_string(const char *s);

/* dtype enum → string (for printing). */
const char *ds4_nrpk_dtype_name(ds4_nrpk_dtype d);

/* Bytes per element for the dtype. */
size_t ds4_nrpk_dtype_bytes(ds4_nrpk_dtype d);

/* Print summary to stderr. */
void ds4_nrpk_print_summary(const ds4_nrpk *p);

/* Convenience: count entries matching a name prefix (e.g. "layers.22."). */
uint32_t ds4_nrpk_count_prefix(const ds4_nrpk *p, const char *prefix);

#ifdef __cplusplus
}
#endif

#endif /* DS4_NONROUTED_PACK_H */
