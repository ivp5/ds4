/* ds4_vqb2_pack.h — pack-backed VQB2 view loader (silv 2026-05-28).
 *
 * Backs codex H2116-H2120 + H2121 + H2125: thousands of per-file VQB2 packets
 * carry a 25-39× packet-abstraction tax; the deployable substrate is a single
 * mmap pack file + offset index + materialized ds4_vqb2_file views that share
 * the pack's mmap region.
 *
 * Pack layout (raw byte concatenation):
 *   bytes [0, pack_size) : raw VQB2 packets back-to-back, each at pack_offset.
 *
 * Index CSV (columns required):
 *   layer, kind_id, k, row_start, pack_offset, packet_bytes,
 *   header_n_experts, header_n_rows, header_n_pairs, header_bit_width,
 *   header_n_codes
 * Other columns ignored (provenance/manifest metadata).
 *
 * Lookup contract: each (layer, kind_id, row_start) is unique in one pack
 * (one resolved policy per cell). row_start is always a multiple of 128.
 *
 * View materialization: ds4_vqb2_pack_get_view fills a ds4_vqb2_file struct
 * with fd=-1, map=NULL, and codebook/codes pointers into the pack's mmap.
 * Calling ds4_vqb2_close on a view is safe (no-op munmap, no-op close); the
 * pack owns the mmap and is responsible for freeing it via ds4_vqb2_pack_close.
 */
#ifndef DS4_VQB2_PACK_H
#define DS4_VQB2_PACK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ds4_vqb2_reader.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_VQB2_PACK_MAX_LAYER 43u
#define DS4_VQB2_PACK_N_KIND     3u   /* gate/up/down */
#define DS4_VQB2_PACK_MAX_ROW_BLOCKS 32u /* down has 32 blocks (4096/128) */

typedef struct ds4_vqb2_pack_entry {
    uint32_t layer;
    uint32_t kind_id;     /* 0=gate, 1=up, 2=down */
    uint32_t k;           /* 4/16/256 */
    uint32_t row_start;
    uint64_t pack_offset;
    uint64_t packet_bytes;
    uint32_t n_experts;
    uint32_t n_rows;
    uint32_t n_pairs;
    uint32_t bit_width;
    uint64_t n_codes;
} ds4_vqb2_pack_entry;

typedef struct ds4_vqb2_pack {
    int      fd;            /* pack file fd; -1 when closed */
    void    *map;           /* mmap base */
    size_t   map_size;
    uint64_t pack_size;
    ds4_vqb2_pack_entry *entries;
    uint32_t n_entries;
    /* O(1) lookup: [layer][kind_id][row_block_idx]. row_block_idx = row_start/128.
     * Value = index into entries, or -1 if absent. */
    int32_t *lookup;
    /* Pack metadata */
    char     pack_path[1024];
} ds4_vqb2_pack;

/* Open pack file + parse index CSV. Returns true on success.
 * Both paths must exist. */
bool ds4_vqb2_pack_open(const char *pack_path,
                        const char *index_csv_path,
                        ds4_vqb2_pack *out);

/* Close pack + free index. Safe on partial/zero struct. */
void ds4_vqb2_pack_close(ds4_vqb2_pack *p);

/* O(1) lookup: returns index into p->entries, or -1 if absent. */
int32_t ds4_vqb2_pack_lookup_index(const ds4_vqb2_pack *p,
                                   uint32_t layer, uint32_t kind_id,
                                   uint32_t row_start);

/* Materialize a ds4_vqb2_file view backed by the pack's mmap.
 * The view's fd = -1 and map = NULL, so ds4_vqb2_close on it is a no-op.
 * Returns false if the (layer, kind_id, row_start) cell is absent. */
bool ds4_vqb2_pack_get_view(const ds4_vqb2_pack *p,
                            uint32_t layer, uint32_t kind_id, uint32_t row_start,
                            ds4_vqb2_file *out_view);

/* Materialize a view directly from a known entry (cheaper than re-lookup). */
bool ds4_vqb2_pack_view_from_entry(const ds4_vqb2_pack *p,
                                   uint32_t entry_index,
                                   ds4_vqb2_file *out_view);

/* H2125 layer-packet locality: iterate every entry for one layer in
 * (kind, row_start)-major order and invoke the callback with a freshly
 * materialized view. The view is valid only during the callback. Returns
 * the number of entries visited; -1 on error.
 *
 * Each routed layer event touches at most 64 packets (16 gate + 16 up + 32
 * down for the standard DS4 V4 shape). With this iterator, decode dispatch
 * can walk the layer's resident packets in cache-friendly order. */
typedef int (*ds4_vqb2_pack_layer_cb)(uint32_t entry_idx,
                                      const ds4_vqb2_pack_entry *entry,
                                      const ds4_vqb2_file *view,
                                      void *userdata);
int ds4_vqb2_pack_for_layer(const ds4_vqb2_pack *p,
                            uint32_t layer,
                            ds4_vqb2_pack_layer_cb cb,
                            void *userdata);

/* Print summary to stderr: pack size, n_entries, K mix, kind mix, layer span. */
void ds4_vqb2_pack_print_summary(const ds4_vqb2_pack *p);

/* silv 2026-05-28 — pack-aware hot-store loader (cache-audit fix A1).
 *
 * Iterates the pack and pins all (layer, kind, expert, row_block) tiles
 * into the hot store. The CSV index entry's row_start / 128 supplies the
 * row_block parameter that ds4_hot_pin_expert_from_vqb2 needs to avoid
 * the silent-overwrite bug captured in tmp/20260528_cache_audit/AUDIT.md.
 *
 * `layer_filter` = -1 pins all layers; else only the specified layer.
 * `kind_filter`  = -1 pins all kinds; else only the specified kind (0/1/2).
 *
 * Returns number of (expert × row_block) pins performed, -1 on hard error.
 *
 * Forward-declares struct ds4_hot_expert_store to avoid header coupling. */
struct ds4_hot_expert_store;
int ds4_vqb2_pack_load_to_hot_store(struct ds4_hot_expert_store *store,
                                    const ds4_vqb2_pack *p,
                                    int layer_filter, int kind_filter);

#ifdef __cplusplus
}
#endif

#endif /* DS4_VQB2_PACK_H */
