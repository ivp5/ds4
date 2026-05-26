/* ds4_polar_reader.h — PLR2 combined polar-encoded expert-weight reader.
 *
 * Zero-copy mmap views into files produced by analyzers/polar_encode_mlx.py.
 * Caller does not own the byte storage and must not write into returned ptrs.
 *
 * On-disk layout (one file per layer × kind):
 *   bytes   field
 *    0..3   magic = "PLR2"
 *    4..7   version = 1 (uint32 LE)
 *    8..11  n_experts (uint32 LE)
 *   12..15  n_rows    (uint32 LE)
 *   16..19  n_pairs   (uint32 LE)
 *   20..23  layer     (uint32 LE)
 *   24..27  kind_id   (uint32 LE)  — 0=gate, 1=up, 2=down
 *   28..31  phase_levels (uint32 LE; 0 in legacy files → interpret as 8)
 *   32..35  mag_levels   (uint32 LE; 0 in legacy files → interpret as 4)
 *   36..63  reserved (zeros)
 *   64..       mag_codes   uint8   [n_experts, n_rows, n_pairs]
 *   ...        phase_codes uint8   [n_experts, n_rows, n_pairs]
 *   ...        levels      float32 [n_experts, n_rows, mag_levels]
 */
#ifndef DS4_POLAR_READER_H
#define DS4_POLAR_READER_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* PLR2 file handle. Opaque struct; read-only after ds4_polar_open(). */
typedef struct {
    void    *mmap_base;     /* base of mmap region, do not free */
    size_t   mmap_bytes;
    int      fd;            /* kept open until close() */
    uint32_t version;
    uint32_t n_experts;
    uint32_t n_rows;
    uint32_t n_pairs;
    uint32_t layer;
    uint32_t kind_id;       /* 0=gate, 1=up, 2=down */
    uint32_t phase_levels;  /* 8 = legacy default */
    uint32_t mag_levels;    /* 4 = legacy default */
    /* Cached pointer arithmetic — recomputed on open(). */
    const uint8_t *mag_base;     /* [n_experts × n_rows × n_pairs] u8 */
    const uint8_t *phase_base;   /* [n_experts × n_rows × n_pairs] u8 */
    const float   *levels_base;  /* [n_experts × n_rows × mag_levels] f32 */
    size_t    expert_mag_stride;     /* n_rows × n_pairs */
    size_t    expert_phase_stride;   /* n_rows × n_pairs */
    size_t    expert_levels_stride;  /* n_rows × mag_levels */
} ds4_polar_file;

enum {
    DS4_POLAR_KIND_GATE = 0,
    DS4_POLAR_KIND_UP   = 1,
    DS4_POLAR_KIND_DOWN = 2,
};

/* Open / close. open() returns false on any error and zero-inits *out. */
bool ds4_polar_open(const char *path, ds4_polar_file *out);
void ds4_polar_close(ds4_polar_file *p);

/* Zero-copy per-expert view. Out-pointers may be NULL. False on bad idx. */
bool ds4_polar_expert_view(const ds4_polar_file *p, uint32_t expert_idx,
                           const uint8_t **mag_out, const uint8_t **phase_out,
                           const float **levels_out);

/* Decode one (row, pair) sample for spot-checking. Returns 0.0 on bad idx. */
float ds4_polar_decode_pair_re(const ds4_polar_file *p, uint32_t expert_idx,
                               uint32_t row, uint32_t pair);
float ds4_polar_decode_pair_im(const ds4_polar_file *p, uint32_t expert_idx,
                               uint32_t row, uint32_t pair);

/* Diagnostic: header + sample decoded values + summary to stderr. */
void ds4_polar_print_summary(const ds4_polar_file *p, const char *label);


/* ============================================================================
 * Polar pool — collection of PLR2 files keyed by (layer, kind).
 *
 * Layout assumption: files named `L{LL:02}_{kind}.polar`
 * (kind ∈ {gate, up, down}). Statically sized for DS4_POLAR_MAX_LAYERS × 3.
 * Pool does NOT own the directory; mmap regions stay valid until
 * ds4_polar_pool_close(). At default 128-row encoding for DS4 V4 Flash,
 * each (layer, kind) is ~135 MB for gate/up and ~67 MB for down; OS pages
 * in only touched ranges.
 * ============================================================================ */

#ifndef DS4_N_LAYER
#define DS4_POLAR_MAX_LAYERS 64   /* must be ≥ DS4_N_LAYER */
#else
#define DS4_POLAR_MAX_LAYERS DS4_N_LAYER
#endif

typedef struct {
    ds4_polar_file files[DS4_POLAR_MAX_LAYERS][3]; /* [layer][kind_id] */
    uint32_t       opened_count;
    uint64_t       total_bytes;
} ds4_polar_pool;

void     ds4_polar_pool_init     (ds4_polar_pool *pool);
/* Scan `dir`, open each L{LL:02}_{kind}.polar. Returns successful-open count.
 * Bad files are skipped with a stderr warning. Idempotent: re-load closes
 * prior opens for any (layer, kind) being replaced. */
uint32_t ds4_polar_pool_load_dir (ds4_polar_pool *pool, const char *dir);
void     ds4_polar_pool_close    (ds4_polar_pool *pool);
/* Lookup. Returns NULL if no file open at (layer, kind). */
const ds4_polar_file *ds4_polar_pool_get(const ds4_polar_pool *pool,
                                         uint32_t layer, uint32_t kind);
void ds4_polar_pool_print_summary(const ds4_polar_pool *pool, const char *label);

#ifdef __cplusplus
}
#endif

#endif /* DS4_POLAR_READER_H */
