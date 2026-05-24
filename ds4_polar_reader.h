/* ds4_polar_reader.h
 *
 * Read PLR2 combined polar-encoded expert weight files produced by
 * analyzers/polar_encode_mlx.py.  Zero-copy mmap views; the caller does
 * not own the byte storage and must not write into the returned pointers.
 *
 * On-disk layout (one file per layer × kind):
 *
 *   bytes  field
 *   0..3   magic = "PLR2"
 *   4..7   version = 1 (uint32 LE)
 *   8..11  n_experts (uint32 LE)
 *   12..15 n_rows    (uint32 LE)
 *   16..19 n_pairs   (uint32 LE)
 *   20..23 layer     (uint32 LE)
 *   24..27 kind_id   (uint32 LE) — 0=gate,1=up,2=down
 *   28..63 reserved (zeros)
 *   64..       mag_codes   uint8   [n_experts, n_rows, n_pairs]
 *   ...        phase_codes uint8   [n_experts, n_rows, n_pairs]
 *   ...        levels      float32 [n_experts, n_rows, 4]
 *
 * Phase A of task #563: provide host-side read access first; GPU
 * MTLResidencySet binding follows in Phase B once correctness against
 * CPU reference is established. */

#ifndef DS4_POLAR_READER_H
#define DS4_POLAR_READER_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* PLR2 file handle.  Opaque; do not access fields directly except as
 * documented below.  Fields are read-only after ds4_polar_open(). */
typedef struct {
    void    *mmap_base;     /* base of mmap region, do not free */
    size_t   mmap_bytes;    /* total mmap length passed to munmap()  */
    int      fd;            /* keep open until close()               */
    uint32_t version;
    uint32_t n_experts;
    uint32_t n_rows;
    uint32_t n_pairs;
    uint32_t layer;
    uint32_t kind_id;       /* 0=gate, 1=up, 2=down */
    /* Cached pointer arithmetic — recomputed on open() so callers don't
     * have to walk the header layout. */
    const uint8_t *mag_base;     /* [n_experts × n_rows × n_pairs] u8 */
    const uint8_t *phase_base;   /* [n_experts × n_rows × n_pairs] u8 */
    const float   *levels_base;  /* [n_experts × n_rows × 4]      f32 */
    size_t    expert_mag_stride;     /* n_rows × n_pairs   */
    size_t    expert_phase_stride;   /* n_rows × n_pairs   */
    size_t    expert_levels_stride;  /* n_rows × 4         */
} ds4_polar_file;

/* Kind ID mapping (matches the Python encoder). */
enum {
    DS4_POLAR_KIND_GATE = 0,
    DS4_POLAR_KIND_UP   = 1,
    DS4_POLAR_KIND_DOWN = 2,
};

/* Open a PLR2 file.  Returns true on success, false on any error (file
 * missing, bad magic, version mismatch, header inconsistency).  On
 * failure, *out is zero-initialized.  Caller MUST pair every successful
 * open() with close() to release the fd + mmap. */
bool ds4_polar_open(const char *path, ds4_polar_file *out);

/* Close a previously-opened PLR2 file. Safe to call on a zeroed handle. */
void ds4_polar_close(ds4_polar_file *p);

/* Get zero-copy views for a single expert slot.  Out-pointers may be NULL
 * if the caller doesn't need that field.  Returns false if expert_idx is
 * out of range. */
bool ds4_polar_expert_view(const ds4_polar_file *p, uint32_t expert_idx,
                            const uint8_t **mag_out,
                            const uint8_t **phase_out,
                            const float   **levels_out);

/* Decode a single (row, pair) sample back to float32 — for spot-checking
 * against CPU reference and printing.  Returns 0.0 on bad indices. */
float ds4_polar_decode_pair_re(const ds4_polar_file *p,
                                uint32_t expert_idx,
                                uint32_t row, uint32_t pair);
float ds4_polar_decode_pair_im(const ds4_polar_file *p,
                                uint32_t expert_idx,
                                uint32_t row, uint32_t pair);

/* Diagnostic: print header + a few sample decoded values + summary stats
 * to stderr.  Useful for sanity-checking a freshly-encoded shard before
 * wiring it into inference. */
void ds4_polar_print_summary(const ds4_polar_file *p, const char *label);

#ifdef __cplusplus
}
#endif

#endif /* DS4_POLAR_READER_H */
