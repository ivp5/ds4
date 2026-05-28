/* ds4_watersic_pack.c — pack-backed WaterSIC view loader (silv 2026-05-28).
 *
 * See ds4_watersic_pack.h for layout + design.
 */
#include "ds4_watersic_pack.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static void ds4_ws_log(const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "ds4_watersic: ");
    __builtin_va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    __builtin_va_end(ap);
    fputc('\n', stderr);
}

bool ds4_watersic_pack_open(const char *pack_path, ds4_watersic_pack *out) {
    if (!pack_path || !out) return false;
    memset(out, 0, sizeof(*out));
    out->fd = -1;
    strncpy(out->pack_path, pack_path, sizeof(out->pack_path) - 1);

    int fd = open(pack_path, O_RDONLY);
    if (fd < 0) {
        ds4_ws_log("open(%s) failed: %s", pack_path, strerror(errno));
        return false;
    }
    struct stat st;
    if (fstat(fd, &st) < 0) {
        ds4_ws_log("fstat failed: %s", strerror(errno));
        close(fd);
        return false;
    }
    if ((size_t)st.st_size < sizeof(ds4_watersic_header)) {
        ds4_ws_log("file too small: %lld bytes < %zu header bytes",
                   (long long)st.st_size, sizeof(ds4_watersic_header));
        close(fd);
        return false;
    }
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        ds4_ws_log("mmap failed: %s", strerror(errno));
        close(fd);
        return false;
    }

    const ds4_watersic_header *hdr = (const ds4_watersic_header *)map;
    if (hdr->magic != DS4_WATERSIC_MAGIC) {
        ds4_ws_log("bad magic 0x%08x (expected 0x%08x)",
                   hdr->magic, DS4_WATERSIC_MAGIC);
        munmap(map, (size_t)st.st_size);
        close(fd);
        return false;
    }
    if (hdr->version != DS4_WATERSIC_VERSION) {
        ds4_ws_log("unsupported version %u (this code supports %u)",
                   hdr->version, DS4_WATERSIC_VERSION);
        munmap(map, (size_t)st.st_size);
        close(fd);
        return false;
    }
    if (hdr->R != 2u && hdr->R != 4u && hdr->R != 8u) {
        ds4_ws_log("R=%u not supported (must be one of 2, 4, 8)", hdr->R);
        munmap(map, (size_t)st.st_size);
        close(fd);
        return false;
    }
    if (hdr->n_layers > DS4_WATERSIC_MAX_LAYERS ||
        hdr->n_kinds != DS4_WATERSIC_N_KINDS) {
        ds4_ws_log("bad dims: n_layers=%u n_kinds=%u",
                   hdr->n_layers, hdr->n_kinds);
        munmap(map, (size_t)st.st_size);
        close(fd);
        return false;
    }
    if (hdr->total_pack_bytes != (uint64_t)st.st_size) {
        ds4_ws_log("size mismatch: header says %llu, file is %lld",
                   (unsigned long long)hdr->total_pack_bytes,
                   (long long)st.st_size);
        munmap(map, (size_t)st.st_size);
        close(fd);
        return false;
    }

    const uint8_t *base = (const uint8_t *)map;
    out->fd          = fd;
    out->map         = map;
    out->map_size    = (size_t)st.st_size;
    out->hdr         = hdr;
    out->lk_table    = (const ds4_watersic_lk_header *)(base + hdr->lk_table_offset);
    out->alpha_arena = base + hdr->alpha_arena_offset;
    out->codes_arena = base + hdr->codes_arena_offset;
    out->n_lk        = hdr->n_layers * hdr->n_kinds;

    /* Validate per-(layer,kind) headers + offsets. */
    if ((uint64_t)hdr->lk_table_offset +
        (uint64_t)out->n_lk * sizeof(ds4_watersic_lk_header) > out->map_size) {
        ds4_ws_log("lk_table OOB");
        munmap(map, out->map_size);
        close(fd);
        memset(out, 0, sizeof(*out));
        out->fd = -1;
        return false;
    }

    /* Build O(1) lookup. */
    out->lookup = (int32_t *)malloc(out->n_lk * sizeof(int32_t));
    if (!out->lookup) {
        ds4_ws_log("lookup alloc failed");
        munmap(map, out->map_size);
        close(fd);
        memset(out, 0, sizeof(*out));
        out->fd = -1;
        return false;
    }
    for (uint32_t i = 0; i < out->n_lk; i++) out->lookup[i] = -1;
    for (uint32_t i = 0; i < out->n_lk; i++) {
        const ds4_watersic_lk_header *e = &out->lk_table[i];
        if (e->layer >= hdr->n_layers || e->kind >= hdr->n_kinds) {
            ds4_ws_log("lk entry %u has bad (layer=%u, kind=%u)",
                       i, e->layer, e->kind);
            free(out->lookup);
            munmap(map, out->map_size);
            close(fd);
            memset(out, 0, sizeof(*out));
            out->fd = -1;
            return false;
        }
        const uint32_t slot = e->layer * hdr->n_kinds + e->kind;
        if (out->lookup[slot] >= 0) {
            ds4_ws_log("duplicate lk entry for (layer=%u, kind=%u): "
                       "first at %d, this at %u",
                       e->layer, e->kind, out->lookup[slot], i);
            free(out->lookup);
            munmap(map, out->map_size);
            close(fd);
            memset(out, 0, sizeof(*out));
            out->fd = -1;
            return false;
        }
        out->lookup[slot] = (int32_t)i;
    }
    return true;
}

void ds4_watersic_pack_close(ds4_watersic_pack *p) {
    if (!p) return;
    if (p->map && p->map_size > 0) {
        munmap(p->map, p->map_size);
    }
    if (p->fd >= 0) close(p->fd);
    free(p->lookup);
    memset(p, 0, sizeof(*p));
    p->fd = -1;
}

int32_t ds4_watersic_pack_lookup_index(const ds4_watersic_pack *p,
                                       uint32_t layer, uint32_t kind) {
    if (!p || !p->lookup || !p->hdr) return -1;
    if (layer >= p->hdr->n_layers || kind >= p->hdr->n_kinds) return -1;
    return p->lookup[layer * p->hdr->n_kinds + kind];
}

bool ds4_watersic_pack_get(const ds4_watersic_pack *p,
                           uint32_t layer, uint32_t kind, uint32_t expert,
                           const float  **out_alpha,
                           const uint8_t **out_codes,
                           uint32_t      *out_n_rows,
                           uint32_t      *out_n_cols) {
    if (!p || !out_alpha || !out_codes || !out_n_rows || !out_n_cols)
        return false;
    if (expert >= p->hdr->n_experts) return false;

    int32_t idx = ds4_watersic_pack_lookup_index(p, layer, kind);
    if (idx < 0) return false;

    const ds4_watersic_lk_header *lk = &p->lk_table[idx];
    *out_n_rows = lk->n_rows;
    *out_n_cols = lk->n_cols;
    *out_alpha  = (const float *)(p->alpha_arena + lk->alpha_offset);
    *out_codes  = p->codes_arena + lk->codes_offset + (uint64_t)expert * lk->codes_stride;
    return true;
}

int ds4_watersic_unpack_code(const uint8_t *codes_row, uint32_t code_idx, uint32_t R) {
    /* Decode one signed R-bit code. Codes are packed little-endian
     * (low bits first within each byte). Two's complement signed. */
    const uint32_t mask = (1u << R) - 1u;
    const uint32_t bit_off = code_idx * R;
    const uint32_t byte_off = bit_off >> 3u;
    const uint32_t shift = bit_off & 7u;
    uint32_t window = (uint32_t)codes_row[byte_off];
    if (R + shift > 8u) {
        window |= ((uint32_t)codes_row[byte_off + 1u]) << 8u;
    }
    uint32_t u = (window >> shift) & mask;
    /* Sign-extend from R bits. */
    const uint32_t sign_bit = 1u << (R - 1u);
    int v = (int)u;
    if (u & sign_bit) v -= (int)(1u << R);
    return v;
}

void ds4_watersic_pack_dequant_row(const ds4_watersic_pack *p,
                                   const float  *alpha,
                                   const uint8_t *codes,
                                   uint32_t n_cols,
                                   uint32_t row,
                                   float *out_w_row) {
    if (!p || !alpha || !codes || !out_w_row) return;
    const uint32_t R = p->hdr->R;
    const float a = alpha[row];
    const uint32_t row_codes_bytes = (n_cols * R + 7u) / 8u;
    const uint8_t *codes_row = codes + (uint64_t)row * row_codes_bytes;
    for (uint32_t j = 0; j < n_cols; j++) {
        const int z = ds4_watersic_unpack_code(codes_row, j, R);
        out_w_row[j] = a * (float)z;
    }
}

float ds4_watersic_pack_matvec_row(const ds4_watersic_pack *p,
                                   const float  *alpha,
                                   const uint8_t *codes_row,
                                   const float  *x,
                                   uint32_t n_cols,
                                   uint32_t row) {
    if (!p || !alpha || !codes_row || !x) return 0.0f;
    const uint32_t R = p->hdr->R;
    double acc = 0.0;  /* fp64 for tight numerical match in canary cross-check */
    for (uint32_t j = 0; j < n_cols; j++) {
        const int z = ds4_watersic_unpack_code(codes_row, j, R);
        acc += (double)z * (double)x[j];
    }
    return (float)(alpha[row] * (float)acc);
}

void ds4_watersic_pack_print_summary(const ds4_watersic_pack *p) {
    if (!p || !p->hdr) {
        fprintf(stderr, "ds4_watersic: pack not open\n");
        return;
    }
    const ds4_watersic_header *h = p->hdr;
    fprintf(stderr,
        "ds4_watersic: pack=%s\n"
        "  magic=0x%08x version=%u R=%u bits\n"
        "  dims: n_layers=%u n_experts=%u n_kinds=%u d_model=%u d_ffn=%u\n"
        "  arenas: lk_table @%llu, alpha @%llu, codes @%llu\n"
        "  total_bytes=%llu (%.2f GB)\n"
        "  lk_entries: %u of %u expected\n",
        p->pack_path,
        h->magic, h->version, h->R,
        h->n_layers, h->n_experts, h->n_kinds, h->d_model, h->d_ffn,
        (unsigned long long)h->lk_table_offset,
        (unsigned long long)h->alpha_arena_offset,
        (unsigned long long)h->codes_arena_offset,
        (unsigned long long)h->total_pack_bytes,
        (double)h->total_pack_bytes / (1024.0 * 1024.0 * 1024.0),
        p->n_lk, h->n_layers * h->n_kinds);
}
