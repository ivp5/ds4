/* ds4_nonrouted_pack.c — DS4 V4 non-routed weights pack reader.
 * Minimal dependency-free JSON manifest parser; format per pack_nonrouted.py.
 * static-top pass 2026-06-04: file-scope dtype table + parse scratch,
 * count_prefix O(log n), overflow-guarded ints. */
#include "ds4_nonrouted_pack.h"
#include "ds4_pack_io.h"   /* shared mmap open/validate/close — one correct copy, not N */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ===================================================================== *
 * File-scope statics (silv doctrine: used/expected state defined + sized
 * at the top; reusable scratch allocated once, not per-call). All touched
 * only on the single-threaded pack-open path — no concurrent mutation.
 * ===================================================================== */

/* Single source of truth for the dtype enum<->name<->element-size mapping.
 * Indexed by ds4_nrpk_dtype value (contiguous 0..7). dtype_from_string,
 * dtype_name and dtype_bytes all derive from this one table, so they cannot
 * drift apart. */
static const struct { const char *name; uint8_t bytes; } NRPK_DTYPE[] = {
    [DS4_NRPK_DTYPE_UNKNOWN]  = { "UNK",     0 },
    [DS4_NRPK_DTYPE_F32]      = { "F32",     4 },
    [DS4_NRPK_DTYPE_F16]      = { "F16",     2 },
    [DS4_NRPK_DTYPE_BF16]     = { "BF16",    2 },
    [DS4_NRPK_DTYPE_I8]       = { "I8",      1 },
    [DS4_NRPK_DTYPE_F8_E4M3]  = { "F8_E4M3", 1 },
    [DS4_NRPK_DTYPE_F8_E8M0]  = { "F8_E8M0", 1 },
    [DS4_NRPK_DTYPE_I32]      = { "I32",     4 },
};
#define NRPK_DTYPE_N (sizeof(NRPK_DTYPE) / sizeof(NRPK_DTYPE[0]))

/* Reusable manifest-parse scratch, reused per entry (open is single-threaded). */
static char g_nrpk_key[64];     /* current object key                   */
static char g_nrpk_dt[16];      /* dtype string being decoded           */
static char g_nrpk_skip[256];   /* sink for skipped unknown string vals */

static void ds4_nrpk_log(const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "ds4_nrpk: ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

ds4_nrpk_dtype ds4_nrpk_dtype_from_string(const char *s) {
    if (!s) return DS4_NRPK_DTYPE_UNKNOWN;
    for (uint32_t d = 1; d < NRPK_DTYPE_N; d++) {
        if (!strcmp(s, NRPK_DTYPE[d].name)) return (ds4_nrpk_dtype)d;
    }
    return DS4_NRPK_DTYPE_UNKNOWN;
}

const char *ds4_nrpk_dtype_name(ds4_nrpk_dtype d) {
    return (uint32_t)d < NRPK_DTYPE_N ? NRPK_DTYPE[d].name : "UNK";
}

size_t ds4_nrpk_dtype_bytes(ds4_nrpk_dtype d) {
    return (uint32_t)d < NRPK_DTYPE_N ? NRPK_DTYPE[d].bytes : 0;
}

/* Skip whitespace. */
static const char *skip_ws(const char *p, const char *end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

/* Match exact char, advance. Returns NULL on failure. */
static const char *expect_char(const char *p, const char *end, char c) {
    p = skip_ws(p, end);
    if (p >= end || *p != c) return NULL;
    return p + 1;
}

/* Parse JSON string into out_buf (no escape handling beyond \"). */
static const char *parse_str(const char *p, const char *end,
                             char *out_buf, size_t out_cap) {
    p = skip_ws(p, end);
    if (p >= end || *p != '"') return NULL;
    p++;
    size_t i = 0;
    while (p < end && *p != '"') {
        if (*p == '\\' && p + 1 < end) p++; /* skip escape char itself */
        if (i + 1 < out_cap) out_buf[i++] = *p;
        p++;
    }
    out_buf[i] = '\0';
    if (p >= end || *p != '"') return NULL;
    return p + 1;
}

/* Parse JSON integer (saturating on overflow rather than wrapping). */
static const char *parse_int(const char *p, const char *end, int64_t *out) {
    p = skip_ws(p, end);
    int neg = 0;
    if (p < end && *p == '-') { neg = 1; p++; }
    uint64_t v = 0;
    int any = 0;
    while (p < end && *p >= '0' && *p <= '9') {
        if (v <= (UINT64_MAX - 9) / 10) v = v * 10 + (uint64_t)(*p - '0');
        /* else: saturate — manifest is malformed; bounds check downstream rejects */
        p++;
        any = 1;
    }
    if (!any) return NULL;
    if (v > (uint64_t)INT64_MAX) v = (uint64_t)INT64_MAX;
    *out = neg ? -(int64_t)v : (int64_t)v;
    return p;
}

static bool nrpk_range_within(uint64_t off, uint64_t len, uint64_t limit) {
    return off <= limit && len <= limit - off;
}

/* Parse one manifest entry: {"name": "...", "dtype": "...", "shape": [...],
 *                            "data_off": N, "data_bytes": N, ... } */
static const char *parse_entry(const char *p, const char *end,
                               ds4_nrpk_entry *out) {
    memset(out, 0, sizeof(*out));
    p = expect_char(p, end, '{');
    if (!p) return NULL;
    while (p < end) {
        p = skip_ws(p, end);
        if (p < end && *p == '}') return p + 1;
        p = parse_str(p, end, g_nrpk_key, sizeof(g_nrpk_key));
        if (!p) return NULL;
        p = expect_char(p, end, ':');
        if (!p) return NULL;
        p = skip_ws(p, end);
        if (p >= end) return NULL;
        if (!strcmp(g_nrpk_key, "name")) {
            p = parse_str(p, end, out->name, sizeof(out->name));
            if (!p) return NULL;
        } else if (!strcmp(g_nrpk_key, "dtype")) {
            p = parse_str(p, end, g_nrpk_dt, sizeof(g_nrpk_dt));
            if (!p) return NULL;
            out->dtype = ds4_nrpk_dtype_from_string(g_nrpk_dt);
        } else if (!strcmp(g_nrpk_key, "shape")) {
            p = expect_char(p, end, '[');
            if (!p) return NULL;
            uint32_t nd = 0;
            while (p < end && nd < DS4_NRPK_MAX_DIMS) {
                p = skip_ws(p, end);
                if (p < end && *p == ']') { p++; break; }
                int64_t d;
                p = parse_int(p, end, &d);
                if (!p) return NULL;
                if (d < 0 || (uint64_t)d > UINT32_MAX) return NULL;
                out->dims[nd++] = (uint32_t)d;
                p = skip_ws(p, end);
                if (p < end && *p == ',') p++;
            }
            out->n_dims = nd;
        } else if (!strcmp(g_nrpk_key, "data_off")) {
            int64_t v;
            p = parse_int(p, end, &v);
            if (!p) return NULL;
            if (v < 0) return NULL;
            out->data_off = (uint64_t)v;
        } else if (!strcmp(g_nrpk_key, "data_bytes")) {
            int64_t v;
            p = parse_int(p, end, &v);
            if (!p) return NULL;
            if (v < 0) return NULL;
            out->data_bytes = (uint64_t)v;
        } else {
            /* Skip unknown value (string, int, or array). */
            p = skip_ws(p, end);
            if (p >= end) return NULL;
            if (*p == '"') {
                p = parse_str(p, end, g_nrpk_skip, sizeof(g_nrpk_skip));
                if (!p) return NULL;
            } else if (*p == '[') {
                int depth = 0;        /* skip to matching ] */
                while (p < end) {
                    if (*p == '[') depth++;
                    else if (*p == ']') { if (--depth == 0) { p++; break; } }
                    p++;
                }
            } else {
                int64_t tmp;
                p = parse_int(p, end, &tmp);
                if (!p) return NULL;
            }
        }
        p = skip_ws(p, end);
        if (p < end && *p == ',') p++;
    }
    return NULL;
}

static int entry_cmp(const void *a, const void *b) {
    return strcmp(((const ds4_nrpk_entry *)a)->name,
                  ((const ds4_nrpk_entry *)b)->name);
}

bool ds4_nrpk_open(const char *pack_path, ds4_nrpk *out) {
    if (!pack_path || !out) return false;
    memset(out, 0, sizeof(*out));
    out->fd = -1;
    strncpy(out->pack_path, pack_path, sizeof(out->pack_path) - 1);

    /* Shared mmap+fstat+min-size open (magic=NULL: the 8-byte magic is verified below). */
    if (!ds4_pack_mmap_open(pack_path, "ds4_nrpk", NULL, sizeof(ds4_nrpk_header),
                            &out->map, &out->map_size, &out->fd)) {
        return false;
    }
    const ds4_nrpk_header *hdr = (const ds4_nrpk_header *)out->map;
    const uint64_t file_bytes = (uint64_t)out->map_size;
    const char *fail =
        memcmp(hdr->magic, DS4_NRPK_MAGIC, 8) != 0                 ? "bad magic" :
        hdr->version != DS4_NRPK_VERSION                           ? "unsupported version" :
        hdr->total_bytes != file_bytes                             ? "size mismatch" :
        !nrpk_range_within(hdr->manifest_offset, hdr->manifest_bytes, file_bytes) ? "manifest out of bounds" :
        !nrpk_range_within(hdr->data_offset, hdr->data_bytes, file_bytes)         ? "data out of bounds" :
        NULL;
    if (fail) {
        ds4_nrpk_log("%s (version=%u total=%llu file=%zu)", fail, hdr->version,
                     (unsigned long long)hdr->total_bytes, out->map_size);
        ds4_pack_munmap_close(&out->map, &out->map_size, &out->fd);
        return false;
    }
    out->hdr        = hdr;
    out->data_arena = (const uint8_t *)out->map + hdr->data_offset;

    /* Entries: the ONE heap allocation, and it stays heap by design. n_tensors
     * is determined by the file at runtime (nonrouted is a subset of a 69187-
     * tensor model — no honest compile-time cap exists; a guessed cap could
     * reject a valid pack, a provably-safe cap wastes ~15 MB for ~1500 actual
     * entries). This is calloc'd ONCE at open and freed at close — the doctrine's
     * allowed case (load-time, runtime-sized), not the per-call realloc it bans. */
    out->entries = (ds4_nrpk_entry *)calloc(hdr->n_tensors, sizeof(ds4_nrpk_entry));
    if (!out->entries) {
        ds4_pack_munmap_close(&out->map, &out->map_size, &out->fd);
        memset(out, 0, sizeof(*out));
        out->fd = -1;
        return false;
    }

    /* Parse manifest. */
    const char *p     = (const char *)out->map + hdr->manifest_offset;
    const char *m_end = p + hdr->manifest_bytes;
    p = expect_char(p, m_end, '[');
    if (!p) {
        ds4_nrpk_log("manifest missing '['");
        ds4_nrpk_close(out);
        return false;
    }
    uint32_t count = 0;
    while (p && p < m_end) {
        p = skip_ws(p, m_end);
        if (p < m_end && *p == ']') break;
        if (count >= hdr->n_tensors) {
            ds4_nrpk_log("manifest has more entries than n_tensors=%u", hdr->n_tensors);
            ds4_nrpk_close(out);
            return false;
        }
        ds4_nrpk_entry *e = &out->entries[count];
        p = parse_entry(p, m_end, e);
        if (!p) {
            ds4_nrpk_log("failed to parse entry %u", count);
            ds4_nrpk_close(out);
            return false;
        }
        if (!nrpk_range_within(e->data_off, e->data_bytes, hdr->data_bytes)) {
            ds4_nrpk_log("entry %s out of data bounds", e->name);
            ds4_nrpk_close(out);
            return false;
        }
        count++;
        p = skip_ws(p, m_end);
        if (p < m_end && *p == ',') p++;
    }
    out->n_entries = count;
    if (count != hdr->n_tensors) {
        ds4_nrpk_log("manifest count %u != header n_tensors %u", count, hdr->n_tensors);
    }

    /* Sort by name for binary lookup. */
    qsort(out->entries, out->n_entries, sizeof(ds4_nrpk_entry), entry_cmp);
    return true;
}

void ds4_nrpk_close(ds4_nrpk *p) {
    if (!p) return;
    ds4_pack_munmap_close(&p->map, &p->map_size, &p->fd);
    free(p->entries);
    memset(p, 0, sizeof(*p));
    p->fd = -1;
}

const ds4_nrpk_entry *ds4_nrpk_lookup(const ds4_nrpk *p, const char *name) {
    if (!p || !name || !p->entries) return NULL;
    int32_t lo = 0, hi = (int32_t)p->n_entries - 1;
    while (lo <= hi) {
        int32_t mid = (lo + hi) >> 1;
        int c = strcmp(p->entries[mid].name, name);
        if (c == 0) return &p->entries[mid];
        if (c < 0) lo = mid + 1; else hi = mid - 1;
    }
    return NULL;
}

const void *ds4_nrpk_get_data(const ds4_nrpk *p, const ds4_nrpk_entry *e) {
    if (!p || !e) return NULL;
    return p->data_arena + e->data_off;
}

/* Count entries whose name starts with `prefix`. The array is sorted by name,
 * so all matches form a contiguous range [first, last): two binary searches
 * make this O(log n) instead of an O(n) scan. */
uint32_t ds4_nrpk_count_prefix(const ds4_nrpk *p, const char *prefix) {
    if (!p || !prefix || !p->entries || p->n_entries == 0) return 0;
    size_t plen = strlen(prefix);
    if (plen == 0) return p->n_entries;
    const uint32_t n = p->n_entries;
    /* first = lowest index with name >= prefix  (lower bound) */
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = (lo + hi) >> 1;
        if (strcmp(p->entries[mid].name, prefix) < 0) lo = mid + 1; else hi = mid;
    }
    uint32_t first = lo;
    /* last = lowest index in [first,n) whose name no longer starts with prefix.
     * Within [first,n), strncmp(name,prefix,plen) is 0 for matches (contiguous,
     * first) then >0 for the rest, so a binary search on (>0) finds the end. */
    hi = n;
    for (lo = first; lo < hi; ) {
        uint32_t mid = (lo + hi) >> 1;
        if (strncmp(p->entries[mid].name, prefix, plen) <= 0) lo = mid + 1; else hi = mid;
    }
    return lo - first;
}

void ds4_nrpk_print_summary(const ds4_nrpk *p) {
    if (!p || !p->hdr) {
        fprintf(stderr, "ds4_nrpk: pack not open\n");
        return;
    }
    fprintf(stderr,
        "ds4_nrpk: pack=%s\n"
        "  version=%u n_tensors=%u\n"
        "  manifest=%llu B  data=%.2f GB  total=%.2f GB\n",
        p->pack_path, p->hdr->version, p->hdr->n_tensors,
        (unsigned long long)p->hdr->manifest_bytes,
        (double)p->hdr->data_bytes / 1e9,
        (double)p->hdr->total_bytes / 1e9);
    /* dtype histogram */
    uint32_t hist[NRPK_DTYPE_N] = {0};
    uint64_t hist_bytes[NRPK_DTYPE_N] = {0};
    for (uint32_t i = 0; i < p->n_entries; i++) {
        const ds4_nrpk_entry *e = &p->entries[i];
        if ((uint32_t)e->dtype < NRPK_DTYPE_N) {
            hist[e->dtype]++;
            hist_bytes[e->dtype] += e->data_bytes;
        }
    }
    for (uint32_t d = 1; d < NRPK_DTYPE_N; d++) {
        if (hist[d] > 0) {
            fprintf(stderr, "  %-8s: %5u tensors, %.2f GB\n",
                NRPK_DTYPE[d].name, hist[d], (double)hist_bytes[d] / 1e9);
        }
    }
    /* Selected sample lookups */
    static const char *samples[] = {
        "embed.weight", "head.weight", "norm.weight",
        "layers.0.attn.wq_b.weight", "layers.0.attn_norm.weight",
        "layers.22.ffn.gate.weight",
        "layers.22.ffn.shared_experts.w1.weight",
        "mtp.0.attn.wq_b.weight",
    };
    fprintf(stderr, "  sample lookups:\n");
    for (size_t i = 0; i < sizeof(samples) / sizeof(samples[0]); i++) {
        const ds4_nrpk_entry *e = ds4_nrpk_lookup(p, samples[i]);
        if (!e) {
            fprintf(stderr, "    %-50s NOT FOUND\n", samples[i]);
            continue;
        }
        fprintf(stderr, "    %-50s dtype=%s shape=[", samples[i],
                ds4_nrpk_dtype_name(e->dtype));
        for (uint32_t d = 0; d < e->n_dims; d++) {
            fprintf(stderr, "%s%u", d > 0 ? "," : "", e->dims[d]);
        }
        fprintf(stderr, "] bytes=%llu\n", (unsigned long long)e->data_bytes);
    }
}
