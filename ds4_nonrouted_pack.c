/* ds4_nonrouted_pack.c — DS4 V4 non-routed weights pack reader.
 *
 * Minimal JSON manifest parser (no external deps): expects format produced
 * by pack_nonrouted.py exactly. Handles dtype strings + shape int arrays.
 */
#include "ds4_nonrouted_pack.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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
    if (!strcmp(s, "F32"))      return DS4_NRPK_DTYPE_F32;
    if (!strcmp(s, "F16"))      return DS4_NRPK_DTYPE_F16;
    if (!strcmp(s, "BF16"))     return DS4_NRPK_DTYPE_BF16;
    if (!strcmp(s, "I8"))       return DS4_NRPK_DTYPE_I8;
    if (!strcmp(s, "F8_E4M3"))  return DS4_NRPK_DTYPE_F8_E4M3;
    if (!strcmp(s, "F8_E8M0"))  return DS4_NRPK_DTYPE_F8_E8M0;
    return DS4_NRPK_DTYPE_UNKNOWN;
}

const char *ds4_nrpk_dtype_name(ds4_nrpk_dtype d) {
    switch (d) {
        case DS4_NRPK_DTYPE_F32:     return "F32";
        case DS4_NRPK_DTYPE_F16:     return "F16";
        case DS4_NRPK_DTYPE_BF16:    return "BF16";
        case DS4_NRPK_DTYPE_I8:      return "I8";
        case DS4_NRPK_DTYPE_F8_E4M3: return "F8_E4M3";
        case DS4_NRPK_DTYPE_F8_E8M0: return "F8_E8M0";
        default:                     return "UNK";
    }
}

size_t ds4_nrpk_dtype_bytes(ds4_nrpk_dtype d) {
    switch (d) {
        case DS4_NRPK_DTYPE_F32: return 4;
        case DS4_NRPK_DTYPE_F16:
        case DS4_NRPK_DTYPE_BF16: return 2;
        case DS4_NRPK_DTYPE_I8:
        case DS4_NRPK_DTYPE_F8_E4M3:
        case DS4_NRPK_DTYPE_F8_E8M0: return 1;
        default: return 0;
    }
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

/* Parse JSON integer. */
static const char *parse_int(const char *p, const char *end, int64_t *out) {
    p = skip_ws(p, end);
    int neg = 0;
    if (p < end && *p == '-') { neg = 1; p++; }
    int64_t v = 0;
    int any = 0;
    while (p < end && *p >= '0' && *p <= '9') {
        v = v * 10 + (*p - '0');
        p++;
        any = 1;
    }
    if (!any) return NULL;
    *out = neg ? -v : v;
    return p;
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
        char key[64];
        p = parse_str(p, end, key, sizeof(key));
        if (!p) return NULL;
        p = expect_char(p, end, ':');
        if (!p) return NULL;
        p = skip_ws(p, end);
        if (!p || p >= end) return NULL;
        if (!strcmp(key, "name")) {
            p = parse_str(p, end, out->name, sizeof(out->name));
            if (!p) return NULL;
        } else if (!strcmp(key, "dtype")) {
            char dt[16];
            p = parse_str(p, end, dt, sizeof(dt));
            if (!p) return NULL;
            out->dtype = ds4_nrpk_dtype_from_string(dt);
        } else if (!strcmp(key, "shape")) {
            p = expect_char(p, end, '[');
            if (!p) return NULL;
            uint32_t nd = 0;
            while (p < end && nd < DS4_NRPK_MAX_DIMS) {
                p = skip_ws(p, end);
                if (p < end && *p == ']') { p++; break; }
                int64_t d;
                p = parse_int(p, end, &d);
                if (!p) return NULL;
                out->dims[nd++] = (uint32_t)d;
                p = skip_ws(p, end);
                if (p < end && *p == ',') p++;
            }
            out->n_dims = nd;
        } else if (!strcmp(key, "data_off")) {
            int64_t v;
            p = parse_int(p, end, &v);
            if (!p) return NULL;
            out->data_off = (uint64_t)v;
        } else if (!strcmp(key, "data_bytes")) {
            int64_t v;
            p = parse_int(p, end, &v);
            if (!p) return NULL;
            out->data_bytes = (uint64_t)v;
        } else {
            /* Skip unknown value (string, int, or array). */
            p = skip_ws(p, end);
            if (p >= end) return NULL;
            if (*p == '"') {
                char tmp[256];
                p = parse_str(p, end, tmp, sizeof(tmp));
                if (!p) return NULL;
            } else if (*p == '[') {
                /* Skip to matching ] */
                int depth = 0;
                while (p < end) {
                    if (*p == '[') depth++;
                    else if (*p == ']') { depth--; if (depth == 0) { p++; break; } }
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
    const ds4_nrpk_entry *ea = (const ds4_nrpk_entry *)a;
    const ds4_nrpk_entry *eb = (const ds4_nrpk_entry *)b;
    return strcmp(ea->name, eb->name);
}

bool ds4_nrpk_open(const char *pack_path, ds4_nrpk *out) {
    if (!pack_path || !out) return false;
    memset(out, 0, sizeof(*out));
    out->fd = -1;
    strncpy(out->pack_path, pack_path, sizeof(out->pack_path) - 1);

    int fd = open(pack_path, O_RDONLY);
    if (fd < 0) {
        ds4_nrpk_log("open(%s) failed: %s", pack_path, strerror(errno));
        return false;
    }
    struct stat st;
    if (fstat(fd, &st) < 0) {
        ds4_nrpk_log("fstat failed: %s", strerror(errno));
        close(fd);
        return false;
    }
    if ((size_t)st.st_size < sizeof(ds4_nrpk_header)) {
        ds4_nrpk_log("file too small");
        close(fd);
        return false;
    }
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        ds4_nrpk_log("mmap failed: %s", strerror(errno));
        close(fd);
        return false;
    }
    const ds4_nrpk_header *hdr = (const ds4_nrpk_header *)map;
    if (memcmp(hdr->magic, DS4_NRPK_MAGIC, 8) != 0) {
        ds4_nrpk_log("bad magic");
        munmap(map, (size_t)st.st_size);
        close(fd);
        return false;
    }
    if (hdr->version != DS4_NRPK_VERSION) {
        ds4_nrpk_log("unsupported version %u", hdr->version);
        munmap(map, (size_t)st.st_size);
        close(fd);
        return false;
    }
    if (hdr->total_bytes != (uint64_t)st.st_size) {
        ds4_nrpk_log("size mismatch: hdr says %llu, file is %lld",
                     (unsigned long long)hdr->total_bytes,
                     (long long)st.st_size);
        munmap(map, (size_t)st.st_size);
        close(fd);
        return false;
    }

    out->fd       = fd;
    out->map      = map;
    out->map_size = (size_t)st.st_size;
    out->hdr      = hdr;
    out->data_arena = (const uint8_t *)map + hdr->data_offset;

    /* Parse manifest. */
    const char *m_start = (const char *)map + hdr->manifest_offset;
    const char *m_end   = m_start + hdr->manifest_bytes;

    out->entries = (ds4_nrpk_entry *)calloc(hdr->n_tensors, sizeof(ds4_nrpk_entry));
    if (!out->entries) {
        munmap(map, out->map_size);
        close(fd);
        memset(out, 0, sizeof(*out));
        out->fd = -1;
        return false;
    }

    const char *p = m_start;
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
        p = parse_entry(p, m_end, &out->entries[count]);
        if (!p) {
            ds4_nrpk_log("failed to parse entry %u", count);
            ds4_nrpk_close(out);
            return false;
        }
        if (out->entries[count].data_off + out->entries[count].data_bytes > hdr->data_bytes) {
            ds4_nrpk_log("entry %s out of data bounds", out->entries[count].name);
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
    if (p->map && p->map_size > 0) munmap(p->map, p->map_size);
    if (p->fd >= 0) close(p->fd);
    free(p->entries);
    memset(p, 0, sizeof(*p));
    p->fd = -1;
}

const ds4_nrpk_entry *ds4_nrpk_lookup(const ds4_nrpk *p, const char *name) {
    if (!p || !name || !p->entries) return NULL;
    /* Binary search */
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

uint32_t ds4_nrpk_count_prefix(const ds4_nrpk *p, const char *prefix) {
    if (!p || !prefix) return 0;
    size_t plen = strlen(prefix);
    uint32_t count = 0;
    for (uint32_t i = 0; i < p->n_entries; i++) {
        if (strncmp(p->entries[i].name, prefix, plen) == 0) count++;
    }
    return count;
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
    uint32_t hist[8] = {0};
    uint64_t hist_bytes[8] = {0};
    for (uint32_t i = 0; i < p->n_entries; i++) {
        const ds4_nrpk_entry *e = &p->entries[i];
        if (e->dtype < 8) {
            hist[e->dtype]++;
            hist_bytes[e->dtype] += e->data_bytes;
        }
    }
    for (uint32_t d = 1; d < 8; d++) {
        if (hist[d] > 0) {
            fprintf(stderr, "  %-8s: %5u tensors, %.2f GB\n",
                ds4_nrpk_dtype_name((ds4_nrpk_dtype)d),
                hist[d], (double)hist_bytes[d] / 1e9);
        }
    }
    /* Selected sample lookups */
    const char *samples[] = {
        "embed.weight", "head.weight", "norm.weight",
        "layers.0.attn.wq_b.weight", "layers.0.attn_norm.weight",
        "layers.22.ffn.gate.weight",
        "layers.22.ffn.shared_experts.w1.weight",
        "mtp.0.attn.wq_b.weight",
    };
    fprintf(stderr, "  sample lookups:\n");
    for (size_t i = 0; i < sizeof(samples) / sizeof(samples[0]); i++) {
        const ds4_nrpk_entry *e = ds4_nrpk_lookup(p, samples[i]);
        if (e) {
            fprintf(stderr, "    %-50s dtype=%s shape=[", samples[i],
                    ds4_nrpk_dtype_name(e->dtype));
            for (uint32_t d = 0; d < e->n_dims; d++) {
                fprintf(stderr, "%s%u", d > 0 ? "," : "", e->dims[d]);
            }
            fprintf(stderr, "] bytes=%llu\n", (unsigned long long)e->data_bytes);
        } else {
            fprintf(stderr, "    %-50s NOT FOUND\n", samples[i]);
        }
    }
}
