/* ds4_polar_reader.c — PLR2 file reader. See ds4_polar_reader.h for the
 * on-disk layout + invariants.  Phase A of task #563 (silv 2026-05-25). */

#include "ds4_polar_reader.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define DS4_POLAR_MAGIC   "PLR2"
#define DS4_POLAR_VERSION 1u
#define DS4_POLAR_HEADER_BYTES 64u

/* Legacy defaults when PLR2 header reserved bytes are zero (encoder
 * pre-cross-layer-Test-A used to hardcode 8 phase × 4 mag). New encoder
 * (post-cross-layer Test A 2026-05-25) records explicit values in
 * bytes 28-35. */
#define DS4_POLAR_LEGACY_PHASE_LEVELS 8u
#define DS4_POLAR_LEGACY_MAG_LEVELS   4u

static uint32_t le32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

bool ds4_polar_open(const char *path, ds4_polar_file *out) {
    if (!path || !out) return false;
    memset(out, 0, sizeof(*out));

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "ds4_polar_open: open(%s) failed: %s\n",
                path, strerror(errno));
        return false;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "ds4_polar_open: fstat failed: %s\n", strerror(errno));
        close(fd);
        return false;
    }
    if ((uint64_t)st.st_size < DS4_POLAR_HEADER_BYTES) {
        fprintf(stderr, "ds4_polar_open: %s too small (%lld bytes)\n",
                path, (long long)st.st_size);
        close(fd);
        return false;
    }
    void *mm = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mm == MAP_FAILED) {
        fprintf(stderr, "ds4_polar_open: mmap failed: %s\n", strerror(errno));
        close(fd);
        return false;
    }
    const uint8_t *base = (const uint8_t *)mm;
    if (memcmp(base, DS4_POLAR_MAGIC, 4) != 0) {
        fprintf(stderr,
                "ds4_polar_open: bad magic in %s (got %02x %02x %02x %02x, want PLR2)\n",
                path, base[0], base[1], base[2], base[3]);
        munmap(mm, (size_t)st.st_size);
        close(fd);
        return false;
    }
    uint32_t version  = le32(base + 4);
    uint32_t n_experts = le32(base + 8);
    uint32_t n_rows    = le32(base + 12);
    uint32_t n_pairs   = le32(base + 16);
    uint32_t layer     = le32(base + 20);
    uint32_t kind_id   = le32(base + 24);
    /* Bytes 28-31, 32-35: phase_levels and mag_levels (legacy=0 → defaults). */
    uint32_t phase_levels = le32(base + 28);
    uint32_t mag_levels   = le32(base + 32);
    if (phase_levels == 0) phase_levels = DS4_POLAR_LEGACY_PHASE_LEVELS;
    if (mag_levels   == 0) mag_levels   = DS4_POLAR_LEGACY_MAG_LEVELS;

    if (version != DS4_POLAR_VERSION) {
        fprintf(stderr, "ds4_polar_open: %s version=%u, want %u\n",
                path, version, DS4_POLAR_VERSION);
        munmap(mm, (size_t)st.st_size);
        close(fd);
        return false;
    }
    if (kind_id > DS4_POLAR_KIND_DOWN) {
        fprintf(stderr, "ds4_polar_open: %s kind_id=%u out of range\n",
                path, kind_id);
        munmap(mm, (size_t)st.st_size);
        close(fd);
        return false;
    }
    if (phase_levels < 4 || phase_levels > 65536 || (phase_levels & (phase_levels - 1)) != 0) {
        fprintf(stderr, "ds4_polar_open: %s phase_levels=%u not a power of 2 in [4, 65536]\n",
                path, phase_levels);
        munmap(mm, (size_t)st.st_size);
        close(fd);
        return false;
    }
    if (mag_levels < 1 || mag_levels > 256) {
        fprintf(stderr, "ds4_polar_open: %s mag_levels=%u out of range\n",
                path, mag_levels);
        munmap(mm, (size_t)st.st_size);
        close(fd);
        return false;
    }
    const size_t per_expert_codes = (size_t)n_rows * (size_t)n_pairs;
    const size_t per_expert_levels = (size_t)n_rows * (size_t)mag_levels;
    const size_t mag_bytes = (size_t)n_experts * per_expert_codes;
    const size_t phase_bytes = mag_bytes;
    const size_t levels_bytes = (size_t)n_experts * per_expert_levels * sizeof(float);
    const size_t want_total = DS4_POLAR_HEADER_BYTES + mag_bytes + phase_bytes + levels_bytes;
    if ((uint64_t)st.st_size < want_total) {
        fprintf(stderr,
                "ds4_polar_open: %s truncated (size=%lld, want %zu)\n",
                path, (long long)st.st_size, want_total);
        munmap(mm, (size_t)st.st_size);
        close(fd);
        return false;
    }

    out->mmap_base = mm;
    out->mmap_bytes = (size_t)st.st_size;
    out->fd = fd;
    out->version = version;
    out->n_experts = n_experts;
    out->n_rows = n_rows;
    out->n_pairs = n_pairs;
    out->layer = layer;
    out->kind_id = kind_id;
    out->phase_levels = phase_levels;
    out->mag_levels = mag_levels;
    out->mag_base = base + DS4_POLAR_HEADER_BYTES;
    out->phase_base = out->mag_base + mag_bytes;
    out->levels_base = (const float *)(out->phase_base + phase_bytes);
    out->expert_mag_stride = per_expert_codes;
    out->expert_phase_stride = per_expert_codes;
    out->expert_levels_stride = per_expert_levels;
    return true;
}

void ds4_polar_close(ds4_polar_file *p) {
    if (!p) return;
    if (p->mmap_base && p->mmap_bytes) {
        munmap(p->mmap_base, p->mmap_bytes);
    }
    if (p->fd >= 0) close(p->fd);
    memset(p, 0, sizeof(*p));
    p->fd = -1;
}

bool ds4_polar_expert_view(const ds4_polar_file *p, uint32_t expert_idx,
                            const uint8_t **mag_out,
                            const uint8_t **phase_out,
                            const float   **levels_out) {
    if (!p || expert_idx >= p->n_experts) return false;
    if (mag_out)
        *mag_out = p->mag_base + (size_t)expert_idx * p->expert_mag_stride;
    if (phase_out)
        *phase_out = p->phase_base + (size_t)expert_idx * p->expert_phase_stride;
    if (levels_out)
        *levels_out = p->levels_base + (size_t)expert_idx * p->expert_levels_stride;
    return true;
}

/* Decode (mag_code, phase_code, levels_row) → complex pair in fp32.
 * Mirror of polar_encode_mlx.py (parameterized for arbitrary phase_levels
 * P and mag_levels M):
 *   qmag = levels[row, mag_code]   (mag_code ∈ [0, M))
 *   qangle = (phase_code - P/2) * 2π/P   (phase_code ∈ [0, P])
 *   re = qmag * cos(qangle)
 *   im = qmag * sin(qangle)
 *
 * Pre-cross-layer-Test-A PLR2 files have phase_levels=0/mag_levels=0
 * in reserved header bytes; reader populates these as 8/4 at open time
 * to maintain legacy behavior.  Decoders operating on the handle always
 * read p->phase_levels / p->mag_levels.
 */
static void decode_pair_internal(uint8_t mag_code, uint8_t phase_code,
                                  const float *levels_row,
                                  uint32_t phase_levels, uint32_t mag_levels,
                                  float *re_out, float *im_out) {
    if ((uint32_t)mag_code >= mag_levels) mag_code = (uint8_t)(mag_levels - 1);
    const float qmag = levels_row[mag_code];
    const double phase_step = (2.0 * M_PI) / (double)phase_levels;
    const double qangle = ((int)phase_code - (int)(phase_levels / 2u)) * phase_step;
    if (re_out) *re_out = (float)(qmag * cos(qangle));
    if (im_out) *im_out = (float)(qmag * sin(qangle));
}

float ds4_polar_decode_pair_re(const ds4_polar_file *p, uint32_t expert_idx,
                                uint32_t row, uint32_t pair) {
    if (!p || expert_idx >= p->n_experts ||
        row >= p->n_rows || pair >= p->n_pairs) return 0.0f;
    const uint8_t *mag = p->mag_base + (size_t)expert_idx * p->expert_mag_stride
                       + (size_t)row * p->n_pairs;
    const uint8_t *phase = p->phase_base + (size_t)expert_idx * p->expert_phase_stride
                         + (size_t)row * p->n_pairs;
    const float *levels = p->levels_base
                        + (size_t)expert_idx * p->expert_levels_stride
                        + (size_t)row * p->mag_levels;
    float re;
    decode_pair_internal(mag[pair], phase[pair], levels, p->phase_levels, p->mag_levels, &re, NULL);
    return re;
}

float ds4_polar_decode_pair_im(const ds4_polar_file *p, uint32_t expert_idx,
                                uint32_t row, uint32_t pair) {
    if (!p || expert_idx >= p->n_experts ||
        row >= p->n_rows || pair >= p->n_pairs) return 0.0f;
    const uint8_t *mag = p->mag_base + (size_t)expert_idx * p->expert_mag_stride
                       + (size_t)row * p->n_pairs;
    const uint8_t *phase = p->phase_base + (size_t)expert_idx * p->expert_phase_stride
                         + (size_t)row * p->n_pairs;
    const float *levels = p->levels_base
                        + (size_t)expert_idx * p->expert_levels_stride
                        + (size_t)row * p->mag_levels;
    float im;
    decode_pair_internal(mag[pair], phase[pair], levels, p->phase_levels, p->mag_levels, NULL, &im);
    return im;
}

static const char *kind_name(uint32_t kind_id) {
    switch (kind_id) {
        case DS4_POLAR_KIND_GATE: return "gate";
        case DS4_POLAR_KIND_UP:   return "up";
        case DS4_POLAR_KIND_DOWN: return "down";
        default: return "?";
    }
}

void ds4_polar_print_summary(const ds4_polar_file *p, const char *label) {
    if (!p || !p->mmap_base) {
        fprintf(stderr, "ds4_polar: %s — handle not open\n",
                label ? label : "(unnamed)");
        return;
    }
    const size_t total_codes = (size_t)p->n_experts * (size_t)p->n_rows * (size_t)p->n_pairs;
    const size_t total_levels = (size_t)p->n_experts * (size_t)p->n_rows * (size_t)p->mag_levels;

    /* Quick histogram on the first expert's first row. */
    /* phase_levels can be up to 65536; cap histogram displayable size at 17 for legible printing.
     * Bins up to MIN(phase_levels+1, 17) are tallied + shown. */
    enum { MAX_PHASE_BUCKETS = 17 };
    const uint32_t hist_buckets = (p->phase_levels + 1u < MAX_PHASE_BUCKETS)
                                    ? (p->phase_levels + 1u) : MAX_PHASE_BUCKETS;
    /* mag_hist sized to mag_levels (≤256). */
    uint64_t mag_hist[256] = {0};
    uint64_t phase_hist[MAX_PHASE_BUCKETS] = {0};
    const uint8_t *m0, *p0;
    const float   *l0;
    if (ds4_polar_expert_view(p, 0, &m0, &p0, &l0)) {
        for (uint32_t k = 0; k < p->n_pairs; k++) {
            if (m0[k] < p->mag_levels) mag_hist[m0[k]]++;
            if (p0[k] < hist_buckets) phase_hist[p0[k]]++;
        }
    }

    fprintf(stderr, "ds4_polar: %s\n", label ? label : "summary");
    fprintf(stderr, "  layer=%u kind=%s (id=%u) version=%u\n",
            p->layer, kind_name(p->kind_id), p->kind_id, p->version);
    fprintf(stderr, "  n_experts=%u n_rows=%u n_pairs=%u  codec=p%u_m%u\n",
            p->n_experts, p->n_rows, p->n_pairs, p->phase_levels, p->mag_levels);
    fprintf(stderr,
            "  bytes mmap=%zu  codes=%zu (mag+phase=%zu)  levels=%zu (%.1f MB total)\n",
            p->mmap_bytes, total_codes, total_codes * 2u, total_levels,
            (double)p->mmap_bytes / (1024.0 * 1024.0));
    fprintf(stderr, "  E0,R0 mag hist (first %u of %u): [",
            p->mag_levels < 8 ? p->mag_levels : 8, p->mag_levels);
    uint32_t mag_show = p->mag_levels < 8 ? p->mag_levels : 8;
    for (uint32_t k = 0; k < mag_show; k++) {
        fprintf(stderr, "%llu%s", (unsigned long long)mag_hist[k],
                k == mag_show - 1 ? "" : ",");
    }
    fprintf(stderr, "]   ");
    fprintf(stderr, "phase hist (first %u of %u): [", hist_buckets, p->phase_levels + 1);
    for (uint32_t k = 0; k < hist_buckets; k++) {
        fprintf(stderr, "%llu%s",
                (unsigned long long)phase_hist[k],
                k == hist_buckets - 1 ? "" : ",");
    }
    fprintf(stderr, "]\n");
    if (l0) {
        fprintf(stderr, "  E0,R0 levels (first %u of %u): [",
                p->mag_levels < 8 ? p->mag_levels : 8, p->mag_levels);
        uint32_t lvl_show = p->mag_levels < 8 ? p->mag_levels : 8;
        for (uint32_t k = 0; k < lvl_show; k++) {
            fprintf(stderr, "%.6f%s", l0[k], k == lvl_show - 1 ? "" : ", ");
        }
        fprintf(stderr, "]\n");
        float re0 = ds4_polar_decode_pair_re(p, 0, 0, 0);
        float im0 = ds4_polar_decode_pair_im(p, 0, 0, 0);
        fprintf(stderr,
                "  E0,R0,P0 decoded: re=%.6f im=%.6f (mag_code=%u phase_code=%u)\n",
                re0, im0, m0[0], p0[0]);
    }
}


/* ============================ ds4_polar_pool ============================ */

#include <dirent.h>

void ds4_polar_pool_init(ds4_polar_pool *pool) {
    if (!pool) return;
    memset(pool, 0, sizeof(*pool));
    for (uint32_t l = 0; l < DS4_POLAR_MAX_LAYERS; l++)
        for (uint32_t k = 0; k < 3; k++)
            pool->files[l][k].fd = -1;
}

void ds4_polar_pool_close(ds4_polar_pool *pool) {
    if (!pool) return;
    for (uint32_t l = 0; l < DS4_POLAR_MAX_LAYERS; l++) {
        for (uint32_t k = 0; k < 3; k++) {
            if (pool->files[l][k].mmap_base) {
                ds4_polar_close(&pool->files[l][k]);
            }
        }
    }
    memset(pool, 0, sizeof(*pool));
    for (uint32_t l = 0; l < DS4_POLAR_MAX_LAYERS; l++)
        for (uint32_t k = 0; k < 3; k++)
            pool->files[l][k].fd = -1;
}

/* Filename matcher: returns true and sets *layer + *kind_id if the
 * filename has the form "L{NN}_{kind}.polar" with kind in {gate,up,down}. */
static bool parse_polar_filename(const char *name, uint32_t *layer, uint32_t *kind) {
    /* expect "L"+digits+"_"+("gate"|"up"|"down")+".polar" */
    if (name[0] != 'L') return false;
    const char *p = name + 1;
    uint32_t l = 0;
    int digits = 0;
    while (*p >= '0' && *p <= '9') {
        l = l * 10 + (uint32_t)(*p - '0');
        p++;
        digits++;
    }
    if (digits == 0 || *p != '_') return false;
    p++;
    uint32_t k;
    if (strncmp(p, "gate.polar", 11) == 0 && strlen(p) == 10) k = DS4_POLAR_KIND_GATE;
    else if (strncmp(p, "up.polar", 9) == 0 && strlen(p) == 8) k = DS4_POLAR_KIND_UP;
    else if (strncmp(p, "down.polar", 11) == 0 && strlen(p) == 10) k = DS4_POLAR_KIND_DOWN;
    else return false;
    *layer = l;
    *kind = k;
    return true;
}

uint32_t ds4_polar_pool_load_dir(ds4_polar_pool *pool, const char *dir) {
    if (!pool || !dir) return 0;
    DIR *d = opendir(dir);
    if (!d) {
        fprintf(stderr, "ds4_polar_pool_load_dir: opendir(%s) failed: %s\n",
                dir, strerror(errno));
        return 0;
    }
    uint32_t opened = 0;
    uint64_t bytes_total = 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        uint32_t layer, kind;
        if (!parse_polar_filename(ent->d_name, &layer, &kind)) continue;
        if (layer >= DS4_POLAR_MAX_LAYERS) {
            fprintf(stderr, "ds4_polar_pool: skipping %s — layer %u out of range\n",
                    ent->d_name, layer);
            continue;
        }
        /* Replace any prior entry for (layer, kind) */
        if (pool->files[layer][kind].mmap_base) {
            ds4_polar_close(&pool->files[layer][kind]);
            if (pool->opened_count > 0) pool->opened_count--;
        }
        /* Build full path */
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir, ent->d_name);
        if (!ds4_polar_open(path, &pool->files[layer][kind])) {
            /* ds4_polar_open already emitted a diagnostic */
            continue;
        }
        /* Verify the file's self-described (layer, kind) matches filename */
        if (pool->files[layer][kind].layer != layer ||
            pool->files[layer][kind].kind_id != kind) {
            fprintf(stderr,
                    "ds4_polar_pool: %s header (layer=%u, kind=%u) disagrees with "
                    "filename (layer=%u, kind=%u); accepting filename binding\n",
                    ent->d_name,
                    pool->files[layer][kind].layer, pool->files[layer][kind].kind_id,
                    layer, kind);
        }
        opened++;
        bytes_total += pool->files[layer][kind].mmap_bytes;
    }
    closedir(d);
    pool->opened_count += opened;
    pool->total_bytes += bytes_total;
    return opened;
}

const ds4_polar_file *ds4_polar_pool_get(const ds4_polar_pool *pool,
                                          uint32_t layer, uint32_t kind) {
    if (!pool || layer >= DS4_POLAR_MAX_LAYERS || kind > DS4_POLAR_KIND_DOWN) return NULL;
    const ds4_polar_file *p = &pool->files[layer][kind];
    return p->mmap_base ? p : NULL;
}

void ds4_polar_pool_print_summary(const ds4_polar_pool *pool, const char *label) {
    if (!pool) {
        fprintf(stderr, "ds4_polar_pool: %s — NULL\n", label ? label : "(unnamed)");
        return;
    }
    fprintf(stderr, "ds4_polar_pool: %s — %u files open, %.1f MB total resident\n",
            label ? label : "summary",
            pool->opened_count,
            (double)pool->total_bytes / (1024.0 * 1024.0));
    /* Build a per-layer presence bitmask: 1 byte per layer, bit 0=gate, 1=up, 2=down */
    uint32_t layers_any = 0;
    for (uint32_t l = 0; l < DS4_POLAR_MAX_LAYERS; l++) {
        uint8_t mask = 0;
        for (uint32_t k = 0; k < 3; k++) {
            if (pool->files[l][k].mmap_base) mask |= (uint8_t)(1u << k);
        }
        if (mask) {
            if (!layers_any) fprintf(stderr, "  layer-kind matrix (G=gate, U=up, D=down):\n");
            char tag[4] = "...";
            tag[0] = (mask & 1) ? 'G' : '.';
            tag[1] = (mask & 2) ? 'U' : '.';
            tag[2] = (mask & 4) ? 'D' : '.';
            fprintf(stderr, "    L%02u: %s  (n_experts=%u, n_rows=%u, n_pairs=%u)\n",
                    l, tag,
                    pool->files[l][(mask&1)?0:(mask&2)?1:2].n_experts,
                    pool->files[l][(mask&1)?0:(mask&2)?1:2].n_rows,
                    pool->files[l][(mask&1)?0:(mask&2)?1:2].n_pairs);
            layers_any++;
        }
    }
    if (!layers_any) fprintf(stderr, "  (no layers populated)\n");
}
