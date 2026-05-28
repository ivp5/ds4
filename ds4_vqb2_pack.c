/* ds4_vqb2_pack.c — pack-backed VQB2 view loader.
 *
 * See ds4_vqb2_pack.h for design. Companion to codex H2116-H2125: thousands
 * of per-file VQB2 packets cost 25-39× per access vs single-mmap pack views.
 *
 * silv 2026-05-28 — first deployable pack reader for the
 * ds4_flash_nonrotated_layer22_k256_gateup_top4 artifact (36.5 GiB, 2752 packets).
 */

#include "ds4_vqb2_pack.h"
#include "ds4_expert_table.h"  /* ds4_hot_pin_expert_from_vqb2 */

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static const uint32_t LOOKUP_DIM_LK = DS4_VQB2_PACK_N_KIND * DS4_VQB2_PACK_MAX_ROW_BLOCKS;
static const uint32_t LOOKUP_DIM    = DS4_VQB2_PACK_MAX_LAYER * DS4_VQB2_PACK_N_KIND * DS4_VQB2_PACK_MAX_ROW_BLOCKS;

static inline int32_t lookup_slot(uint32_t layer, uint32_t kind_id, uint32_t row_start) {
    if (layer >= DS4_VQB2_PACK_MAX_LAYER) return -1;
    if (kind_id >= DS4_VQB2_PACK_N_KIND) return -1;
    if ((row_start & 127u) != 0) return -1;
    const uint32_t rb = row_start >> 7;
    if (rb >= DS4_VQB2_PACK_MAX_ROW_BLOCKS) return -1;
    return (int32_t)(layer * LOOKUP_DIM_LK + kind_id * DS4_VQB2_PACK_MAX_ROW_BLOCKS + rb);
}

/* CSV column index map (resolved from header row). -1 = column missing. */
typedef struct {
    int layer;
    int kind_id;
    int k;
    int row_start;
    int pack_offset;
    int packet_bytes;
    int header_n_experts;
    int header_n_rows;
    int header_n_pairs;
    int header_bit_width;
    int header_n_codes;
} csv_cols_t;

static int find_col(char **headers, int n_cols, const char *name) {
    for (int i = 0; i < n_cols; i++) {
        if (strcmp(headers[i], name) == 0) return i;
    }
    return -1;
}

/* Split a CSV line in-place (modifies buf). Returns number of fields written
 * to out_fields (capped at max_fields). Does NOT handle quoted commas — fine
 * for our index format (no commas inside fields). */
static int split_csv_line(char *buf, char **out_fields, int max_fields) {
    int n = 0;
    char *p = buf;
    out_fields[n++] = p;
    while (*p && n < max_fields) {
        if (*p == ',') {
            *p = '\0';
            out_fields[n++] = p + 1;
        } else if (*p == '\n' || *p == '\r') {
            *p = '\0';
        }
        p++;
    }
    /* Trim trailing newline on last field */
    while (*p) {
        if (*p == '\n' || *p == '\r') { *p = '\0'; break; }
        p++;
    }
    return n;
}

static bool parse_index_csv(const char *path, ds4_vqb2_pack *p) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "ds4_vqb2_pack: open index %s failed: %s\n", path, strerror(errno));
        return false;
    }

    /* Two-pass: count rows, allocate, then re-read.
     * Lines are typically ~600B; the file is ~1.5 MB. */
    size_t n_rows = 0;
    {
        int c, prev = 0;
        while ((c = fgetc(fp)) != EOF) {
            if (c == '\n') n_rows++;
            prev = c;
        }
        if (prev != '\n' && prev != 0) n_rows++;
        rewind(fp);
    }
    if (n_rows < 2) {
        fprintf(stderr, "ds4_vqb2_pack: index %s has %zu lines (need header + ≥1 row)\n",
                path, n_rows);
        fclose(fp);
        return false;
    }
    const uint32_t cap = (uint32_t)(n_rows - 1);

    /* Allocate entries + lookup */
    p->entries = (ds4_vqb2_pack_entry *)calloc(cap, sizeof(ds4_vqb2_pack_entry));
    p->lookup  = (int32_t *)malloc(LOOKUP_DIM * sizeof(int32_t));
    if (!p->entries || !p->lookup) {
        fprintf(stderr, "ds4_vqb2_pack: alloc failed (entries=%u)\n", cap);
        fclose(fp);
        return false;
    }
    for (uint32_t i = 0; i < LOOKUP_DIM; i++) p->lookup[i] = -1;

    /* Parse header */
    char linebuf[2048];
    if (!fgets(linebuf, sizeof(linebuf), fp)) {
        fprintf(stderr, "ds4_vqb2_pack: empty index %s\n", path);
        fclose(fp);
        return false;
    }
    char *headers[64];
    int n_hdr = split_csv_line(linebuf, headers, 64);
    csv_cols_t cols = {
        .layer = find_col(headers, n_hdr, "layer"),
        .kind_id = find_col(headers, n_hdr, "kind_id"),
        .k = find_col(headers, n_hdr, "k"),
        .row_start = find_col(headers, n_hdr, "row_start"),
        .pack_offset = find_col(headers, n_hdr, "pack_offset"),
        .packet_bytes = find_col(headers, n_hdr, "packet_bytes"),
        .header_n_experts = find_col(headers, n_hdr, "header_n_experts"),
        .header_n_rows = find_col(headers, n_hdr, "header_n_rows"),
        .header_n_pairs = find_col(headers, n_hdr, "header_n_pairs"),
        .header_bit_width = find_col(headers, n_hdr, "header_bit_width"),
        .header_n_codes = find_col(headers, n_hdr, "header_n_codes"),
    };
    if (cols.layer < 0 || cols.kind_id < 0 || cols.k < 0 || cols.row_start < 0 ||
        cols.pack_offset < 0 || cols.packet_bytes < 0 ||
        cols.header_n_experts < 0 || cols.header_n_rows < 0 || cols.header_n_pairs < 0 ||
        cols.header_bit_width < 0 || cols.header_n_codes < 0) {
        fprintf(stderr, "ds4_vqb2_pack: index %s missing required columns\n", path);
        fclose(fp);
        return false;
    }

    /* Parse rows */
    uint32_t row_idx = 0;
    while (fgets(linebuf, sizeof(linebuf), fp) && row_idx < cap) {
        char *fields[64];
        int n_fields = split_csv_line(linebuf, fields, 64);
        if (n_fields < n_hdr) continue;  /* malformed row, skip */
        if (fields[cols.layer][0] == '\0') continue;  /* empty row */

        ds4_vqb2_pack_entry *e = &p->entries[row_idx];
        e->layer        = (uint32_t)strtoul(fields[cols.layer], NULL, 10);
        e->kind_id      = (uint32_t)strtoul(fields[cols.kind_id], NULL, 10);
        e->k            = (uint32_t)strtoul(fields[cols.k], NULL, 10);
        e->row_start    = (uint32_t)strtoul(fields[cols.row_start], NULL, 10);
        e->pack_offset  = strtoull(fields[cols.pack_offset], NULL, 10);
        e->packet_bytes = strtoull(fields[cols.packet_bytes], NULL, 10);
        e->n_experts    = (uint32_t)strtoul(fields[cols.header_n_experts], NULL, 10);
        e->n_rows       = (uint32_t)strtoul(fields[cols.header_n_rows], NULL, 10);
        e->n_pairs      = (uint32_t)strtoul(fields[cols.header_n_pairs], NULL, 10);
        e->bit_width    = (uint32_t)strtoul(fields[cols.header_bit_width], NULL, 10);
        e->n_codes      = strtoull(fields[cols.header_n_codes], NULL, 10);

        const int32_t slot = lookup_slot(e->layer, e->kind_id, e->row_start);
        if (slot >= 0) {
            if (p->lookup[slot] != -1) {
                fprintf(stderr,
                    "ds4_vqb2_pack: duplicate cell (layer=%u, kind=%u, row_start=%u) at row %u\n",
                    e->layer, e->kind_id, e->row_start, row_idx);
            }
            p->lookup[slot] = (int32_t)row_idx;
        }
        row_idx++;
    }
    p->n_entries = row_idx;
    fclose(fp);
    return true;
}

bool ds4_vqb2_pack_open(const char *pack_path,
                        const char *index_csv_path,
                        ds4_vqb2_pack *out) {
    if (!pack_path || !index_csv_path || !out) return false;
    memset(out, 0, sizeof(*out));
    out->fd = -1;
    strncpy(out->pack_path, pack_path, sizeof(out->pack_path) - 1);

    if (!parse_index_csv(index_csv_path, out)) {
        ds4_vqb2_pack_close(out);
        return false;
    }

    int fd = open(pack_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "ds4_vqb2_pack: open pack %s failed: %s\n", pack_path, strerror(errno));
        ds4_vqb2_pack_close(out);
        return false;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "ds4_vqb2_pack: fstat %s failed: %s\n", pack_path, strerror(errno));
        close(fd);
        ds4_vqb2_pack_close(out);
        return false;
    }
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        fprintf(stderr, "ds4_vqb2_pack: mmap %s failed: %s\n", pack_path, strerror(errno));
        close(fd);
        ds4_vqb2_pack_close(out);
        return false;
    }
    out->fd = fd;
    out->map = map;
    out->map_size = (size_t)st.st_size;
    out->pack_size = (uint64_t)st.st_size;

    /* Validate every entry's offset+bytes fits within pack */
    for (uint32_t i = 0; i < out->n_entries; i++) {
        const ds4_vqb2_pack_entry *e = &out->entries[i];
        if (e->pack_offset + e->packet_bytes > out->pack_size) {
            fprintf(stderr,
                "ds4_vqb2_pack: entry %u (L%u K%u rs%u) offset=%llu + bytes=%llu > pack_size=%llu\n",
                i, e->layer, e->kind_id, e->row_start,
                (unsigned long long)e->pack_offset,
                (unsigned long long)e->packet_bytes,
                (unsigned long long)out->pack_size);
            ds4_vqb2_pack_close(out);
            return false;
        }
    }
    return true;
}

void ds4_vqb2_pack_close(ds4_vqb2_pack *p) {
    if (!p) return;
    if (p->map && p->map != MAP_FAILED) munmap(p->map, p->map_size);
    if (p->fd >= 0) close(p->fd);
    free(p->entries);
    free(p->lookup);
    memset(p, 0, sizeof(*p));
    p->fd = -1;
}

int32_t ds4_vqb2_pack_lookup_index(const ds4_vqb2_pack *p,
                                   uint32_t layer, uint32_t kind_id,
                                   uint32_t row_start) {
    if (!p || !p->lookup) return -1;
    const int32_t slot = lookup_slot(layer, kind_id, row_start);
    if (slot < 0) return -1;
    return p->lookup[slot];
}

bool ds4_vqb2_pack_view_from_entry(const ds4_vqb2_pack *p,
                                   uint32_t entry_index,
                                   ds4_vqb2_file *out_view) {
    if (!p || !p->map || !out_view || entry_index >= p->n_entries) return false;
    const ds4_vqb2_pack_entry *e = &p->entries[entry_index];
    const uint8_t *base = (const uint8_t *)p->map + e->pack_offset;

    /* Verify magic at start of packet — defensive */
    if (memcmp(base, DS4_VQB2_MAGIC, 4) != 0) {
        fprintf(stderr,
            "ds4_vqb2_pack: bad magic at entry %u (L%u K%u rs%u offset=%llu)\n",
            entry_index, e->layer, e->kind_id, e->row_start,
            (unsigned long long)e->pack_offset);
        return false;
    }
    /* Cache-audit finding A2: re-parse the 40-byte packet header at base and
     * verify it matches the CSV index entry. Catches CSV/pack drift cheaply. */
    uint32_t hdr_version, hdr_n_experts, hdr_n_rows, hdr_n_pairs;
    uint32_t hdr_layer, hdr_kind_id, hdr_k, hdr_bit_width, hdr_n_codes_lo;
    memcpy(&hdr_version,   base + 4,  4);
    memcpy(&hdr_n_experts, base + 8,  4);
    memcpy(&hdr_n_rows,    base + 12, 4);
    memcpy(&hdr_n_pairs,   base + 16, 4);
    memcpy(&hdr_layer,     base + 20, 4);
    memcpy(&hdr_kind_id,   base + 24, 4);
    memcpy(&hdr_k,         base + 28, 4);
    memcpy(&hdr_bit_width, base + 32, 4);
    memcpy(&hdr_n_codes_lo,base + 36, 4);
    if (hdr_n_experts != e->n_experts || hdr_n_rows != e->n_rows ||
        hdr_n_pairs   != e->n_pairs   || hdr_layer != e->layer ||
        hdr_kind_id   != e->kind_id   || hdr_k != e->k ||
        hdr_bit_width != e->bit_width || hdr_n_codes_lo != (uint32_t)e->n_codes) {
        fprintf(stderr,
            "ds4_vqb2_pack: CSV vs header drift at entry %u\n"
            "  CSV:    L%u kind=%u rs=%u K=%u bw=%u experts=%u rows=%u pairs=%u n_codes=%llu\n"
            "  header: L%u kind=%u (no rs in header) K=%u bw=%u experts=%u rows=%u pairs=%u n_codes=%u\n",
            entry_index,
            e->layer, e->kind_id, e->row_start, e->k, e->bit_width,
            e->n_experts, e->n_rows, e->n_pairs, (unsigned long long)e->n_codes,
            hdr_layer, hdr_kind_id, hdr_k, hdr_bit_width,
            hdr_n_experts, hdr_n_rows, hdr_n_pairs, hdr_n_codes_lo);
        return false;
    }
    (void)hdr_version;

    memset(out_view, 0, sizeof(*out_view));
    out_view->fd = -1;
    out_view->map = NULL;
    out_view->map_size = 0;
    out_view->version = 1;
    out_view->n_experts = e->n_experts;
    out_view->n_rows = e->n_rows;
    out_view->n_pairs = e->n_pairs;
    out_view->layer = e->layer;
    out_view->kind_id = e->kind_id;
    out_view->k = e->k;
    out_view->bit_width = e->bit_width;
    out_view->n_codes = e->n_codes;
    out_view->code_mask = (uint8_t)((1u << e->bit_width) - 1u);

    const size_t cb_bytes = (size_t)e->k * 2u * sizeof(float);
    const size_t codes_bytes = ds4_vqb2_codes_bytes(e->n_codes, e->bit_width);
    const size_t header_bytes = DS4_VQB2_HEADER_BYTES;
    if (header_bytes + cb_bytes + codes_bytes != e->packet_bytes) {
        fprintf(stderr,
            "ds4_vqb2_pack: entry %u layout mismatch (header+cb+codes=%zu vs packet=%llu)\n",
            entry_index, header_bytes + cb_bytes + codes_bytes,
            (unsigned long long)e->packet_bytes);
        return false;
    }
    out_view->codebook    = (const float   *)(base + header_bytes);
    out_view->codes       = (const uint8_t *)(base + header_bytes + cb_bytes);
    out_view->codes_bytes = codes_bytes;
    return true;
}

bool ds4_vqb2_pack_get_view(const ds4_vqb2_pack *p,
                            uint32_t layer, uint32_t kind_id, uint32_t row_start,
                            ds4_vqb2_file *out_view) {
    const int32_t idx = ds4_vqb2_pack_lookup_index(p, layer, kind_id, row_start);
    if (idx < 0) return false;
    return ds4_vqb2_pack_view_from_entry(p, (uint32_t)idx, out_view);
}

int ds4_vqb2_pack_for_layer(const ds4_vqb2_pack *p,
                            uint32_t layer,
                            ds4_vqb2_pack_layer_cb cb,
                            void *userdata) {
    if (!p || !cb) return -1;
    int visited = 0;
    /* Walk lookup in (kind_id, row_block_idx) order so callback sees
     * gate-rows-0..15, up-rows-0..15, down-rows-0..31 — the layer-major
     * access pattern H2125 calls out as the canonical decode order. */
    for (uint32_t kind = 0; kind < DS4_VQB2_PACK_N_KIND; kind++) {
        for (uint32_t rb = 0; rb < DS4_VQB2_PACK_MAX_ROW_BLOCKS; rb++) {
            const int32_t slot = lookup_slot(layer, kind, rb * 128u);
            if (slot < 0) continue;
            const int32_t idx = p->lookup[slot];
            if (idx < 0) continue;
            ds4_vqb2_file view;
            if (!ds4_vqb2_pack_view_from_entry(p, (uint32_t)idx, &view)) continue;
            const int rc = cb((uint32_t)idx, &p->entries[idx], &view, userdata);
            visited++;
            if (rc < 0) return -visited;  /* callback abort */
        }
    }
    return visited;
}

/* silv 2026-05-28 — smoke test entry point used by --vqb2-pack-probe.
 * Opens pack, prints summary, materializes views from 5 entries spread
 * across the pack range, decodes a sample pair from each, and (if a
 * matching standalone packet file is referenced) cross-checks bytes. */
/* Internal helper: locate one standalone packet file path corresponding to a
 * pack entry. Returns "" on parse failure. Looks at the source_path column. */
static int find_source_path_for_entry(const char *index_csv_path,
                                      uint32_t target_idx,
                                      char *out_path, size_t out_capacity) {
    if (!index_csv_path || !out_path || out_capacity == 0) return 0;
    FILE *fp = fopen(index_csv_path, "r");
    if (!fp) return 0;
    char header[2048];
    if (!fgets(header, sizeof(header), fp)) { fclose(fp); return 0; }
    char *headers[64];
    int n_hdr = split_csv_line(header, headers, 64);
    int src_col = find_col(headers, n_hdr, "source_path");
    if (src_col < 0) { fclose(fp); return 0; }
    char linebuf[4096];
    uint32_t row = 0;
    while (fgets(linebuf, sizeof(linebuf), fp)) {
        if (row == target_idx) {
            char *fields[64];
            int n_fields = split_csv_line(linebuf, fields, 64);
            if (n_fields > src_col) {
                strncpy(out_path, fields[src_col], out_capacity - 1);
                out_path[out_capacity - 1] = '\0';
                fclose(fp);
                return 1;
            }
            break;
        }
        row++;
    }
    fclose(fp);
    return 0;
}

int ds4_cli_vqb2_pack_probe(const char *pack_path, const char *index_csv_path) {
    ds4_vqb2_pack pack;
    if (!ds4_vqb2_pack_open(pack_path, index_csv_path, &pack)) {
        fprintf(stderr, "ds4_cli_vqb2_pack_probe: open failed\n");
        return 1;
    }
    ds4_vqb2_pack_print_summary(&pack);

    /* Cross-check: pick entry 0, open the standalone packet file, decode the
     * same (e=0, r=0, p=0), and verify it matches the pack-view decode. */
    {
        char src_path[1024] = {0};
        if (find_source_path_for_entry(index_csv_path, 0, src_path, sizeof(src_path)) &&
            src_path[0] != '\0') {
            ds4_vqb2_file standalone;
            ds4_vqb2_file pack_view;
            const int stand_ok = ds4_vqb2_open(src_path, &standalone);
            const int view_ok = ds4_vqb2_pack_view_from_entry(&pack, 0, &pack_view);
            if (stand_ok && view_ok) {
                float s_re = 0, s_im = 0, v_re = 0, v_im = 0;
                ds4_vqb2_decode_pair(&standalone, 0, 0, 0, &s_re, &s_im);
                ds4_vqb2_decode_pair(&pack_view,  0, 0, 0, &v_re, &v_im);
                const int match = (s_re == v_re) && (s_im == v_im);
                fprintf(stderr,
                    "  cross-check entry 0: standalone=(%+.6f,%+.6f) pack_view=(%+.6f,%+.6f) %s\n",
                    s_re, s_im, v_re, v_im, match ? "MATCH" : "MISMATCH");
            }
            if (stand_ok) ds4_vqb2_close(&standalone);
            if (view_ok)  ds4_vqb2_close(&pack_view);
        } else {
            fprintf(stderr, "  cross-check: no source_path column or standalone file unavailable\n");
        }
    }

    if (pack.n_entries == 0) {
        ds4_vqb2_pack_close(&pack);
        fprintf(stderr, "ds4_cli_vqb2_pack_probe: empty pack\n");
        return 1;
    }
    /* Sample 5 entries: first, last, and three evenly spaced */
    const uint32_t sample_n = 5;
    const uint32_t step = pack.n_entries > sample_n ? pack.n_entries / sample_n : 1;
    int ok_views = 0;
    for (uint32_t s = 0; s < sample_n; s++) {
        const uint32_t idx = (s * step) < pack.n_entries ? (s * step) : pack.n_entries - 1;
        const ds4_vqb2_pack_entry *e = &pack.entries[idx];
        ds4_vqb2_file view;
        if (!ds4_vqb2_pack_view_from_entry(&pack, idx, &view)) {
            fprintf(stderr, "  sample %u idx=%u: view materialize FAILED\n", s, idx);
            continue;
        }
        /* Decode (expert=0, row=0, pair=0) — proves codebook + codes pointers
         * are correctly positioned within the pack. */
        float re = 0.f, im = 0.f;
        const int dec_ok = ds4_vqb2_decode_pair(&view, 0, 0, 0, &re, &im);
        fprintf(stderr,
            "  sample %u idx=%u: L%u kind=%u rs=%u  K=%u bw=%u  experts=%u rows=%u pairs=%u  "
            "offset=%llu bytes=%llu  decode[0,0,0]=(%+.4f, %+.4f) %s\n",
            s, idx, e->layer, e->kind_id, e->row_start, e->k, e->bit_width,
            e->n_experts, e->n_rows, e->n_pairs,
            (unsigned long long)e->pack_offset, (unsigned long long)e->packet_bytes,
            re, im, dec_ok ? "OK" : "DECODE-FAIL");
        if (dec_ok) ok_views++;
        ds4_vqb2_close(&view);  /* safe: view's fd=-1, map=NULL */
    }
    /* Spot-check O(1) lookup: pick a known entry, look it up, confirm match. */
    if (pack.n_entries > 100) {
        const ds4_vqb2_pack_entry *probe = &pack.entries[pack.n_entries / 2];
        const int32_t found = ds4_vqb2_pack_lookup_index(
            &pack, probe->layer, probe->kind_id, probe->row_start);
        const int lookup_ok = (found >= 0 && (uint32_t)found < pack.n_entries &&
                                pack.entries[found].pack_offset == probe->pack_offset);
        fprintf(stderr,
            "  lookup probe (L%u kind=%u rs=%u): found_idx=%d %s\n",
            probe->layer, probe->kind_id, probe->row_start, found,
            lookup_ok ? "OK" : "MISMATCH");
        if (lookup_ok) ok_views++;
    }
    ds4_vqb2_pack_close(&pack);
    fprintf(stderr, "ds4_cli_vqb2_pack_probe: %d/%u views decoded\n", ok_views, sample_n + 1);
    return ok_views == (int)sample_n + 1 ? 0 : 1;
}

/* Layer-tour callback: count and decode-probe each entry. */
typedef struct {
    uint32_t n_visited;
    uint32_t n_decoded;
    uint32_t n_gate;
    uint32_t n_up;
    uint32_t n_down;
    uint32_t k_min;
    uint32_t k_max;
} layer_tour_state;

static int layer_tour_cb(uint32_t entry_idx,
                         const ds4_vqb2_pack_entry *entry,
                         const ds4_vqb2_file *view,
                         void *userdata) {
    layer_tour_state *s = (layer_tour_state *)userdata;
    s->n_visited++;
    float re = 0.f, im = 0.f;
    if (ds4_vqb2_decode_pair(view, 0, 0, 0, &re, &im)) s->n_decoded++;
    if (entry->kind_id == 0) s->n_gate++;
    else if (entry->kind_id == 1) s->n_up++;
    else if (entry->kind_id == 2) s->n_down++;
    if (entry->k < s->k_min) s->k_min = entry->k;
    if (entry->k > s->k_max) s->k_max = entry->k;
    (void)entry_idx;
    return 0;
}

int ds4_cli_vqb2_pack_layer_tour(const char *pack_path, const char *index_csv_path, uint32_t layer) {
    ds4_vqb2_pack pack;
    if (!ds4_vqb2_pack_open(pack_path, index_csv_path, &pack)) {
        fprintf(stderr, "ds4_cli_vqb2_pack_layer_tour: open failed\n");
        return 1;
    }
    layer_tour_state s = { .n_visited = 0, .n_decoded = 0,
                           .n_gate = 0, .n_up = 0, .n_down = 0,
                           .k_min = 0xFFFFFFFFu, .k_max = 0 };
    const int visited = ds4_vqb2_pack_for_layer(&pack, layer, layer_tour_cb, &s);
    fprintf(stderr,
        "ds4_cli_vqb2_pack_layer_tour: layer=%u visited=%d (gate=%u up=%u down=%u) "
        "decoded=%u K=[%u, %u]\n",
        layer, visited, s.n_gate, s.n_up, s.n_down, s.n_decoded,
        s.k_min == 0xFFFFFFFFu ? 0 : s.k_min, s.k_max);
    ds4_vqb2_pack_close(&pack);
    /* Standard DS4 V4 layer: 16 gate + 16 up + 32 down = 64 packets. */
    return (visited == 64 && s.n_decoded == 64) ? 0 : 1;
}

/* Cache-audit A1 fix: pack-aware loader. Iterates entries in layer-major
 * order (preserves H2125 locality) and pins each expert × row_block. */
/* Pin one entry into the hot store. Pulled out so the layer-filter and
 * full-scan paths share identical semantics. */
static int load_entry_to_hot_store(struct ds4_hot_expert_store *store,
                                   const ds4_vqb2_pack *p,
                                   uint32_t i, int *n_pinned, int *n_errors) {
    const ds4_vqb2_pack_entry *e = &p->entries[i];
    if ((e->row_start & 127u) != 0) {
        fprintf(stderr,
            "ds4_vqb2_pack_load_to_hot_store: entry %u row_start=%u not multiple of 128\n",
            i, e->row_start);
        (*n_errors)++;
        return -1;
    }
    const uint32_t row_block = e->row_start >> 7;
    ds4_vqb2_file view;
    if (!ds4_vqb2_pack_view_from_entry(p, i, &view)) {
        (*n_errors)++;
        return -1;
    }
    for (uint32_t expert = 0; expert < e->n_experts; expert++) {
        if (ds4_hot_pin_expert_from_vqb2(store, e->layer, e->kind_id,
                                          row_block, &view, expert) == 0) {
            (*n_pinned)++;
        } else {
            (*n_errors)++;
        }
    }
    /* view shares pack mmap — ds4_vqb2_close is safe no-op */
    return 0;
}

int ds4_vqb2_pack_load_to_hot_store(struct ds4_hot_expert_store *store,
                                    const ds4_vqb2_pack *p,
                                    int layer_filter, int kind_filter) {
    if (!store || !p) return -1;
    int n_pinned = 0;
    int n_errors = 0;

    if (layer_filter >= 0 && (uint32_t)layer_filter < DS4_VQB2_PACK_MAX_LAYER) {
        /* O(N_per_layer) path via lookup array: ≤96 slots, not p->n_entries. */
        const uint32_t L = (uint32_t)layer_filter;
        const uint32_t k_lo = kind_filter >= 0 ? (uint32_t)kind_filter : 0u;
        const uint32_t k_hi = kind_filter >= 0 ? (uint32_t)kind_filter + 1u
                                               : DS4_VQB2_PACK_N_KIND;
        for (uint32_t kind = k_lo; kind < k_hi && kind < DS4_VQB2_PACK_N_KIND; kind++) {
            for (uint32_t rb = 0; rb < DS4_VQB2_PACK_MAX_ROW_BLOCKS; rb++) {
                const int32_t slot = lookup_slot(L, kind, rb << 7);
                if (slot < 0) continue;
                const int32_t idx = p->lookup[slot];
                if (idx < 0) continue;
                load_entry_to_hot_store(store, p, (uint32_t)idx, &n_pinned, &n_errors);
            }
        }
    } else {
        /* Full scan: layer_filter=-1 or out-of-range. */
        for (uint32_t i = 0; i < p->n_entries; i++) {
            const ds4_vqb2_pack_entry *e = &p->entries[i];
            if (kind_filter >= 0 && (int)e->kind_id != kind_filter) continue;
            load_entry_to_hot_store(store, p, i, &n_pinned, &n_errors);
        }
    }
    fprintf(stderr,
        "ds4_vqb2_pack_load_to_hot_store: layer_filter=%d kind_filter=%d "
        "pinned=%d errors=%d (of %u entries)\n",
        layer_filter, kind_filter, n_pinned, n_errors, p->n_entries);
    return n_errors == 0 ? n_pinned : -1;
}

void ds4_vqb2_pack_print_summary(const ds4_vqb2_pack *p) {
    if (!p) { fprintf(stderr, "ds4_vqb2_pack: (null)\n"); return; }
    fprintf(stderr, "ds4_vqb2_pack: pack=%s  pack_size=%llu  entries=%u\n",
            p->pack_path, (unsigned long long)p->pack_size, p->n_entries);
    /* K mix */
    uint32_t k4 = 0, k16 = 0, k256 = 0, k_other = 0;
    uint32_t kind_gate = 0, kind_up = 0, kind_down = 0;
    uint32_t min_layer = 0xFFFFFFFFu, max_layer = 0;
    for (uint32_t i = 0; i < p->n_entries; i++) {
        const ds4_vqb2_pack_entry *e = &p->entries[i];
        if (e->k == 4) k4++;
        else if (e->k == 16) k16++;
        else if (e->k == 256) k256++;
        else k_other++;
        if (e->kind_id == 0) kind_gate++;
        else if (e->kind_id == 1) kind_up++;
        else if (e->kind_id == 2) kind_down++;
        if (e->layer < min_layer) min_layer = e->layer;
        if (e->layer > max_layer) max_layer = e->layer;
    }
    fprintf(stderr, "  K mix:    K4=%u  K16=%u  K256=%u  other=%u\n", k4, k16, k256, k_other);
    fprintf(stderr, "  kind mix: gate=%u  up=%u  down=%u\n", kind_gate, kind_up, kind_down);
    fprintf(stderr, "  layers:   [%u, %u]\n", min_layer, max_layer);
}
