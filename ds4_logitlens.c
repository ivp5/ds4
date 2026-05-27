/* ds4_logitlens — observe top-K logprobs at the next-token position after prefill.
 *
 * Purpose: directly answer "does the model HAVE the answer in its distribution
 * at the commit position." For AIME P10 the no-commit + commit-forced-wrong
 * evidence so far suggests no, but we haven't looked at the distribution.
 *
 * Usage: ./ds4-logitlens -m MODEL.gguf --prompt-file FILE [-k N] [--prefill-metal-phases auto] [--cpu-moe]
 * Output: top-K tokens with text + logprob at the next-token position.
 */
#include "ds4.h"
#include "ds4_expert_table.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: ds4-logitlens -m MODEL --prompt-file FILE [-k N] [--cpu-moe] [--prefill-metal-phases auto|N]\n"
        "  Loads the model, prefills the given prompt, dumps the top-K next-token logprobs.\n"
        "  -k N           Number of top tokens to print. Default 30.\n"
    );
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(fp, 0, SEEK_END);
    long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)n + 1);
    fread(buf, 1, (size_t)n, fp);
    buf[n] = '\0';
    fclose(fp);
    return buf;
}

/* Sync session under current skip-mask, optionally dump logits binary,
 * optionally emit top-K stdout. Returns 0 on success. */
static int sync_and_dump(ds4_session *session, ds4_engine *engine,
                         const ds4_tokens *prompt, const char *out_path,
                         bool emit_top, int top_k) {
    char err[256];
    ds4_session_invalidate(session);
    if (ds4_session_sync(session, prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "[logitlens] sync failed: %s\n", err);
        return 1;
    }
    if (out_path) {
        const size_t V = (size_t)DS4_VOCAB_SIZE;
        float *logits = (float *)malloc(V * sizeof(float));
        if (!ds4_session_dump_logits(session, logits, V)) {
            fprintf(stderr, "[logitlens] dump_logits failed\n");
            free(logits); return 1;
        }
        FILE *fp = fopen(out_path, "wb");
        if (!fp) { fprintf(stderr, "cannot open %s\n", out_path); free(logits); return 1; }
        uint32_t header[2] = { (uint32_t)V, 0u };
        fwrite(header, sizeof(uint32_t), 2, fp);
        fwrite(logits, sizeof(float), V, fp);
        fclose(fp);
        free(logits);
    }
    if (emit_top) {
        ds4_token_score *scores = (ds4_token_score *)calloc((size_t)top_k, sizeof(scores[0]));
        int got = ds4_session_top_logprobs(session, scores, top_k);
        if (got <= 0) { fprintf(stderr, "top_logprobs returned %d\n", got); free(scores); return 1; }
        printf("rank,token_id,logprob,logit,text\n");
        for (int i = 0; i < got; i++) {
            size_t tlen = 0;
            char *t = ds4_token_text(engine, scores[i].id, &tlen);
            char buf[256]; size_t out = 0;
            for (size_t j = 0; j < tlen && out + 4 < sizeof(buf); j++) {
                unsigned char c = (unsigned char)t[j];
                if (c == '\n') { buf[out++] = '\\'; buf[out++] = 'n'; }
                else if (c == '\t') { buf[out++] = '\\'; buf[out++] = 't'; }
                else if (c == '\r') { buf[out++] = '\\'; buf[out++] = 'r'; }
                else if (c < 0x20 || c == '"' || c == ',') {
                    out += snprintf(buf+out, sizeof(buf)-out, "\\x%02x", c);
                }
                else buf[out++] = (char)c;
            }
            buf[out] = '\0';
            printf("%d,%d,%.4f,%.4f,\"%s\"\n", i, scores[i].id, scores[i].logprob, scores[i].logit, buf);
            free(t);
        }
        free(scores);
    }
    return 0;
}

/* silv 2026-05-27 task #659 — variant of sync_and_dump that emits the
 * top-K CSV (rank,token_id,logprob,logit,text) to a NAMED FILE instead
 * of stdout. The daemon batched-organ-skip mode uses this so each
 * config writes structured CSV the harm_score.py driver can read.
 * Mirrors the stdout emit branch above; returns 0 on success. */
static int sync_and_dump_csv_to_file(ds4_session *session, ds4_engine *engine,
                                      const ds4_tokens *prompt,
                                      const char *out_csv_path, int top_k) {
    char err[256];
    ds4_session_invalidate(session);
    if (ds4_session_sync(session, prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "[logitlens] sync failed: %s\n", err);
        return 1;
    }
    FILE *fp = fopen(out_csv_path, "w");
    if (!fp) { fprintf(stderr, "cannot open %s\n", out_csv_path); return 1; }
    ds4_token_score *scores = (ds4_token_score *)calloc((size_t)top_k, sizeof(scores[0]));
    int got = ds4_session_top_logprobs(session, scores, top_k);
    if (got <= 0) {
        fprintf(stderr, "top_logprobs returned %d\n", got);
        free(scores); fclose(fp); return 1;
    }
    fprintf(fp, "rank,token_id,logprob,logit,text\n");
    for (int i = 0; i < got; i++) {
        size_t tlen = 0;
        char *t = ds4_token_text(engine, scores[i].id, &tlen);
        char buf[256]; size_t out = 0;
        for (size_t j = 0; j < tlen && out + 4 < sizeof(buf); j++) {
            unsigned char c = (unsigned char)t[j];
            if (c == '\n') { buf[out++] = '\\'; buf[out++] = 'n'; }
            else if (c == '\t') { buf[out++] = '\\'; buf[out++] = 't'; }
            else if (c == '\r') { buf[out++] = '\\'; buf[out++] = 'r'; }
            else if (c < 0x20 || c == '"' || c == ',') {
                out += snprintf(buf+out, sizeof(buf)-out, "\\x%02x", c);
            }
            else buf[out++] = (char)c;
        }
        buf[out] = '\0';
        fprintf(fp, "%d,%d,%.4f,%.4f,\"%s\"\n", i, scores[i].id, scores[i].logprob, scores[i].logit, buf);
        free(t);
    }
    free(scores);
    fclose(fp);
    return 0;
}

/* Parse a comma-separated list of unsigned integers into out[], up to max_n.
 * Returns the number parsed. */
static int parse_uint_list(const char *s, uint32_t *out, int max_n) {
    int n = 0;
    while (s && *s && n < max_n) {
        char *end = NULL;
        long v = strtol(s, &end, 10);
        if (end == s) break;
        if (v >= 0) out[n++] = (uint32_t)v;
        if (*end == ',') s = end + 1;
        else s = end;
    }
    return n;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt_path = NULL;
    int k = 30;
    bool cpu_moe = false;
    int n_cpu_moe_layers = 0;
    int prefill_metal_phases = 0;
    const char *dump_kv_layers_arg = NULL;
    const char *dump_kv_positions_arg = NULL;
    bool derope = false;
    const char *dump_kv_out_path = NULL;
    const char *dump_logits_out_path = NULL;
    const char *skip_configs_file = NULL;
    const char *organ_skip_configs_file = NULL;  /* silv 2026-05-27 task #659 */
    bool quiet = false;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(stdout); return 0; }
        else if (!strcmp(a, "-m") || !strcmp(a, "--model")) model_path = argv[++i];
        else if (!strcmp(a, "--prompt-file")) prompt_path = argv[++i];
        else if (!strcmp(a, "-k")) k = atoi(argv[++i]);
        else if (!strcmp(a, "--cpu-moe")) cpu_moe = true;
        else if (!strcmp(a, "--n-cpu-moe")) n_cpu_moe_layers = atoi(argv[++i]);
        else if (!strcmp(a, "--dump-kv-layers")) dump_kv_layers_arg = argv[++i];
        else if (!strcmp(a, "--dump-kv-positions")) dump_kv_positions_arg = argv[++i];
        else if (!strcmp(a, "--derope")) derope = true;
        else if (!strcmp(a, "--dump-kv-out")) dump_kv_out_path = argv[++i];
        else if (!strcmp(a, "--dump-logits-out")) dump_logits_out_path = argv[++i];
        else if (!strcmp(a, "--skip-configs-file")) skip_configs_file = argv[++i];
        else if (!strcmp(a, "--organ-skip-configs-file")) organ_skip_configs_file = argv[++i];
        else if (!strcmp(a, "--quiet")) quiet = true;
        else if (!strcmp(a, "--prefill-metal-phases")) {
            const char *s = argv[++i];
            prefill_metal_phases = !strcmp(s, "auto") ? -1 : atoi(s);
        }
        else { fprintf(stderr, "unknown: %s\n", a); usage(stderr); return 2; }
    }
    if (!model_path || !prompt_path) { usage(stderr); return 2; }

    ds4_engine_options opt = {
        .model_path = model_path,
        .backend = DS4_BACKEND_METAL,
        .n_threads = 10,
        .cpu_moe = cpu_moe,
        .n_cpu_moe_layers = n_cpu_moe_layers,
        .prefill_metal_phases = prefill_metal_phases,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }

    char *text = read_file(prompt_path);
    fprintf(stderr, "[logitlens] prompt bytes: %zu\n", strlen(text));
    ds4_tokens prompt = {0};
    ds4_tokenize_text(engine, text, &prompt);
    fprintf(stderr, "[logitlens] tokenized to %d tokens\n", prompt.len);

    /* ctx must hold prompt + a bit of headroom. */
    int ctx_size = prompt.len + 16;
    if (ctx_size < 1024) ctx_size = 1024;
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, ctx_size) != 0) { fprintf(stderr, "session create failed\n"); return 1; }

    if (skip_configs_file) {
        FILE *cfp = fopen(skip_configs_file, "r");
        if (!cfp) { fprintf(stderr, "cannot open %s\n", skip_configs_file); return 1; }
        char line[1024];
        int n_configs = 0;
        struct timespec ts0; clock_gettime(CLOCK_MONOTONIC, &ts0);
        while (fgets(line, sizeof(line), cfp)) {
            line[strcspn(line, "\n")] = '\0';
            if (line[0] == '\0' || line[0] == '#') continue;
            char *name = strtok(line, " \t");
            char *skip_csv = strtok(NULL, " \t");
            char *out_path = strtok(NULL, " \t");
            if (!name || !skip_csv || !out_path) { fprintf(stderr, "[logitlens] bad config line\n"); continue; }
            if (!strcmp(skip_csv, "-") || !strcmp(skip_csv, "()")) skip_csv = NULL;
            ds4_set_skip_list(skip_csv);
            struct timespec ts1; clock_gettime(CLOCK_MONOTONIC, &ts1);
            double t1 = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) / 1e9;
            if (!quiet) fprintf(stderr, "[logitlens] config %s skip=%s out=%s t=%.2fs\n",
                                name, skip_csv ? skip_csv : "(none)", out_path, t1);
            if (sync_and_dump(session, engine, &prompt, out_path, false, k) != 0) {
                fprintf(stderr, "[logitlens] config %s failed\n", name);
            }
            n_configs++;
        }
        fclose(cfp);
        struct timespec tsE; clock_gettime(CLOCK_MONOTONIC, &tsE);
        double total = (tsE.tv_sec - ts0.tv_sec) + (tsE.tv_nsec - ts0.tv_nsec) / 1e9;
        fprintf(stderr, "[logitlens] batched %d configs in %.2fs (%.2fs/config)\n",
                n_configs, total, total / (n_configs > 0 ? n_configs : 1));
    } else if (organ_skip_configs_file) {
        /* silv 2026-05-27 task #659 — daemon mode for organ-skip A/B sweeps.
         *
         * Config file format (one per line; '#' = comment, blanks ignored):
         *   <name>  <cells_csv_path_or_-> <out_top_k_path>
         *
         * Where <cells_csv_path_or_-> is either:
         *   - the path to a cells CSV ("L,E,K" rows, same as DS4_ORGAN_SKIP_CSV)
         *   - "-" or "()" → baseline (no organ-skip)
         *
         * Per iteration:
         *   1. ds4_organ_skip_reset() — clear any prior cells
         *   2. ds4_load_organ_skip_csv(cells_path) — load new cells
         *   3. ds4_session_invalidate + ds4_session_sync — re-prefill (cache reset
         *      because routing differs per ablation)
         *   4. ds4_session_top_logprobs → top-K written to <out_top_k_path>
         *
         * Model is loaded ONCE; per-config overhead is just re-prefill (~1s on
         * warm pages for a 25-byte prompt). For a 13-config sweep, that's ~15
         * seconds vs the 13-launch SLOW path's ~3 hours (cold model = ~5min/launch).
         *
         * This is the OOM speedup ladder's first lever from DESIGN.md (model-load
         * amortization). KV-shared-prefix + structural-panel + batched-case
         * remain follow-up levers if iteration count grows past ~100. */
        FILE *cfp = fopen(organ_skip_configs_file, "r");
        if (!cfp) { fprintf(stderr, "cannot open %s\n", organ_skip_configs_file); return 1; }
        char line[1024];
        int n_configs = 0;
        struct timespec ts0; clock_gettime(CLOCK_MONOTONIC, &ts0);
        while (fgets(line, sizeof(line), cfp)) {
            line[strcspn(line, "\n")] = '\0';
            if (line[0] == '\0' || line[0] == '#') continue;
            char *name = strtok(line, " \t");
            char *cells_path = strtok(NULL, " \t");
            char *out_path = strtok(NULL, " \t");
            if (!name || !cells_path || !out_path) {
                fprintf(stderr, "[logitlens] bad organ-skip config line\n"); continue;
            }
            ds4_organ_skip_reset();
            int n_cells = 0;
            if (strcmp(cells_path, "-") != 0 && strcmp(cells_path, "()") != 0) {
                n_cells = ds4_load_organ_skip_csv(cells_path);
                if (n_cells < 0) {
                    fprintf(stderr, "[logitlens] config %s: failed to load %s\n",
                            name, cells_path);
                    continue;
                }
            }
            struct timespec ts1; clock_gettime(CLOCK_MONOTONIC, &ts1);
            double t1 = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) / 1e9;
            if (!quiet) {
                fprintf(stderr, "[logitlens] organ_config %s cells=%s (%d) out=%s t=%.2fs\n",
                        name, cells_path, n_cells, out_path, t1);
            }
            if (sync_and_dump_csv_to_file(session, engine, &prompt, out_path, k) != 0) {
                fprintf(stderr, "[logitlens] organ_config %s failed\n", name);
            }
            n_configs++;
        }
        fclose(cfp);
        struct timespec tsE; clock_gettime(CLOCK_MONOTONIC, &tsE);
        double total = (tsE.tv_sec - ts0.tv_sec) + (tsE.tv_nsec - ts0.tv_nsec) / 1e9;
        fprintf(stderr, "[logitlens] organ-skip batched %d configs in %.2fs (%.2fs/config)\n",
                n_configs, total, total / (n_configs > 0 ? n_configs : 1));
    } else {
        fprintf(stderr, "[logitlens] syncing session (prefill %d tokens)...\n", prompt.len);
        if (sync_and_dump(session, engine, &prompt, dump_logits_out_path, !quiet, k) != 0) return 1;
    }

    /* Dump kv_raw at requested (layer, position) tuples */
    if (dump_kv_layers_arg && dump_kv_positions_arg) {
        uint32_t layers[64], positions[1024];
        int n_layers = parse_uint_list(dump_kv_layers_arg, layers, 64);
        int n_positions = parse_uint_list(dump_kv_positions_arg, positions, 1024);
        fprintf(stderr, "[logitlens] dumping kv_raw: %d layers x %d positions (derope=%d)\n",
                n_layers, n_positions, derope ? 1 : 0);

        FILE *fp = NULL;
        if (dump_kv_out_path) {
            fp = fopen(dump_kv_out_path, "wb");
            if (!fp) { fprintf(stderr, "cannot open %s\n", dump_kv_out_path); return 1; }
            /* Binary format: header { uint32 n_layers, uint32 n_positions, uint32 head_dim, uint32 derope_flag }
             * then float32[n_layers*n_positions*head_dim] in (layer, position, dim) order. */
            uint32_t header[4] = { (uint32_t)n_layers, (uint32_t)n_positions, (uint32_t)512, derope ? 1u : 0u };
            fwrite(header, sizeof(uint32_t), 4, fp);
            fwrite(layers, sizeof(uint32_t), n_layers, fp);
            fwrite(positions, sizeof(uint32_t), n_positions, fp);
        }

        const size_t HEAD_DIM = 512;
        float *buf = malloc(HEAD_DIM * sizeof(float));
        if (!fp) {
            /* Stdout text mode: one line per (layer, position) with summary stats */
            printf("\n# kv_raw dump (head_dim=%zu, derope=%d)\n", HEAD_DIM, derope ? 1 : 0);
            printf("layer,position,min,max,mean,abs_mean,std,first8...\n");
        }
        for (int li = 0; li < n_layers; li++) {
            for (int pi = 0; pi < n_positions; pi++) {
                uint32_t L = layers[li];
                uint32_t P = positions[pi];
                int ok = ds4_session_dump_kv_raw(session, L, P, buf, HEAD_DIM, derope ? 1 : 0);
                if (!ok) {
                    fprintf(stderr, "[logitlens] dump_kv_raw failed for layer=%u pos=%u\n", L, P);
                    if (fp) {
                        float zero = 0.0f;
                        for (size_t z = 0; z < HEAD_DIM; z++) fwrite(&zero, sizeof(float), 1, fp);
                    }
                    continue;
                }
                if (fp) {
                    fwrite(buf, sizeof(float), HEAD_DIM, fp);
                } else {
                    /* Summary stats */
                    float minv = buf[0], maxv = buf[0], sum = 0.0f, sumabs = 0.0f;
                    for (size_t d = 0; d < HEAD_DIM; d++) {
                        float v = buf[d];
                        if (v < minv) minv = v;
                        if (v > maxv) maxv = v;
                        sum += v;
                        sumabs += v >= 0 ? v : -v;
                    }
                    float mean = sum / (float)HEAD_DIM;
                    float absmean = sumabs / (float)HEAD_DIM;
                    float var = 0.0f;
                    for (size_t d = 0; d < HEAD_DIM; d++) { float dv = buf[d] - mean; var += dv*dv; }
                    float stddev = sqrtf(var / (float)HEAD_DIM);
                    printf("%u,%u,%.4f,%.4f,%.4f,%.4f,%.4f,",
                           L, P, minv, maxv, mean, absmean, stddev);
                    for (int d = 0; d < 8; d++) printf("%.4f%s", buf[d], d<7 ? "," : "\n");
                }
            }
        }
        free(buf);
        if (fp) {
            fclose(fp);
            fprintf(stderr, "[logitlens] wrote kv_raw binary to %s\n", dump_kv_out_path);
        }
    }

    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    free(text);
    ds4_engine_close(engine);
    return 0;
}
