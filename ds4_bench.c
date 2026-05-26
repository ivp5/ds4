#include "ds4.h"
/* silv 2026-05-26: HOT dispatch diagnostics. ds4_bench.c sees DS4_N_LAYER /
 * DS4_N_EXPERT as the header's #defines (no enum in scope here). */
#include "ds4_expert_table.h"

/* Purpose-built throughput benchmark.
 *
 * The benchmark walks one fixed token sequence to configurable context
 * frontiers, measuring only the newest prefill interval at each frontier.  It
 * then snapshots the live session in memory, performs a fixed greedy decode
 * run without allowing EOS, restores the snapshot, and continues to the next
 * frontier.  Snapshot save/restore time is intentionally outside both timing
 * windows.
 */

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    const char *model_path;
    const char *prompt_path;
    const char *chat_prompt_path;
    const char *system;
    const char *csv_path;
    ds4_backend backend;
    int threads;
    int ctx_start;
    int ctx_max;
    int ctx_alloc;
    int step_incr;
    int gen_tokens;
    double step_mul;
    ds4_mpp_mode mpp_mode;
    const char *dump_frontier_logits_dir;
    bool warm_weights;
    bool quality;
    /* M1 STICKY HAZARD safety flags (CLAUDE.md doctrine).
     * cpu_moe=true runs ALL routed MoE on CPU (Metal fits).
     * prefill_metal_phases=-1 means auto (resolve from iogpu.wired_limit_mb);
     * prefill_metal_phases=N means explicit phase count.
     * One of these MUST be set on M1 for DS4-Flash (86.7 GB IQ2_XXS).
     * Default 0 = unsafe path — left for backward-compat with non-DS4 models. */
    bool cpu_moe;
    int n_cpu_moe_layers;
    int prefill_metal_phases;
    /* silv 2026-05-26: VQB2 hot-store load at engine init.
     * Per codex H1884: candidate manifest is the runtime contract,
     * not directory traversal (H1883 alias hazard). */
    const char *vqb2_manifest;
    const char *vqb2_candidate;
    uint64_t vqb2_budget_mb;   /* hot-store budget in MB, 0 = default 32 GB */
} bench_config;

static double bench_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: ds4-bench --prompt-file FILE [options]\n"
        "\n"
        "Benchmarks instantaneous prefill and generation throughput at context\n"
        "frontiers such as 2048, 4096, 6144, ... . Generation is always greedy,\n"
        "runs for exactly --gen-tokens tokens, and skips EOS so every row is\n"
        "comparable.\n"
        "\n"
        "Input:\n"
        "  --prompt-file FILE\n"
        "      Raw benchmark text. The fixed token sequence is sliced at each frontier.\n"
        "  --chat-prompt-file FILE\n"
        "      Render FILE as one no-thinking chat user message, then slice that sequence.\n"
        "  -sys, --system TEXT\n"
        "      System prompt used only with --chat-prompt-file.\n"
        "\n"
        "Model and backend:\n"
        "  -m, --model FILE       GGUF model path. Default: ds4flash.gguf\n"
        "  --metal | --cuda | --cpu | --backend NAME\n"
        "      Select backend explicitly. Defaults to Metal on macOS, CUDA elsewhere.\n"
        "  -t, --threads N        CPU helper threads.\n"
        "  --quality              Prefer exact kernels where applicable.\n"
        "  -mt MODE, --mt MODE    Metal Tensor route mode: auto, on, or off.\n"
        "      Legacy alias: --mpp MODE.\n"
        "  --warm-weights         Touch mapped tensor pages before benchmarking.\n"
        "\n"
        "M1 SAFETY (mandatory for DS4-Flash 86.7 GB IQ2_XXS on Metal):\n"
        "  --prefill-metal-phases auto|N\n"
        "      Phase-split Metal residency so weights don't exceed wired_limit_mb.\n"
        "      'auto' resolves N from iogpu.wired_limit_mb. MANDATORY on M1 for\n"
        "      DS4-Flash unless --cpu-moe is set.\n"
        "  --cpu-moe              Run ALL routed MoE on CPU (Metal fits without).\n"
        "  --n-cpu-moe N          Run routed MoE for N layers on CPU.\n"
        "      Mutually exclusive with --prefill-metal-phases.\n"
        "\n"
        "Sweep:\n"
        "  --ctx-start N          First measured frontier. Default: 2048\n"
        "  --ctx-max N            Last measured frontier. Default: 32768\n"
        "  --ctx-alloc N          Allocated context. Default: ctx-max + gen-tokens + 1\n"
        "  --step-mul F           Multiplicative step. Default: 1\n"
        "  --step-incr N          Linear step when --step-mul is 1. Default: 2048\n"
        "  --gen-tokens N         Greedy decode tokens per frontier. Default: 128\n"
        "\n"
        "Output:\n"
        "  --csv FILE             Write CSV there instead of stdout.\n"
        "  --dump-frontier-logits-dir DIR\n"
        "      Write one full-logit JSON file per measured frontier. DIR must exist.\n"
        "  -h, --help             Show this help.\n");
}

static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static double parse_double_arg(const char *s, const char *opt) {
    char *end = NULL;
    double v = strtod(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v)) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-bench: %s requires an argument\n", opt);
        exit(2);
    }
    return argv[++*i];
}

static ds4_backend parse_backend(const char *s, const char *opt) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
    fprintf(stderr, "ds4-bench: valid backends are: metal, cuda, cpu\n");
    exit(2);
}

static ds4_mpp_mode parse_mpp_mode(const char *s, const char *opt) {
    if (!strcmp(s, "auto")) return DS4_MPP_AUTO;
    if (!strcmp(s, "on")) return DS4_MPP_ON;
    if (!strcmp(s, "off")) return DS4_MPP_OFF;
    fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
    fprintf(stderr, "ds4-bench: valid Metal Tensor modes are: auto, on, off\n");
    exit(2);
}

static ds4_backend default_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4-bench: failed to open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "ds4-bench: failed to seek %s\n", path);
        fclose(fp);
        exit(1);
    }
    long n = ftell(fp);
    if (n < 0) {
        fprintf(stderr, "ds4-bench: failed to tell %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "ds4-bench: failed to rewind %s\n", path);
        fclose(fp);
        exit(1);
    }
    char *buf = malloc((size_t)n + 1);
    if (!buf) {
        fprintf(stderr, "ds4-bench: out of memory reading %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        fprintf(stderr, "ds4-bench: failed to read %s\n", path);
        free(buf);
        fclose(fp);
        exit(1);
    }
    fclose(fp);
    buf[n] = '\0';
    return buf;
}

static bench_config parse_options(int argc, char **argv) {
    bench_config c = {
        .model_path = "ds4flash.gguf",
        .system = "You are a helpful assistant.",
        .backend = default_backend(),
        .ctx_start = 2048,
        .ctx_max = 32768,
        .step_incr = 2048,
        .gen_tokens = 128,
        .step_mul = 1.0,
        .mpp_mode = DS4_MPP_AUTO,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            c.prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--chat-prompt-file")) {
            c.chat_prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-sys") || !strcmp(arg, "--system")) {
            c.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--ctx-start")) {
            c.ctx_start = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--ctx-max")) {
            c.ctx_max = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--ctx-alloc")) {
            c.ctx_alloc = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--step-incr")) {
            c.step_incr = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--step-mul")) {
            c.step_mul = parse_double_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--gen-tokens") || !strcmp(arg, "--tokens") || !strcmp(arg, "-n")) {
            c.gen_tokens = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--csv")) {
            c.csv_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dump-frontier-logits-dir")) {
            c.dump_frontier_logits_dir = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--backend")) {
            c.backend = parse_backend(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--metal")) {
            c.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--cuda")) {
            c.backend = DS4_BACKEND_CUDA;
        } else if (!strcmp(arg, "--cpu")) {
            c.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "--cpu-moe")) {
            /* CLAUDE.md DS4 STICKY HAZARD safety flag — run ALL routed MoE on CPU. */
            c.cpu_moe = true;
        } else if (!strcmp(arg, "--n-cpu-moe")) {
            c.n_cpu_moe_layers = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--prefill-metal-phases")) {
            /* CLAUDE.md DS4 STICKY HAZARD safety flag — phase-split Metal residency.
             * "auto" resolves N from iogpu.wired_limit_mb at runtime.
             * MANDATORY on M1 for DS4-Flash 86.7 GB IQ2_XXS (panics 2026-05-19,
             * 2026-05-23 were both from launches lacking this flag). */
            const char *s = need_arg(&i, argc, argv, arg);
            if (!strcmp(s, "auto")) {
                c.prefill_metal_phases = -1;
            } else {
                c.prefill_metal_phases = parse_int(s, arg);
            }
        } else if (!strcmp(arg, "--quality")) {
            c.quality = true;
        } else if (!strcmp(arg, "-mt") || !strcmp(arg, "--mt") || !strcmp(arg, "--mpp")) {
            c.mpp_mode = parse_mpp_mode(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--warm-weights")) {
            c.warm_weights = true;
        } else if (!strcmp(arg, "--vqb2-manifest")) {
            /* silv 2026-05-26 / codex H1884: load VQB2 hot-store from a
             * candidate-keyed manifest CSV. Required with --vqb2-candidate. */
            c.vqb2_manifest = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--vqb2-candidate")) {
            c.vqb2_candidate = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--vqb2-budget-mb")) {
            c.vqb2_budget_mb = (uint64_t)parse_int(need_arg(&i, argc, argv, arg), arg);
        } else {
            fprintf(stderr, "ds4-bench: unknown option: %s\n", arg);
            usage(stderr);
            exit(2);
        }
    }

    if (!!c.prompt_path == !!c.chat_prompt_path) {
        fprintf(stderr, "ds4-bench: specify exactly one of --prompt-file or --chat-prompt-file\n");
        exit(2);
    }
    if (c.ctx_start > c.ctx_max) {
        fprintf(stderr, "ds4-bench: --ctx-start must be <= --ctx-max\n");
        exit(2);
    }
    if (c.step_mul < 1.0) {
        fprintf(stderr, "ds4-bench: --step-mul must be >= 1\n");
        exit(2);
    }
    if (c.step_mul == 1.0 && c.step_incr <= 0) {
        fprintf(stderr, "ds4-bench: --step-incr must be positive when --step-mul is 1\n");
        exit(2);
    }
    if (c.ctx_max > INT_MAX - c.gen_tokens - 1) {
        fprintf(stderr, "ds4-bench: requested context is too large\n");
        exit(2);
    }
    if (c.ctx_alloc == 0) c.ctx_alloc = c.ctx_max + c.gen_tokens + 1;
    if (c.ctx_alloc <= c.ctx_max + c.gen_tokens) {
        fprintf(stderr, "ds4-bench: --ctx-alloc must be greater than ctx-max + gen-tokens\n");
        exit(2);
    }
    return c;
}

static void json_write_string(FILE *fp, const char *s) {
    fputc('"', fp);
    if (s) {
        for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
            switch (*p) {
            case '"':  fputs("\\\"", fp); break;
            case '\\': fputs("\\\\", fp); break;
            case '\b': fputs("\\b", fp); break;
            case '\f': fputs("\\f", fp); break;
            case '\n': fputs("\\n", fp); break;
            case '\r': fputs("\\r", fp); break;
            case '\t': fputs("\\t", fp); break;
            default:
                if (*p < 0x20) fprintf(fp, "\\u%04x", (unsigned)*p);
                else fputc((char)*p, fp);
                break;
            }
        }
    }
    fputc('"', fp);
}

static int write_frontier_logits_json(
        const bench_config *cfg,
        ds4_engine         *engine,
        ds4_session        *session,
        int                 frontier,
        int                 previous) {
    if (!cfg->dump_frontier_logits_dir) return 0;

    const int vocab = ds4_engine_vocab_size(engine);
    float *logits = malloc((size_t)vocab * sizeof(logits[0]));
    if (!logits) {
        fprintf(stderr, "ds4-bench: out of memory copying frontier logits\n");
        return 1;
    }
    if (ds4_session_copy_logits(session, logits, vocab) != vocab) {
        fprintf(stderr, "ds4-bench: failed to copy frontier logits at %d\n", frontier);
        free(logits);
        return 1;
    }

    char path[PATH_MAX];
    const int n = snprintf(path,
                           sizeof(path),
                           "%s/frontier_%06d.logits.json",
                           cfg->dump_frontier_logits_dir,
                           frontier);
    if (n <= 0 || (size_t)n >= sizeof(path)) {
        fprintf(stderr, "ds4-bench: frontier logits path is too long\n");
        free(logits);
        return 1;
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4-bench: failed to open %s: %s\n", path, strerror(errno));
        free(logits);
        return 1;
    }

    const int argmax = ds4_session_argmax(session);
    fprintf(fp, "{\n  \"source\":\"ds4-bench\",\n  \"model\":");
    json_write_string(fp, cfg->model_path);
    fprintf(fp,
            ",\n  \"backend\":\"%s\",\n  \"mt\":\"%s\",\n  \"quality\":%s,\n"
            "  \"quant_bits\":%d,\n  \"prompt_tokens\":%d,\n"
            "  \"frontier_tokens\":%d,\n  \"prefill_tokens\":%d,\n"
            "  \"ctx\":%d,\n  \"vocab\":%d,\n"
            "  \"argmax_id\":%d,\n  \"argmax_logit\":%.9g,\n  \"logits\":[",
            ds4_backend_name(cfg->backend),
            ds4_mpp_mode_name(cfg->mpp_mode),
            cfg->quality ? "true" : "false",
            ds4_engine_routed_quant_bits(engine),
            frontier,
            frontier,
            frontier - previous,
            cfg->ctx_alloc,
            vocab,
            argmax,
            logits[argmax]);
    for (int i = 0; i < vocab; i++) {
        if (i) fputc(',', fp);
        if ((i % 8) == 0) fputs("\n    ", fp);
        if (isfinite(logits[i])) fprintf(fp, "%.9g", logits[i]);
        else fputs("null", fp);
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4-bench: failed to close %s\n", path);
        free(logits);
        return 1;
    }
    free(logits);
    return 0;
}

static int next_frontier(const bench_config *c, int cur) {
    if (cur >= c->ctx_max) return c->ctx_max;
    int next;
    if (c->step_mul == 1.0) {
        if (cur > INT_MAX - c->step_incr) next = c->ctx_max;
        else next = cur + c->step_incr;
    } else {
        const double v = ceil((double)cur * c->step_mul);
        next = v > (double)INT_MAX ? c->ctx_max : (int)v;
        if (next <= cur) next = cur + 1;
    }
    if (next > c->ctx_max) next = c->ctx_max;
    return next;
}

static void log_context_memory(ds4_backend backend, int ctx_size) {
    ds4_context_memory m = ds4_context_memory_estimate(backend, ctx_size);
    fprintf(stderr,
            "ds4-bench: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            ctx_size,
            ds4_backend_name(backend),
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap);
}

int main(int argc, char **argv) {
    bench_config cfg = parse_options(argc, argv);

    ds4_engine_options opt = {
        .model_path = cfg.model_path,
        .backend = cfg.backend,
        .n_threads = cfg.threads,
        .warm_weights = cfg.warm_weights,
        .quality = cfg.quality,
        .mpp_mode = cfg.mpp_mode,
        /* M1 STICKY HAZARD safety: pass parsed safety flags through to engine.
         * Default 0 / false = backward-compat with non-DS4 models on non-M1.
         * On M1 DS4-Flash launches MUST set one of these or expect kernel
         * panic from exceeding 48-60GB Metal wired cap with 86.7GB model. */
        .cpu_moe = cfg.cpu_moe,
        .n_cpu_moe_layers = cfg.n_cpu_moe_layers,
        .prefill_metal_phases = cfg.prefill_metal_phases,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) return 1;
    log_context_memory(cfg.backend, cfg.ctx_alloc);

    /* silv 2026-05-26: VQB2 candidate-keyed hot-store load.
     * Per codex H1884, runtime contract is `(candidate, layer, kind, k)` —
     * NOT directory traversal (H1883 alias hazard). Loaded into the global
     * hot-store; subsequent ds4.c dispatch can check via ds4_hot_get_*.
     * The store is currently INERT — wire to dispatch path lives in a
     * separate change. This flag just validates load-time correctness. */
    static ds4_hot_expert_store *g_bench_hot_store = NULL;
    if (cfg.vqb2_manifest && cfg.vqb2_candidate) {
        const uint64_t budget = cfg.vqb2_budget_mb > 0
            ? cfg.vqb2_budget_mb * (1ull << 20)
            : (32ull << 30); /* 32 GB default for full corpus FP16 */
        fprintf(stderr,
            "ds4-bench: VQB2 hot-store init — manifest=%s candidate=%s budget=%llu MB\n",
            cfg.vqb2_manifest, cfg.vqb2_candidate,
            (unsigned long long)(budget >> 20));
        g_bench_hot_store = ds4_hot_expert_store_alloc(budget);
        if (!g_bench_hot_store) {
            fprintf(stderr, "ds4-bench: hot-store alloc failed\n");
            ds4_engine_close(engine);
            return 1;
        }
        const double t0 = bench_now_sec();
        const int pinned = ds4_vqb2_candidate_manifest_load(
            g_bench_hot_store, cfg.vqb2_manifest, cfg.vqb2_candidate);
        const double t1 = bench_now_sec();
        if (pinned < 0) {
            fprintf(stderr, "ds4-bench: VQB2 candidate manifest load failed\n");
            ds4_hot_expert_store_free(g_bench_hot_store);
            ds4_engine_close(engine);
            return 1;
        }
        fprintf(stderr, "ds4-bench: VQB2 hot-store loaded %d tiles in %.2fs\n",
                pinned, t1 - t0);
        ds4_hot_expert_store_print(g_bench_hot_store);
        /* silv 2026-05-26: publish as active so ds4.c dispatch can check
         * via ds4_hot_store_get_active() at the MoE call site. */
        ds4_hot_store_set_active(g_bench_hot_store);
        /* silv 2026-05-26: bind FP16 heap as MTLBuffer via
         * newBufferWithBytesNoCopy (zero-copy on Apple Silicon unified
         * memory). Required before Metal kernel dispatch lands.
         * Currently CPU-path dispatch is the implementation; the
         * binding is a no-cost prep for the Metal kernel work. */
        extern int ds4_metal_vqb2_fp16_init(void);
        extern int ds4_metal_vqb2_fp16_bind_store(struct ds4_hot_expert_store *);
        if (cfg.backend != DS4_BACKEND_CPU) {
            (void)ds4_metal_vqb2_fp16_init();
            (void)ds4_metal_vqb2_fp16_bind_store(g_bench_hot_store);
        }
    }

    char *text = read_file(cfg.prompt_path ? cfg.prompt_path : cfg.chat_prompt_path);
    ds4_tokens prompt = {0};
    if (cfg.chat_prompt_path) {
        ds4_encode_chat_prompt(engine, cfg.system, text, DS4_THINK_NONE, &prompt);
    } else {
        ds4_tokenize_text(engine, text, &prompt);
    }
    free(text);

    if (prompt.len < cfg.ctx_max) {
        fprintf(stderr,
                "ds4-bench: prompt has %d tokens, need at least --ctx-max=%d\n",
                prompt.len,
                cfg.ctx_max);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg.ctx_alloc) != 0) {
        fprintf(stderr, "ds4-bench: failed to create session\n");
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }

    FILE *out = stdout;
    if (cfg.csv_path) {
        out = fopen(cfg.csv_path, "wb");
        if (!out) {
            fprintf(stderr, "ds4-bench: failed to open %s: %s\n", cfg.csv_path, strerror(errno));
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            ds4_engine_close(engine);
            return 1;
        }
    }
    fprintf(out, "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes\n");
    fflush(out);

    const int eos = ds4_token_eos(engine);
    ds4_session_snapshot snap = {0};
    char err[256];
    int previous = 0;
    int rc = 0;

    for (int frontier = cfg.ctx_start; ; frontier = next_frontier(&cfg, frontier)) {
        ds4_tokens prefix = {
            .v = prompt.v,
            .len = frontier,
            .cap = frontier,
        };

        const double prefill_t0 = bench_now_sec();
        if (ds4_session_sync(session, &prefix, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4-bench: prefill to %d failed: %s\n", frontier, err);
            rc = 1;
            break;
        }
        const double prefill_t1 = bench_now_sec();
        const double prefill_sec = prefill_t1 - prefill_t0;
        const int prefill_tokens = frontier - previous;

        if (write_frontier_logits_json(&cfg, engine, session, frontier, previous) != 0) {
            rc = 1;
            break;
        }

        if (ds4_session_save_snapshot(session, &snap, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4-bench: snapshot at %d failed: %s\n", frontier, err);
            rc = 1;
            break;
        }

        const double gen_t0 = bench_now_sec();
        /* Streaming progress per silv shift #295: no silent long-running.
         * Emit inst+avg t/s at the smaller of {16 tokens, 2 seconds} cadence.
         * Disabled by setting DS4_BENCH_QUIET=1 (e.g. for CSV-only piped output). */
        const int stream_quiet = getenv("DS4_BENCH_QUIET") != NULL;
        const int stream_token_step = 16;
        const double stream_time_step = 2.0;
        double stream_last_t = gen_t0;
        int    stream_last_i = 0;
        if (!stream_quiet) {
            fprintf(stderr, "ds4-bench: gen-start frontier=%d target=%d\n",
                    frontier, cfg.gen_tokens);
            fflush(stderr);
        }
        for (int i = 0; i < cfg.gen_tokens; i++) {
            if (ds4_session_pos(session) + 1 >= ds4_session_ctx(session)) {
                fprintf(stderr, "ds4-bench: generation would exceed allocated context at frontier %d\n", frontier);
                rc = 1;
                break;
            }
            const int token = ds4_session_argmax_excluding(session, eos);
            if (token < 0) {
                fprintf(stderr, "ds4-bench: failed to choose non-EOS token at frontier %d\n", frontier);
                rc = 1;
                break;
            }
            if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: decode at frontier %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
            /* Stream cadence — fixed-work-unit OR fixed-time, whichever fires first */
            const int generated = i + 1;
            const double now = bench_now_sec();
            const int    token_due = (generated - stream_last_i) >= stream_token_step;
            const int    time_due  = (now - stream_last_t) >= stream_time_step;
            if (!stream_quiet && (token_due || time_due) && generated < cfg.gen_tokens) {
                const double inst_tps = (generated - stream_last_i) /
                                        ((now - stream_last_t) > 1e-9 ? (now - stream_last_t) : 1.0);
                const double avg_tps  = generated / ((now - gen_t0) > 1e-9 ? (now - gen_t0) : 1.0);
                const double pct      = 100.0 * (double)generated / (double)cfg.gen_tokens;
                fprintf(stderr,
                        "ds4-bench:   gen %4d/%d (%.1f%%) inst=%5.2f t/s avg=%5.2f t/s elapsed=%5.1fs\n",
                        generated, cfg.gen_tokens, pct, inst_tps, avg_tps, now - gen_t0);
                fflush(stderr);
                stream_last_t = now;
                stream_last_i = generated;
            }
        }
        const double gen_t1 = bench_now_sec();
        if (!stream_quiet && rc == 0) {
            const double total_sec = gen_t1 - gen_t0;
            const double final_tps = cfg.gen_tokens / (total_sec > 1e-9 ? total_sec : 1.0);
            fprintf(stderr,
                    "ds4-bench: gen-end   frontier=%d generated=%d wall=%.2fs avg=%.2f t/s\n",
                    frontier, cfg.gen_tokens, total_sec, final_tps);
            /* silv 2026-05-26: HOT dispatch diagnostic. Reports per-tier
             * routing distribution so future hot-expert pre-dequant work
             * can size its budget against observed dispatch density. */
            ds4_hot_print_stats();
            fflush(stderr);
        }
        if (rc != 0) break;

        if (ds4_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4-bench: restore at %d failed: %s\n", frontier, err);
            rc = 1;
            break;
        }

        const double gen_sec = gen_t1 - gen_t0;
        fprintf(out,
                "%d,%d,%.2f,%d,%.2f,%llu\n",
                frontier,
                prefill_tokens,
                prefill_sec > 0.0 ? (double)prefill_tokens / prefill_sec : 0.0,
                cfg.gen_tokens,
                gen_sec > 0.0 ? (double)cfg.gen_tokens / gen_sec : 0.0,
                (unsigned long long)snap.len);
        fflush(out);

        previous = frontier;
        if (frontier >= cfg.ctx_max) break;
    }

    if (out != stdout) fclose(out);
    ds4_session_snapshot_free(&snap);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return rc;
}
