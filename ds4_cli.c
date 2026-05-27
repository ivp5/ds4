#include "ds4.h"
#include "ds4_gpu.h"
#include "ds4_polar_reader.h"
#include "linenoise.h"

/* ds4 CLI.
 *
 * One-shot mode builds a single DeepSeek chat prompt and exits.  Interactive
 * mode keeps a rendered token transcript plus one ds4_session, so follow-up
 * turns reuse the live Metal KV checkpoint just like the server does.  The CLI
 * deliberately keeps policy here and leaves graph/cache mechanics inside the
 * engine API. */

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    const char *prompt;
    const char *system;
    int n_predict;
    int ctx_size;
    float temperature;
    float top_p;
    float min_p;
    uint64_t seed;
    bool dump_tokens;
    const char *dump_logits_path;
    const char *dump_logprobs_path;
    int dump_logprobs_top_k;
    const char *perplexity_file_path;
    const char *imatrix_dataset_path;
    const char *imatrix_output_path;
    int imatrix_max_prompts;
    int imatrix_max_tokens;
    ds4_think_mode think_mode;
    bool head_test;
    bool first_token_test;
    bool metal_graph_test;
    bool metal_graph_full_test;
    bool metal_graph_prompt_test;
} cli_generation_options;

typedef struct {
    ds4_engine_options engine;
    cli_generation_options gen;
    char *prompt_owned;
    bool inspect;
} cli_config;

static volatile sig_atomic_t cli_interrupted;

static void cli_sigint_handler(int sig) {
    (void)sig;
    cli_interrupted = 1;
}

static bool cli_interrupt_requested(void) {
    return cli_interrupted != 0;
}

static void cli_interrupt_clear(void) {
    cli_interrupted = 0;
}

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: ds4 [(-p PROMPT | --prompt-file FILE)] [options]\n"
        "\n"
        "Invocation modes:\n"
        "  ds4\n"
        "      Start the interactive chat prompt with a session backend: ds4>\n"
        "  ds4 -p TEXT\n"
        "      Run one prompt and exit.\n"
        "  ds4 --prompt-file FILE\n"
        "      Run one prompt read from FILE and exit. Useful for long prompts.\n"
        "\n"
        "Model and runtime:\n"
        "  -m, --model FILE\n"
        "      GGUF model path. Default: ds4flash.gguf\n"
        "  --mtp FILE\n"
        "      Optional MTP support GGUF used for draft-token probes.\n"
        "  --mtp-draft N\n"
        "      Maximum autoregressive MTP draft tokens per speculative step. Default: 1\n"
        "  --mtp-margin F\n"
        "      Minimum recursive-draft confidence for the fast N=2 verifier. Default: 3\n"
        "  -c, --ctx N\n"
        "      Context size allocated for the session. Default: 32768\n"
        "  --metal\n"
        "      Use the Metal graph backend. This is the normal fast path on macOS.\n"
        "  --cuda\n"
        "      Use the CUDA graph backend. This is the normal fast path on CUDA builds.\n"
        "  --cpu\n"
        "      Use the CPU reference/debug backend. Not recommended for normal inference.\n"
        "  --backend NAME\n"
        "      Select backend explicitly: metal, cuda, or cpu.\n"
        "  -t, --threads N\n"
        "      CPU helper threads for host-side or reference work.\n"
        "  --quality\n"
        "      Prefer exact kernels where faster approximate paths exist; MTP uses strict verification.\n"
        "  --dir-steering-file FILE\n"
        "      Load one f32 direction vector per layer for directional steering.\n"
        "  --dir-steering-ffn F\n"
        "      Apply steering after FFN outputs: y -= F*v*dot(v,y). Default with file: 1\n"
        "  --dir-steering-attn F\n"
        "      Apply steering after attention outputs. Default: 0\n"
        "  --warm-weights\n"
        "      Touch mapped tensor pages before generation. Slower startup, fewer first-use stalls.\n"
        "  --cpu-moe\n"
        "      Compute routed MoE experts on the CPU and keep their weights out of the Metal\n"
        "      residency set. Lets very large GGUFs (e.g. q4 on 128 GB) run by leaving routed\n"
        "      expert pages to the OS page cache. Metal backend only.\n"
        "      Equivalent to --n-cpu-moe with all layers on CPU.\n"
        "  --n-cpu-moe N\n"
        "      Compute the routed MoE on the CPU only for the first N layers; the\n"
        "      remaining layers stay on the GPU. Matches llama.cpp's --n-cpu-moe\n"
        "      semantics. Tune N to trade VRAM for speed: N=10..20 is a typical\n"
        "      sweet spot for q4 on 128 GB. N=0 disables CPU MoE entirely.\n"
        "  --mtl4-moe\n"
        "      Enable MTL4 ML packed-MoE path (group=6 for DS4 active-experts).\n"
        "      Env-gated; preflight requires n_expert==6 + n_tokens≤16; falls\n"
        "      back to legacy SIMD MoE when conditions not met. SCAFFOLD: full\n"
        "      packed dispatch + IQ2_XXS dequant pending (see ds4_metal.m).\n"
        "  --prefill-metal-phases auto|N\n"
        "      Run prefill on Metal in N evenly-split phases, swapping the routed\n"
        "      expert residency between phases. Generation falls back to cpu-moe.\n"
        "      \"auto\" sizes N from sysctl iogpu.wired_limit_mb (bounded by\n"
        "      hw.memsize) so each phase fits the Metal wired-memory cap.\n"
        "      Mutually exclusive with --cpu-moe / --n-cpu-moe. Metal backend only.\n"
        "      Env overrides: DS4_PREFILL_METAL_PHASES_WIRED_LIMIT_MIB,\n"
        "      DS4_PREFILL_METAL_PHASES_HEADROOM_MIB (default 14336),\n"
        "      DS4_PREFILL_METAL_PHASES_MIN_TOKENS (default 0; set e.g. 1500\n"
        "      for single-shot chat to fall back to cpu-moe on short prompts).\n"
        "  --power N\n"
        "      Target GPU duty cycle percentage, 1..100. Default: 100\n"
        "\n"
        "Prompt and generation:\n"
        "  -p, --prompt TEXT\n"
        "      Prompt to generate from.\n"
        "  --prompt-file FILE\n"
        "      Read the prompt text from FILE.\n"
        "  -sys, --system TEXT\n"
        "      System prompt. Empty string disables the default. Default: You are a helpful assistant\n"
        "  -n, --tokens N\n"
        "      Maximum tokens to generate. Default: 50000\n"
        "  --temp F\n"
        "      Sampling temperature. 0 is greedy/deterministic. Default: 1\n"
        "  --top-p F\n"
        "      Nucleus sampling probability. Default: 1\n"
        "  --min-p F\n"
        "      Keep tokens scoring at least F times the top token. Default: 0.05\n"
        "  --seed N\n"
        "      Sampling seed for reproducible non-greedy runs. Default: time-based\n"
        "  --think\n"
        "      Use normal thinking mode. This is the default.\n"
        "  --think-max\n"
        "      Use Think Max when --ctx is at least 393216 tokens; otherwise normal thinking.\n"
        "  --nothink\n"
        "      Start assistant turns with </think> for direct non-thinking replies.\n"
        "\n"
        "Interactive commands:\n"
        "  /help\n"
        "      Show interactive commands.\n"
        "  /think, /think-max, /nothink\n"
        "      Select normal thinking, context-gated Think Max, or non-thinking mode.\n"
        "  /ctx N\n"
        "      Recreate the interactive session with a new context size.\n"
        "  /power N\n"
        "      Set GPU duty cycle percentage, 1..100.\n"
        "  /read FILE\n"
        "      Read a prompt from FILE and run it as the next user message.\n"
        "  /quit, /exit\n"
        "      Leave the interactive prompt.\n"
        "  Ctrl+C\n"
        "      Stop the current generation and return to ds4> without exiting.\n"
        "\n"
        "Diagnostics:\n"
        "  --inspect\n"
        "      Load the model and print a summary only.\n"
        "  --dump-tokens\n"
        "      Tokenize -p/--prompt-file exactly as written, then exit without inference.\n"
        "  --dump-logits FILE\n"
        "      Write full next-token logits as JSON after prompt prefill, then exit.\n"
        "  --dump-logprobs FILE\n"
        "      Write greedy continuation top-logprobs as JSON without printing text.\n"
        "  --logprobs-top-k N\n"
        "      Number of local alternatives stored by --dump-logprobs. Default: 20\n"
        "  --perplexity-file FILE\n"
        "      Score raw text with teacher-forced next-token negative log likelihood.\n"
        "  --imatrix-dataset FILE\n"
        "      Rendered DS4 prompt dataset produced by misc/imatrix_dataset.\n"
        "  --imatrix-out FILE\n"
        "      Collect a routed-MoE activation imatrix and write llama-compatible .dat.\n"
        "  --imatrix-max-prompts N\n"
        "      Stop imatrix collection after N prompts. Default: no prompt limit\n"
        "  --imatrix-max-tokens N\n"
        "      Stop imatrix collection after N prompt tokens. Default: no token limit\n"
        "  --head-test\n"
        "      Run the output HC/logits head after the native slice.\n"
        "  --first-token-test\n"
        "      Run an exact CPU whole-model pass for the first prompt token.\n"
        "  --metal-graph-test\n"
        "      Compare first GPU-resident graph stages with CPU.\n"
        "  --metal-graph-full-test\n"
        "      Run the GPU-resident self-token graph across all layers.\n"
        "  --metal-graph-prompt-test\n"
        "      Compare CPU and GPU graph logits for the full prompt.\n"
        "\n"
        "Normal CLI commands:\n"
        "  ./ds4\n"
        "  ./ds4 -p \"Scrivi una storia su una papera scansafatiche\"\n"
        "  ./ds4 --think-max --prompt-file prompt.txt --ctx 393216\n"
        "\n"
        "Notes:\n"
        "  The CLI keeps KV cache state across interactive turns on session backends.\n"
        "  CPU mode supports interactive chat too, but it is a slow reference/debug path.\n"
        "  Long added input is processed with batched prefill; short continuations use decode.\n"
        "  Startup prints the extra context-buffer memory for the selected context size.\n"
        "\n"
        "  -h, --help\n"
        "      Show this help.\n");
}

static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT32_MAX) {
        fprintf(stderr, "ds4: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static uint64_t parse_u64(const char *s, const char *opt) {
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v == 0) {
        fprintf(stderr, "ds4: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (uint64_t)v;
}

static float parse_float_range(const char *s, const char *opt, float min, float max) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v) || v < min || v > max) {
        fprintf(stderr, "ds4: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static ds4_backend parse_backend(const char *s) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4: invalid backend: %s\n", s);
    fprintf(stderr, "ds4: valid backends are: metal, cuda, cpu\n");
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

static void log_context_memory(ds4_backend backend, int ctx_size) {
    ds4_context_memory m = ds4_context_memory_estimate(backend, ctx_size);
    fprintf(stderr,
            "ds4: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            ctx_size,
            ds4_backend_name(backend),
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap);
}

static ds4_think_mode cli_effective_think_mode(const cli_generation_options *gen) {
    return ds4_think_mode_for_context(gen->think_mode, gen->ctx_size);
}

static bool cli_think_max_downgraded(const cli_generation_options *gen) {
    return gen->think_mode == DS4_THINK_MAX &&
           cli_effective_think_mode(gen) != DS4_THINK_MAX;
}

static void cli_warn_think_max_downgraded(const cli_generation_options *gen, const char *name) {
    if (!cli_think_max_downgraded(gen)) return;
    ds4_log(stderr,
        DS4_LOG_WARNING,
        "ds4: warning: %s needs --ctx >= %u; ctx=%d uses normal thinking instead\n",
        name,
        ds4_think_max_min_context(),
        gen->ctx_size);
}

static double cli_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static char *read_prompt_file(const char *path, bool fatal);

typedef struct {
    int base_tokens;
    int input_tokens;
    bool use_color;
} cli_prefill_progress;

static void cli_prefill_progress_cb(void *ud, const char *event, int current, int total) {
    (void)total;
    cli_prefill_progress *p = ud;
    if (!p || !event || p->input_tokens <= 0) return;
    const bool is_display = strcmp(event, "prefill_display") == 0;
    if (strcmp(event, "prefill_chunk") && !is_display) return;
    if (is_display && !p->use_color) return;

    int processed = current - p->base_tokens;
    if (processed < 0) processed = 0;
    if (processed > p->input_tokens) processed = p->input_tokens;
    double pct = 100.0 * (double)processed / (double)p->input_tokens;
    if (pct > 100.0) pct = 100.0;

    if (p->use_color) {
        fputc('\r', stderr);
        ds4_log(stderr,
                DS4_LOG_PREFILL,
                "processing %d input tokens: %d/%d (%.1f%%)",
                p->input_tokens,
                processed,
                p->input_tokens,
                pct);
        fputs("\x1b[K", stderr);
        if (processed >= p->input_tokens) fputc('\n', stderr);
    } else {
        fprintf(stderr,
                "processing %d input tokens: %d/%d (%.1f%%)\n",
                p->input_tokens,
                processed,
                p->input_tokens,
                pct);
    }
    fflush(stderr);
}

static bool is_rendered_chat_prompt(const char *prompt) {
    const char *bos = "<｜begin▁of▁sentence｜>";
    return prompt && strncmp(prompt, bos, strlen(bos)) == 0;
}

typedef struct {
    ds4_engine *engine;
    FILE *fp;
    bool format_thinking;
    bool in_think;
    bool color_open;
    bool use_color;
    bool last_output_newline;
    char pending[16];
    size_t pending_len;
} token_printer;

static bool bytes_has_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n >= plen && memcmp(p, prefix, plen) == 0;
}

static bool bytes_is_partial_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n < plen && memcmp(prefix, p, n) == 0;
}

static void token_printer_set_grey(token_printer *p) {
    if (p->use_color && !p->color_open) {
        fputs("\x1b[90m", p->fp);
        p->color_open = true;
    }
}

static void token_printer_reset_color(token_printer *p) {
    if (p->use_color && p->color_open) {
        fputs("\x1b[0m", p->fp);
        p->color_open = false;
    }
}

static void token_printer_write_char(token_printer *p, char c) {
    if (p->in_think) token_printer_set_grey(p);
    fputc((unsigned char)c, p->fp);
    p->last_output_newline = c == '\n';
}

static void token_printer_process(token_printer *p, const char *text, size_t len, bool finish) {
    const char *think_open = "<think>";
    const char *think_close = "</think>";
    size_t total = p->pending_len + len;
    char *buf = malloc(total ? total : 1);
    if (!buf) return;
    if (p->pending_len) memcpy(buf, p->pending, p->pending_len);
    if (len) memcpy(buf + p->pending_len, text, len);
    p->pending_len = 0;

    size_t i = 0;
    while (i < total) {
        const char *cur = buf + i;
        const size_t rem = total - i;
        if (bytes_has_prefix(cur, rem, think_open)) {
            p->in_think = true;
            i += strlen(think_open);
            continue;
        }
        if (bytes_has_prefix(cur, rem, think_close)) {
            p->in_think = false;
            token_printer_reset_color(p);
            if (!p->last_output_newline) {
                fputc('\n', p->fp);
                p->last_output_newline = true;
            }
            i += strlen(think_close);
            continue;
        }
        if (!finish && cur[0] == '<' &&
            (bytes_is_partial_prefix(cur, rem, think_open) ||
             bytes_is_partial_prefix(cur, rem, think_close)))
        {
            if (rem < sizeof(p->pending)) {
                memcpy(p->pending, cur, rem);
                p->pending_len = rem;
            }
            break;
        }
        token_printer_write_char(p, cur[0]);
        i++;
    }

    free(buf);
}

static void token_printer_finish(token_printer *p) {
    if (p->format_thinking) {
        token_printer_process(p, NULL, 0, true);
        token_printer_reset_color(p);
    }
    fflush(p->fp);
}

static void generation_done(void *ud) {
    token_printer *p = ud;
    token_printer_finish(p);
    if (!p->last_output_newline) {
        fputc('\n', p->fp);
        p->last_output_newline = true;
    }
    fflush(p->fp);
}

static void token_printer_write_text(token_printer *p, const char *text, size_t len) {
    if (p->format_thinking) {
        token_printer_process(p, text, len, false);
    } else if (len) {
        fwrite(text, 1, len, p->fp);
        p->last_output_newline = text[len - 1] == '\n';
    }
}

static void print_generated_token(void *ud, int token) {
    token_printer *p = ud;
    size_t len = 0;
    char *text = ds4_token_text(p->engine, token, &len);
    token_printer_write_text(p, text, len);
    fflush(p->fp);
    free(text);
}

static void build_prompt(ds4_engine *engine, const cli_generation_options *gen, ds4_tokens *out) {
    if (is_rendered_chat_prompt(gen->prompt)) {
        ds4_tokenize_rendered_chat(engine, gen->prompt, out);
    } else {
        ds4_encode_chat_prompt(engine, gen->system, gen->prompt,
                               cli_effective_think_mode(gen), out);
    }
}

static int run_sampled_generation(ds4_engine *engine, const cli_config *cfg, const ds4_tokens *prompt) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg->gen.ctx_size) != 0) {
        fprintf(stderr, "ds4: sampled CLI generation requires a session backend\n");
        return 1;
    }

    char err[160];
    ds4_think_mode think_mode = cli_effective_think_mode(&cfg->gen);
    token_printer printer = {
        .engine = engine,
        .fp = stdout,
        .format_thinking = ds4_think_mode_enabled(think_mode),
        .in_think = ds4_think_mode_enabled(think_mode),
        .use_color = isatty(fileno(stdout)) != 0,
        .last_output_newline = true,
    };
    cli_prefill_progress progress = {
        .base_tokens = 0,
        .input_tokens = prompt->len,
        .use_color = ds4_log_is_tty(stderr),
    };

    const double t_prefill0 = cli_now_sec();
    ds4_session_set_progress(session, cli_prefill_progress_cb, &progress);
    ds4_session_set_display_progress(session,
                                     progress.use_color ? cli_prefill_progress_cb : NULL,
                                     progress.use_color ? &progress : NULL);
    if (ds4_session_sync(session, prompt, err, sizeof(err)) != 0) {
        ds4_session_set_progress(session, NULL, NULL);
        ds4_session_set_display_progress(session, NULL, NULL);
        fprintf(stderr, "ds4: prompt processing failed: %s\n", err);
        ds4_session_free(session);
        return 1;
    }
    ds4_session_set_progress(session, NULL, NULL);
    ds4_session_set_display_progress(session, NULL, NULL);
    const double t_prefill1 = cli_now_sec();

    int max_tokens = cfg->gen.n_predict;
    int room = ds4_session_ctx(session) - ds4_session_pos(session);
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;

    uint64_t rng = cfg->gen.seed ? cfg->gen.seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());
    int generated = 0;
    const double t_decode0 = cli_now_sec();
    while (generated < max_tokens && !cli_interrupt_requested()) {
        int token = ds4_session_sample(session, cfg->gen.temperature, 0,
                                       cfg->gen.top_p, cfg->gen.min_p, &rng);
        if (token == ds4_token_eos(engine)) break;

        int toks[17];
        int ntok = 0;
        if (cfg->gen.temperature <= 0.0f && ds4_engine_mtp_draft_tokens(engine) > 1 &&
            getenv("DS4_MTP_SPEC_DISABLE") == NULL) {
            ntok = ds4_session_eval_speculative_argmax(session,
                                                       token,
                                                       max_tokens - generated,
                                                       ds4_token_eos(engine),
                                                       toks,
                                                       (int)(sizeof(toks) / sizeof(toks[0])),
                                                       err,
                                                       sizeof(err));
            if (ntok < 0) {
                fprintf(stderr, "ds4: decode failed: %s\n", err);
                ds4_session_free(session);
                return 1;
            }
        } else {
            if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4: decode failed: %s\n", err);
                ds4_session_free(session);
                return 1;
            }
            toks[0] = token;
            ntok = 1;
        }

        bool stop = false;
        for (int j = 0; j < ntok; j++) {
            if (toks[j] == ds4_token_eos(engine)) {
                stop = true;
                break;
            }
            size_t piece_len = 0;
            char *piece = ds4_token_text(engine, toks[j], &piece_len);
            token_printer_write_text(&printer, piece, piece_len);
            fflush(stdout);
            free(piece);
            generated++;
            if (generated >= max_tokens) break;
        }
        if (stop) break;
    }
    const double t_decode1 = cli_now_sec();
    generation_done(&printer);
    if (cli_interrupt_requested()) cli_interrupt_clear();

    const double prefill_s = t_prefill1 - t_prefill0;
    const double decode_s = t_decode1 - t_decode0;
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: prefill: %.2f t/s, generation: %.2f t/s\n",
            prefill_s > 0.0 ? (double)prompt->len / prefill_s : 0.0,
            decode_s > 0.0 ? (double)generated / decode_s : 0.0);

    ds4_session_free(session);
    return 0;
}

static bool json_utf8_valid(const char *s, size_t n) {
    size_t i = 0;
    while (i < n) {
        unsigned char c = (unsigned char)s[i++];
        if (c < 0x80) continue;
        int need = 0;
        if (c >= 0xc2 && c <= 0xdf) need = 1;
        else if (c >= 0xe0 && c <= 0xef) need = 2;
        else if (c >= 0xf0 && c <= 0xf4) need = 3;
        else return false;
        if (i + (size_t)need > n) return false;
        unsigned char c1 = (unsigned char)s[i];
        if (c == 0xe0 && c1 < 0xa0) return false;
        if (c == 0xed && c1 >= 0xa0) return false;
        if (c == 0xf0 && c1 < 0x90) return false;
        if (c == 0xf4 && c1 >= 0x90) return false;
        for (int j = 0; j < need; j++) {
            unsigned char cc = (unsigned char)s[i + (size_t)j];
            if ((cc & 0xc0) != 0x80) return false;
        }
        i += (size_t)need;
    }
    return true;
}

static void json_write_string(FILE *fp, const char *s, size_t n) {
    bool valid_utf8 = json_utf8_valid(s, n);
    fputc('"', fp);
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') {
            fputc('\\', fp);
            fputc((char)c, fp);
        } else if (c == '\n') {
            fputs("\\n", fp);
        } else if (c == '\r') {
            fputs("\\r", fp);
        } else if (c == '\t') {
            fputs("\\t", fp);
        } else if (c < 0x20) {
            fprintf(fp, "\\u%04x", (unsigned)c);
        } else if (!valid_utf8 && c >= 0x80) {
            /* Tokenizer pieces can be arbitrary byte fragments.  The bytes
             * array is authoritative; this escape keeps the JSON valid. */
            fprintf(fp, "\\u%04x", (unsigned)c);
        } else {
            fputc((char)c, fp);
        }
    }
    fputc('"', fp);
}

static void json_write_token(FILE *fp, ds4_engine *engine, int token) {
    size_t n = 0;
    char *text = ds4_token_text(engine, token, &n);
    fprintf(fp, "{\"id\":%d,\"text\":", token);
    json_write_string(fp, text, n);
    fputs(",\"bytes\":[", fp);
    for (size_t i = 0; i < n; i++) {
        if (i) fputc(',', fp);
        fprintf(fp, "%u", (unsigned)(unsigned char)text[i]);
    }
    fputc(']', fp);
    fputc('}', fp);
    free(text);
}

static int run_logits_dump(ds4_engine *engine, const cli_config *cfg, const ds4_tokens *prompt) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg->gen.ctx_size) != 0) {
        fprintf(stderr, "ds4: --dump-logits requires a graph session backend\n");
        return 1;
    }

    char err[160];
    cli_prefill_progress progress = {
        .base_tokens = 0,
        .input_tokens = prompt->len,
        .use_color = ds4_log_is_tty(stderr),
    };
    ds4_session_set_progress(session, cli_prefill_progress_cb, &progress);
    ds4_session_set_display_progress(session,
                                     progress.use_color ? cli_prefill_progress_cb : NULL,
                                     progress.use_color ? &progress : NULL);
    if (ds4_session_sync(session, prompt, err, sizeof(err)) != 0) {
        ds4_session_set_progress(session, NULL, NULL);
        ds4_session_set_display_progress(session, NULL, NULL);
        fprintf(stderr, "ds4: prompt processing failed: %s\n", err);
        ds4_session_free(session);
        return 1;
    }
    ds4_session_set_progress(session, NULL, NULL);
    ds4_session_set_display_progress(session, NULL, NULL);

    const int vocab = ds4_engine_vocab_size(engine);
    float *logits = malloc((size_t)vocab * sizeof(logits[0]));
    if (!logits) {
        ds4_session_free(session);
        return 1;
    }
    if (ds4_session_copy_logits(session, logits, vocab) != vocab) {
        fprintf(stderr, "ds4: failed to copy session logits\n");
        free(logits);
        ds4_session_free(session);
        return 1;
    }

    FILE *fp = fopen(cfg->gen.dump_logits_path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open --dump-logits file: %s\n", cfg->gen.dump_logits_path);
        free(logits);
        ds4_session_free(session);
        return 1;
    }

    fprintf(fp, "{\n  \"source\":\"ds4\",\n  \"model\":");
    json_write_string(fp, cfg->engine.model_path, strlen(cfg->engine.model_path));
    fprintf(fp,
            ",\n  \"backend\":\"%s\",\n  \"quant_bits\":%d,\n"
            "  \"prompt_tokens\":%d,\n  \"ctx\":%d,\n  \"vocab\":%d,\n",
            ds4_backend_name(cfg->engine.backend),
            ds4_engine_routed_quant_bits(engine),
            prompt->len,
            cfg->gen.ctx_size,
            vocab);
    const int argmax = ds4_session_argmax(session);
    fputs("  \"argmax_token\":", fp);
    json_write_token(fp, engine, argmax);
    fprintf(fp, ",\n  \"argmax_logit\":%.9g,\n  \"logits\":[", logits[argmax]);
    for (int i = 0; i < vocab; i++) {
        if (i) fputc(',', fp);
        if ((i % 8) == 0) fputs("\n    ", fp);
        if (isfinite(logits[i])) {
            fprintf(fp, "%.9g", logits[i]);
        } else {
            fputs("null", fp);
        }
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close --dump-logits file: %s\n", cfg->gen.dump_logits_path);
        free(logits);
        ds4_session_free(session);
        return 1;
    }

    free(logits);
    ds4_session_free(session);
    return 0;
}

static int run_logprob_dump(ds4_engine *engine, const cli_config *cfg, const ds4_tokens *prompt) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg->gen.ctx_size) != 0) {
        fprintf(stderr, "ds4: --dump-logprobs requires a graph session backend\n");
        return 1;
    }

    char err[160];
    cli_prefill_progress progress = {
        .base_tokens = 0,
        .input_tokens = prompt->len,
        .use_color = ds4_log_is_tty(stderr),
    };
    ds4_session_set_progress(session, cli_prefill_progress_cb, &progress);
    ds4_session_set_display_progress(session,
                                     progress.use_color ? cli_prefill_progress_cb : NULL,
                                     progress.use_color ? &progress : NULL);
    if (ds4_session_sync(session, prompt, err, sizeof(err)) != 0) {
        ds4_session_set_progress(session, NULL, NULL);
        ds4_session_set_display_progress(session, NULL, NULL);
        fprintf(stderr, "ds4: prompt processing failed: %s\n", err);
        ds4_session_free(session);
        return 1;
    }
    ds4_session_set_progress(session, NULL, NULL);
    ds4_session_set_display_progress(session, NULL, NULL);

    FILE *fp = fopen(cfg->gen.dump_logprobs_path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open --dump-logprobs file: %s\n", cfg->gen.dump_logprobs_path);
        ds4_session_free(session);
        return 1;
    }

    int k = cfg->gen.dump_logprobs_top_k > 0 ? cfg->gen.dump_logprobs_top_k : 20;
    if (k > 128) k = 128;
    ds4_token_score *scores = calloc((size_t)k, sizeof(scores[0]));
    if (!scores) {
        fclose(fp);
        ds4_session_free(session);
        return 1;
    }

    fprintf(fp, "{\n  \"source\":\"ds4\",\n  \"prompt_tokens\":%d,\n  \"ctx\":%d,\n  \"top_k\":%d,\n  \"steps\":[\n",
            prompt->len, cfg->gen.ctx_size, k);
    int generated = 0;
    int max_tokens = cfg->gen.n_predict;
    int room = ds4_session_ctx(session) - ds4_session_pos(session);
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;
    for (; generated < max_tokens; generated++) {
        int n = ds4_session_top_logprobs(session, scores, k);
        int token = ds4_session_argmax(session);
        if (generated) fputs(",\n", fp);
        fprintf(fp, "    {\"step\":%d,\"selected\":", generated);
        json_write_token(fp, engine, token);
        fputs(",\"top_logprobs\":[", fp);
        for (int i = 0; i < n && scores[i].id >= 0; i++) {
            if (i) fputc(',', fp);
            fputs("{\"token\":", fp);
            json_write_token(fp, engine, scores[i].id);
            fprintf(fp, ",\"logit\":%.9g,\"logprob\":%.9g}", scores[i].logit, scores[i].logprob);
        }
        fputs("]}", fp);

        if (token == ds4_token_eos(engine)) break;
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4: decode failed while dumping logprobs: %s\n", err);
            free(scores);
            fclose(fp);
            ds4_session_free(session);
            return 1;
        }
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close --dump-logprobs file: %s\n", cfg->gen.dump_logprobs_path);
        free(scores);
        ds4_session_free(session);
        return 1;
    }
    free(scores);
    ds4_session_free(session);
    return 0;
}

static int run_perplexity_file(ds4_engine *engine, const cli_config *cfg) {
    char *text = read_prompt_file(cfg->gen.perplexity_file_path, true);
    ds4_tokens tokens = {0};
    ds4_tokenize_text(engine, text, &tokens);
    free(text);

    /* Seed the graph with enough real context to stay on the normal Metal
     * prefill path; scoring starts immediately after this fixed prefix. */
    const int prefix_len = 32;
    if (tokens.len <= prefix_len) {
        fprintf(stderr, "ds4: --perplexity-file needs more than %d tokens\n", prefix_len);
        ds4_tokens_free(&tokens);
        return 1;
    }

    int scored = tokens.len - prefix_len;
    if (cfg->gen.n_predict > 0 && scored > cfg->gen.n_predict) scored = cfg->gen.n_predict;
    if (scored > cfg->gen.ctx_size - prefix_len) scored = cfg->gen.ctx_size - prefix_len;
    if (scored <= 0) {
        fprintf(stderr, "ds4: context too small for perplexity scoring\n");
        ds4_tokens_free(&tokens);
        return 1;
    }

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg->gen.ctx_size) != 0) {
        fprintf(stderr, "ds4: --perplexity-file requires a graph session backend\n");
        ds4_tokens_free(&tokens);
        return 1;
    }

    ds4_tokens prefix = {0};
    for (int i = 0; i < prefix_len; i++) ds4_tokens_push(&prefix, tokens.v[i]);
    char err[160];
    if (ds4_session_sync(session, &prefix, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4: perplexity initial token failed: %s\n", err);
        ds4_tokens_free(&prefix);
        ds4_session_free(session);
        ds4_tokens_free(&tokens);
        return 1;
    }
    ds4_tokens_free(&prefix);

    double nll = 0.0;
    for (int j = 0; j < scored; j++) {
        const int i = prefix_len + j;
        ds4_token_score score;
        if (!ds4_session_token_logprob(session, tokens.v[i], &score)) {
            fprintf(stderr, "ds4: failed to score token %d\n", i);
            ds4_session_free(session);
            ds4_tokens_free(&tokens);
            return 1;
        }
        nll -= (double)score.logprob;

        if (((j + 1) % 256) == 0 || j + 1 == scored) {
            fprintf(stderr, "ds4: perplexity scored %d/%d\r", j + 1, scored);
            fflush(stderr);
        }

        if (j + 1 < scored && ds4_session_eval(session, tokens.v[i], err, sizeof(err)) != 0) {
            fprintf(stderr, "\nds4: perplexity decode failed at token %d: %s\n", i, err);
            ds4_session_free(session);
            ds4_tokens_free(&tokens);
            return 1;
        }
    }
    fputc('\n', stderr);

    const double avg_nll = nll / (double)scored;
    printf("tokens=%d scored=%d nll=%.9f avg_nll=%.9f ppl=%.9f\n",
           tokens.len, scored, nll, avg_nll, exp(avg_nll));

    ds4_session_free(session);
    ds4_tokens_free(&tokens);
    return 0;
}

static int run_generation(ds4_engine *engine, const cli_config *cfg) {
    ds4_tokens prompt = {0};
    build_prompt(engine, &cfg->gen, &prompt);

    int rc = 0;
    if (cfg->gen.metal_graph_test) {
        rc = ds4_engine_metal_graph_test(engine, &prompt);
        ds4_tokens_free(&prompt);
        return rc;
    }
    if (cfg->gen.metal_graph_full_test) {
        rc = ds4_engine_metal_graph_full_test(engine, &prompt);
        ds4_tokens_free(&prompt);
        return rc;
    }
    if (cfg->gen.metal_graph_prompt_test) {
        rc = ds4_engine_metal_graph_prompt_test(engine, &prompt, cfg->gen.ctx_size);
        ds4_tokens_free(&prompt);
        return rc;
    }
    if (cfg->gen.dump_logits_path) {
        rc = run_logits_dump(engine, cfg, &prompt);
        ds4_tokens_free(&prompt);
        return rc;
    }
    if (cfg->gen.dump_logprobs_path) {
        rc = run_logprob_dump(engine, cfg, &prompt);
        ds4_tokens_free(&prompt);
        return rc;
    }

    const bool diagnostic = cfg->gen.dump_tokens ||
                            cfg->gen.head_test ||
                            cfg->gen.first_token_test;
    if (cfg->gen.head_test) {
        rc = ds4_engine_head_test(engine, &prompt);
    }
    if (rc == 0 && cfg->gen.first_token_test) {
        rc = ds4_engine_first_token_test(engine, &prompt);
    }
    if (cfg->gen.dump_tokens) {
        ds4_engine_dump_tokens(engine, &prompt);
    }

    if (diagnostic) {
        if (rc == 0) {
            fprintf(stderr, "ds4: diagnostic run completed on the native %s path.\n",
                    ds4_backend_name(cfg->engine.backend));
        }
    } else if (cfg->gen.temperature > 0.0f || ds4_engine_mtp_draft_tokens(engine) > 1) {
        rc = run_sampled_generation(engine, cfg, &prompt);
    } else {
        token_printer printer = {
            .engine = engine,
            .fp = stdout,
            .format_thinking = ds4_think_mode_enabled(cli_effective_think_mode(&cfg->gen)),
            .in_think = ds4_think_mode_enabled(cli_effective_think_mode(&cfg->gen)),
            .use_color = isatty(fileno(stdout)) != 0,
            .last_output_newline = true,
        };
        cli_prefill_progress progress = {
            .base_tokens = 0,
            .input_tokens = prompt.len,
            .use_color = ds4_log_is_tty(stderr),
        };
        rc = ds4_engine_generate_argmax(engine, &prompt, cfg->gen.n_predict,
                                        cfg->gen.ctx_size,
                                        print_generated_token,
                                        generation_done,
                                        &printer,
                                        cli_prefill_progress_cb,
                                        &progress);
    }

    ds4_tokens_free(&prompt);
    return rc;
}

static char *trim_inplace(char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) end--;
    *end = '\0';
    return s;
}

static void print_repl_help(void) {
    puts("Commands:");
    puts("  /help          Show this help.");
    puts("  /think         Use normal thinking mode.");
    puts("  /think-max     Use Think Max only when context is at least 393216 tokens.");
    puts("  /nothink       Disable thinking mode.");
    puts("  /ctx N         Set context size for following prompts.");
    puts("  /power N       Set GPU duty cycle percentage, 1..100.");
    puts("  /read FILE     Read a prompt from FILE and run it.");
    puts("  /quit, /exit   Leave the prompt.");
    puts("  Ctrl+C         Stop generation and return to the prompt.");
}

static bool parse_power_percent(const char *arg, int *out) {
    char *end = NULL;
    long v = strtol(arg, &end, 10);
    if (!arg[0] || *end != '\0' || v < 1 || v > 100) return false;
    *out = (int)v;
    return true;
}

static void history_file_path(char *buf, size_t len) {
    const char *home = getenv("HOME");
    if (!home || !home[0]) home = ".";
    snprintf(buf, len, "%s/.ds4_history", home);
}

typedef struct {
    ds4_session *session;
    ds4_tokens transcript;
    int ctx_size;
    int max_prefix_tokens;
} repl_chat;

static void tokens_insert(ds4_tokens *dst, int pos, const ds4_tokens *src) {
    if (!src || src->len <= 0) return;
    if (pos < 0) pos = 0;
    if (pos > dst->len) pos = dst->len;
    while (dst->len + src->len > dst->cap) {
        dst->cap = dst->cap ? dst->cap * 2 : 64;
        int *next = realloc(dst->v, (size_t)dst->cap * sizeof(dst->v[0]));
        if (!next) {
            perror("ds4: realloc");
            exit(1);
        }
        dst->v = next;
    }
    memmove(dst->v + pos + src->len, dst->v + pos,
            (size_t)(dst->len - pos) * sizeof(dst->v[0]));
    memcpy(dst->v + pos, src->v, (size_t)src->len * sizeof(src->v[0]));
    dst->len += src->len;
}

static void tokens_remove(ds4_tokens *dst, int pos, int n) {
    if (n <= 0 || pos < 0 || pos >= dst->len) return;
    if (pos + n > dst->len) n = dst->len - pos;
    memmove(dst->v + pos, dst->v + pos + n,
            (size_t)(dst->len - pos - n) * sizeof(dst->v[0]));
    dst->len -= n;
}

/* Insert/remove the Think Max prefix inside the existing transcript.  The
 * prefix lives after BOS, before any system/developer text, which mirrors the
 * API rendering path.  Changing it invalidates the session because every later
 * token position would otherwise refer to the wrong prefix. */
static void repl_chat_apply_max_prefix(ds4_engine *engine, repl_chat *chat, bool enable) {
    if (enable && chat->max_prefix_tokens == 0) {
        ds4_tokens prefix = {0};
        ds4_chat_append_max_effort_prefix(engine, &prefix);
        tokens_insert(&chat->transcript, 1, &prefix);
        chat->max_prefix_tokens = prefix.len;
        ds4_tokens_free(&prefix);
        if (chat->session) ds4_session_invalidate(chat->session);
    } else if (!enable && chat->max_prefix_tokens > 0) {
        tokens_remove(&chat->transcript, 1, chat->max_prefix_tokens);
        chat->max_prefix_tokens = 0;
        if (chat->session) ds4_session_invalidate(chat->session);
    }
}

static int repl_chat_create_session(ds4_engine *engine, repl_chat *chat, int ctx_size) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, ctx_size) != 0) {
        fprintf(stderr, "ds4: interactive chat KV cache requires a session backend\n");
        return 1;
    }
    if (chat->session) ds4_session_free(chat->session);
    chat->session = session;
    chat->ctx_size = ctx_size;
    return 0;
}

static int repl_chat_init(ds4_engine *engine, repl_chat *chat, const cli_config *cfg) {
    memset(chat, 0, sizeof(*chat));
    ds4_chat_begin(engine, &chat->transcript);
    repl_chat_apply_max_prefix(engine, chat,
                               cli_effective_think_mode(&cfg->gen) == DS4_THINK_MAX);
    if (cfg->gen.system && cfg->gen.system[0]) {
        ds4_chat_append_message(engine, &chat->transcript, "system", cfg->gen.system);
    }
    return repl_chat_create_session(engine, chat, cfg->gen.ctx_size);
}

static void repl_chat_free(repl_chat *chat) {
    if (!chat) return;
    ds4_session_free(chat->session);
    ds4_tokens_free(&chat->transcript);
    memset(chat, 0, sizeof(*chat));
}

static int repl_chat_set_ctx(ds4_engine *engine, repl_chat *chat, int ctx_size) {
    ds4_session_free(chat->session);
    chat->session = NULL;
    chat->ctx_size = 0;
    return repl_chat_create_session(engine, chat, ctx_size);
}

/* Run one interactive turn.  The transcript is tentatively extended with user
 * and assistant markers, then ds4_session_sync() decides whether this is a KV
 * continuation.  If prompt processing fails, the transcript rolls back before
 * returning to the prompt. */
static int run_chat_turn(ds4_engine *engine, cli_config *cfg, repl_chat *chat, const char *user_text) {
    if (!chat->session) {
        fprintf(stderr, "ds4: no active interactive KV cache\n");
        return 1;
    }

    ds4_think_mode think_mode = ds4_think_mode_for_context(cfg->gen.think_mode,
                                                           chat->ctx_size);
    repl_chat_apply_max_prefix(engine, chat, think_mode == DS4_THINK_MAX);
    const int rollback_len = chat->transcript.len;
    ds4_chat_append_message(engine, &chat->transcript, "user", user_text);
    ds4_chat_append_assistant_prefix(engine, &chat->transcript, think_mode);

    const int old_pos = ds4_session_pos(chat->session);
    const int common = ds4_session_common_prefix(chat->session, &chat->transcript);
    const int cached = common == old_pos && chat->transcript.len >= old_pos ? common : 0;
    const int suffix = chat->transcript.len - cached;

    char err[160];
    cli_prefill_progress progress = {
        .base_tokens = cached,
        .input_tokens = suffix,
        .use_color = ds4_log_is_tty(stderr),
    };
    const double t_prefill0 = cli_now_sec();
    ds4_session_set_progress(chat->session, cli_prefill_progress_cb, &progress);
    ds4_session_set_display_progress(chat->session,
                                     progress.use_color ? cli_prefill_progress_cb : NULL,
                                     progress.use_color ? &progress : NULL);
    if (ds4_session_sync(chat->session, &chat->transcript, err, sizeof(err)) != 0) {
        ds4_session_set_progress(chat->session, NULL, NULL);
        ds4_session_set_display_progress(chat->session, NULL, NULL);
        chat->transcript.len = rollback_len;
        fprintf(stderr, "ds4: prompt processing failed: %s\n", err);
        return 1;
    }
    ds4_session_set_progress(chat->session, NULL, NULL);
    ds4_session_set_display_progress(chat->session, NULL, NULL);
    const double t_prefill1 = cli_now_sec();

    token_printer printer = {
        .engine = engine,
        .fp = stdout,
        .format_thinking = ds4_think_mode_enabled(think_mode),
        .in_think = ds4_think_mode_enabled(think_mode),
        .use_color = isatty(fileno(stdout)) != 0,
        .last_output_newline = true,
    };

    int max_tokens = cfg->gen.n_predict;
    int room = ds4_session_ctx(chat->session) - ds4_session_pos(chat->session);
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;

    uint64_t rng = cfg->gen.seed ? cfg->gen.seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());
    int generated = 0;
    const double t_decode0 = cli_now_sec();
    while (generated < max_tokens && !cli_interrupt_requested()) {
        int token = ds4_session_sample(chat->session,
                                       cfg->gen.temperature,
                                       0,
                                       cfg->gen.top_p,
                                       cfg->gen.min_p,
                                       &rng);
        if (token == ds4_token_eos(engine)) break;

        int toks[17];
        int ntok = 0;
        if (cfg->gen.temperature <= 0.0f && ds4_engine_mtp_draft_tokens(engine) > 1 &&
            getenv("DS4_MTP_SPEC_DISABLE") == NULL) {
            ntok = ds4_session_eval_speculative_argmax(chat->session,
                                                       token,
                                                       max_tokens - generated,
                                                       ds4_token_eos(engine),
                                                       toks,
                                                       (int)(sizeof(toks) / sizeof(toks[0])),
                                                       err,
                                                       sizeof(err));
            if (ntok < 0) {
                fprintf(stderr, "ds4: decode failed: %s\n", err);
                return 1;
            }
        } else {
            if (ds4_session_eval(chat->session, token, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4: decode failed: %s\n", err);
                return 1;
            }
            toks[0] = token;
            ntok = 1;
        }

        bool stop = false;
        for (int j = 0; j < ntok; j++) {
            if (toks[j] == ds4_token_eos(engine)) {
                stop = true;
                break;
            }
            size_t piece_len = 0;
            char *piece = ds4_token_text(engine, toks[j], &piece_len);
            ds4_tokens_push(&chat->transcript, toks[j]);
            token_printer_write_text(&printer, piece, piece_len);
            fflush(stdout);
            free(piece);
            generated++;
            if (generated >= max_tokens) break;
        }
        if (stop) break;
    }
    const double t_decode1 = cli_now_sec();
    generation_done(&printer);

    const bool interrupted = cli_interrupt_requested();
    if (interrupted && generated == 0) {
        chat->transcript.len = rollback_len;
        ds4_session_invalidate(chat->session);
    } else {
        ds4_tokens_push(&chat->transcript, ds4_token_eos(engine));
    }

    const double prefill_s = t_prefill1 - t_prefill0;
    const double decode_s = t_decode1 - t_decode0;
    if (interrupted) cli_interrupt_clear();
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: prefill: %.2f t/s, generation: %.2f t/s\n",
            prefill_s > 0.0 ? (double)suffix / prefill_s : 0.0,
            decode_s > 0.0 ? (double)generated / decode_s : 0.0);
    return 0;
}

static int run_repl(ds4_engine *engine, cli_config *cfg) {
    repl_chat chat;
    if (repl_chat_init(engine, &chat, cfg) != 0) return 1;

    struct sigaction old_int;
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sigemptyset(&sa.sa_mask);
    sa.sa_handler = cli_sigint_handler;
    bool sigint_installed = sigaction(SIGINT, &sa, &old_int) == 0;
    cli_interrupt_clear();

    char hist[PATH_MAX];
    history_file_path(hist, sizeof(hist));
    linenoiseSetMultiLine(1);
    linenoiseHistorySetMaxLen(512);
    linenoiseHistoryLoad(hist);
    print_repl_help();

    int rc = 0;
    for (;;) {
        errno = 0;
        char *line = linenoise("ds4> ");
        if (!line) {
            if (errno == EAGAIN || cli_interrupt_requested()) {
                cli_interrupt_clear();
                continue;
            }
            break;
        }
        char *cmd = trim_inplace(line);
        if (!cmd[0]) {
            linenoiseFree(line);
            continue;
        }
        linenoiseHistoryAdd(cmd);
        linenoiseHistorySave(hist);

        if (!strcmp(cmd, "/help")) {
            print_repl_help();
        } else if (!strcmp(cmd, "/think")) {
            cfg->gen.think_mode = DS4_THINK_HIGH;
            repl_chat_apply_max_prefix(engine, &chat, false);
            puts("Thinking mode: high.");
        } else if (!strcmp(cmd, "/think-max")) {
            cfg->gen.think_mode = DS4_THINK_MAX;
            bool active = ds4_think_mode_for_context(cfg->gen.think_mode,
                                                     chat.ctx_size) == DS4_THINK_MAX;
            repl_chat_apply_max_prefix(engine, &chat, active);
            cli_warn_think_max_downgraded(&cfg->gen, "/think-max");
            printf("Thinking mode: %s.\n", active ? "max" : "high (ctx below 393216)");
        } else if (!strcmp(cmd, "/nothink")) {
            cfg->gen.think_mode = DS4_THINK_NONE;
            repl_chat_apply_max_prefix(engine, &chat, false);
            puts("Thinking mode: none.");
        } else if (!strncmp(cmd, "/power", 6) && (cmd[6] == '\0' || isspace((unsigned char)cmd[6]))) {
            char *arg = trim_inplace(cmd + 6);
            if (!arg[0]) {
                printf("Power: %d%%.\n", ds4_session_power(chat.session));
            } else {
                int power = 0;
                if (!parse_power_percent(arg, &power)) {
                    fprintf(stderr, "ds4: /power must be between 1 and 100\n");
                } else if (ds4_session_set_power(chat.session, power) != 0) {
                    fprintf(stderr, "ds4: failed to set /power\n");
                } else {
                    cfg->engine.power_percent = power;
                    printf("Power: %d%%.\n", power);
                }
            }
        } else if (!strncmp(cmd, "/ctx", 4) && (cmd[4] == '\0' || isspace((unsigned char)cmd[4]))) {
            char *arg = trim_inplace(cmd + 4);
            if (!arg[0]) {
                fprintf(stderr, "ds4: /ctx needs a positive integer\n");
            } else {
                cfg->gen.ctx_size = parse_int(arg, "/ctx");
                log_context_memory(cfg->engine.backend, cfg->gen.ctx_size);
                rc = repl_chat_set_ctx(engine, &chat, cfg->gen.ctx_size);
                if (rc != 0) {
                    linenoiseFree(line);
                    break;
                }
                bool active = ds4_think_mode_for_context(cfg->gen.think_mode,
                                                         chat.ctx_size) == DS4_THINK_MAX;
                repl_chat_apply_max_prefix(engine, &chat, active);
                cli_warn_think_max_downgraded(&cfg->gen, "/ctx");
            }
        } else if (!strcmp(cmd, "/quit") || !strcmp(cmd, "/exit")) {
            linenoiseFree(line);
            break;
        } else if (!strncmp(cmd, "/read", 5) && (cmd[5] == '\0' || isspace((unsigned char)cmd[5]))) {
            char *path = trim_inplace(cmd + 5);
            if (!path[0]) {
                fprintf(stderr, "ds4: /read needs a file path\n");
            } else {
                char *prompt = read_prompt_file(path, false);
                if (prompt) {
                    rc = run_chat_turn(engine, cfg, &chat, prompt);
                    free(prompt);
                }
            }
        } else if (cmd[0] == '/') {
            fprintf(stderr, "ds4: unknown command: %s\n", cmd);
            fprintf(stderr, "ds4: type /help for commands\n");
        } else {
            rc = run_chat_turn(engine, cfg, &chat, cmd);
        }
        linenoiseFree(line);
    }
    if (sigint_installed) sigaction(SIGINT, &old_int, NULL);
    repl_chat_free(&chat);
    return rc;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4: missing value for %s\n", opt);
        exit(2);
    }
    return argv[++(*i)];
}

static char *read_prompt_file(const char *path, bool fatal) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open prompt file: %s\n", path);
        if (fatal) exit(2);
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "ds4: failed to seek prompt file: %s\n", path);
        fclose(fp);
        if (fatal) exit(2);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0) {
        fprintf(stderr, "ds4: failed to size prompt file: %s\n", path);
        fclose(fp);
        if (fatal) exit(2);
        return NULL;
    }
    rewind(fp);

    char *buf = malloc((size_t)len + 1);
    if (!buf) {
        fprintf(stderr, "ds4: out of memory reading prompt file: %s\n", path);
        fclose(fp);
        if (fatal) exit(2);
        return NULL;
    }
    size_t nread = fread(buf, 1, (size_t)len, fp);
    if (nread != (size_t)len) {
        fprintf(stderr, "ds4: failed to read prompt file: %s\n", path);
        free(buf);
        fclose(fp);
        if (fatal) exit(2);
        return NULL;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close prompt file: %s\n", path);
        free(buf);
        if (fatal) exit(2);
        return NULL;
    }
    buf[len] = '\0';
    return buf;
}

static cli_config parse_options(int argc, char **argv) {
    cli_config c = {
        .engine = {
            .model_path = "ds4flash.gguf",
            .backend = default_backend(),
            .mtp_draft_tokens = 1,
            .mtp_draft_tree_width = 1,  /* silv 2026-05-27: spec-tree default linear */
            .mtp_margin = 3.0f,
        },
        .gen = {
            .prompt = NULL,
            .system = "You are a helpful assistant",
            .n_predict = 50000,
            .ctx_size = 32768,
            .temperature = DS4_DEFAULT_TEMPERATURE,
            .top_p = DS4_DEFAULT_TOP_P,
            .min_p = DS4_DEFAULT_MIN_P,
            .dump_logprobs_top_k = 20,
            .think_mode = DS4_THINK_HIGH,
        },
    };

    bool directional_steering_scale_set = false;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "-p") || !strcmp(arg, "--prompt")) {
            if (c.gen.prompt) {
                fprintf(stderr, "ds4: specify only one prompt source\n");
                exit(2);
            }
            c.gen.prompt = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            if (c.gen.prompt) {
                fprintf(stderr, "ds4: specify only one prompt source\n");
                exit(2);
            }
            c.prompt_owned = read_prompt_file(need_arg(&i, argc, argv, arg), true);
            c.gen.prompt = c.prompt_owned;
        } else if (!strcmp(arg, "-sys") || !strcmp(arg, "--system")) {
            c.gen.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.engine.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp")) {
            c.engine.mtp_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp-draft")) {
            c.engine.mtp_draft_tokens = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--mtp-tree-width")) {
            /* silv 2026-05-27: spec-tree branching factor (1=linear, 2-4=tree). */
            c.engine.mtp_draft_tree_width = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--mtp-margin")) {
            c.engine.mtp_margin = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1000.0f);
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) {
            c.gen.n_predict = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            c.gen.ctx_size = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--temp")) {
            c.gen.temperature = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 100.0f);
        } else if (!strcmp(arg, "--top-p")) {
            c.gen.top_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--min-p")) {
            c.gen.min_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--seed")) {
            c.gen.seed = parse_u64(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--quality")) {
            c.engine.quality = true;
        } else if (!strcmp(arg, "--power")) {
            c.engine.power_percent = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (c.engine.power_percent < 1 || c.engine.power_percent > 100) {
                fprintf(stderr, "ds4: --power must be between 1 and 100\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--dir-steering-file")) {
            c.engine.directional_steering_file = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dir-steering-ffn")) {
            c.engine.directional_steering_ffn = parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            directional_steering_scale_set = true;
        } else if (!strcmp(arg, "--dir-steering-attn")) {
            c.engine.directional_steering_attn = parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            directional_steering_scale_set = true;
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.engine.n_threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--backend")) {
            c.engine.backend = parse_backend(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--cpu")) {
            c.engine.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "--metal")) {
            c.engine.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--cuda")) {
            c.engine.backend = DS4_BACKEND_CUDA;
        } else if (!strcmp(arg, "--cpu-moe")) {
            c.engine.cpu_moe = true;
        } else if (!strcmp(arg, "--n-cpu-moe")) {
            c.engine.n_cpu_moe_layers = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--prefill-metal-phases")) {
            const char *s = need_arg(&i, argc, argv, arg);
            if (!strcmp(s, "auto")) {
                c.engine.prefill_metal_phases = -1;
            } else {
                c.engine.prefill_metal_phases = parse_int(s, arg);
            }
        } else if (!strcmp(arg, "--mtl4-moe")) {
            /* MTL4 ML packed-MoE path (group=6, n_tokens≤16). Sets the env
             * var that ds4_gpu_routed_moe_batch_tensor_mtl4 reads at first
             * call. Scaffold path: falls back to legacy at runtime when
             * not yet implemented; preserves the architectural seam +
             * journal trace. See ds4_metal.m for the H1672 pattern outline. */
            setenv("DS4_MTL4_MOE_ENABLE", "1", 1);
            fprintf(stderr, "ds4: --mtl4-moe enabled (DS4_MTL4_MOE_ENABLE=1)\n");
        } else if (!strcmp(arg, "--dump-tokens")) {
            c.gen.dump_tokens = true;
        } else if (!strcmp(arg, "--dump-logits")) {
            c.gen.dump_logits_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dump-logprobs")) {
            c.gen.dump_logprobs_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--logprobs-top-k")) {
            c.gen.dump_logprobs_top_k = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--perplexity-file")) {
            c.gen.perplexity_file_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--imatrix-dataset")) {
            c.gen.imatrix_dataset_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--imatrix-out")) {
            c.gen.imatrix_output_path = need_arg(&i, argc, argv, arg);
            c.engine.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--imatrix-max-prompts")) {
            c.gen.imatrix_max_prompts = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--imatrix-max-tokens")) {
            c.gen.imatrix_max_tokens = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--think")) {
            c.gen.think_mode = DS4_THINK_HIGH;
        } else if (!strcmp(arg, "--think-max")) {
            c.gen.think_mode = DS4_THINK_MAX;
        } else if (!strcmp(arg, "--nothink")) {
            c.gen.think_mode = DS4_THINK_NONE;
        } else if (!strcmp(arg, "--head-test")) {
            c.gen.head_test = true;
        } else if (!strcmp(arg, "--first-token-test")) {
            c.gen.first_token_test = true;
        } else if (!strcmp(arg, "--metal-graph-test")) {
            c.gen.metal_graph_test = true;
            c.engine.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--metal-graph-full-test")) {
            c.gen.metal_graph_full_test = true;
            c.engine.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--metal-graph-prompt-test")) {
            c.gen.metal_graph_prompt_test = true;
            c.engine.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--metal-graph-generate")) {
            fprintf(stderr, "ds4: --metal-graph-generate was removed; --metal is the graph path\n");
            exit(2);
        } else if (!strcmp(arg, "--inspect")) {
            c.inspect = true;
        } else if (!strcmp(arg, "--warm-weights")) {
            c.engine.warm_weights = true;
        } else if (!strcmp(arg, "--server")) {
            fprintf(stderr, "ds4: use ds4-server for the HTTP server\n");
            exit(2);
        } else {
            fprintf(stderr, "ds4: unknown option: %s\n", arg);
            usage(stderr);
            exit(2);
        }
    }

    if (c.engine.directional_steering_file && !directional_steering_scale_set) {
        c.engine.directional_steering_ffn = 1.0f;
    }
    if (c.gen.imatrix_output_path && !c.gen.imatrix_dataset_path) {
        fprintf(stderr, "ds4: --imatrix-out requires --imatrix-dataset\n");
        exit(2);
    }
    if (c.gen.imatrix_dataset_path && !c.gen.imatrix_output_path) {
        fprintf(stderr, "ds4: --imatrix-dataset requires --imatrix-out\n");
        exit(2);
    }
    if (c.gen.perplexity_file_path && c.gen.prompt) {
        fprintf(stderr, "ds4: --perplexity-file does not use -p/--prompt-file\n");
        exit(2);
    }

    return c;
}

int main(int argc, char **argv) {
    /* --polar-canary [packets [pairs]] : dispatch the MTL4 polar_dot kernel
     * on synthetic deterministic inputs and report GPU elapsed + max error.
     * Standalone diagnostic for task #563 (codex H1725 port). Bypasses
     * engine init since no model is needed. */
    if (argc >= 2 && !strcmp(argv[1], "--polar-canary")) {
        uint32_t packets = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 7776;
        uint32_t pairs   = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 2048;
        return ds4_gpu_mtl4_polar_dot_canary(packets, pairs) ? 0 : 1;
    }
    /* --polar-tile-canary [tiles [rows [batches [pairs]]]] : H1729 tile×row×batch
     * polar dot, the deployable layout for routed-MoE inference. Defaults
     * match codex H1727: tiles=2592/32=81 (one expert layer's gate split into
     * 32-row tiles × 32 batches), rows=32, batches=8, pairs=2048. */
    if (argc >= 2 && !strcmp(argv[1], "--polar-tile-canary")) {
        uint32_t tiles   = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 81;
        uint32_t rows    = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        uint32_t batches = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 8;
        uint32_t pairs   = (argc >= 6) ? (uint32_t)atoi(argv[5]) : 2048;
        return ds4_gpu_mtl4_polar_tile_canary(tiles, rows, batches, pairs) ? 0 : 1;
    }
    /* --polar-tile-real <prefix> [tiles [rows [batches [pairs]]]] : load real
     * polar binaries from <prefix>.{mag,phase,levels,hidden,cos_lut,sin_lut}.bin
     * and validate GPU output against <prefix>.expected_polar.bin. */
    if (argc >= 3 && !strcmp(argv[1], "--polar-tile-real")) {
        const char *prefix = argv[2];
        uint32_t tiles   = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 2;
        uint32_t rows    = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 32;
        uint32_t batches = (argc >= 6) ? (uint32_t)atoi(argv[5]) : 2;
        uint32_t pairs   = (argc >= 7) ? (uint32_t)atoi(argv[6]) : 2048;
        return ds4_gpu_mtl4_polar_tile_real(prefix, tiles, rows, batches, pairs) ? 0 : 1;
    }
    /* --polar-fused-canary [n_codes [route_pairs [rows [batches [pairs]]]]] :
     * H1733 fused gate*silu*up*route_weight in one dispatch. The deployable
     * shape for routed-MoE inference. */
    if (argc >= 2 && !strcmp(argv[1], "--polar-fused-canary")) {
        uint32_t n_codes     = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 16;
        uint32_t route_pairs = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 8;
        uint32_t rows        = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 32;
        uint32_t batches     = (argc >= 6) ? (uint32_t)atoi(argv[5]) : 1;
        uint32_t pairs       = (argc >= 7) ? (uint32_t)atoi(argv[6]) : 2048;
        return ds4_gpu_mtl4_polar_fused_canary(n_codes, route_pairs, rows, batches, pairs) ? 0 : 1;
    }
    /* --polar-file-info <path> : open a PLR2 combined polar-encoded file
     * (produced by analyzers/polar_encode_mlx.py --format combined) and
     * print header + per-expert decode sanity. Phase A of task #563 —
     * verifies the host-side reader before GPU MTLResidencySet binding. */
    if (argc >= 3 && !strcmp(argv[1], "--polar-file-info")) {
        ds4_polar_file pf = { .fd = -1 };
        if (!ds4_polar_open(argv[2], &pf)) return 1;
        ds4_polar_print_summary(&pf, argv[2]);
        ds4_polar_close(&pf);
        return 0;
    }
    /* --polar-dir-info <dir> : scan dir for L{LL}_{kind}.polar files and
     * report the layer/kind matrix + total mmap bytes resident. Phase
     * B-1 of #563 — verifies the pool API before engine integration. */
    if (argc >= 3 && !strcmp(argv[1], "--polar-dir-info")) {
        ds4_polar_pool pool;
        ds4_polar_pool_init(&pool);
        uint32_t opened = ds4_polar_pool_load_dir(&pool, argv[2]);
        ds4_polar_pool_print_summary(&pool, argv[2]);
        ds4_polar_pool_close(&pool);
        return opened > 0 ? 0 : 1;
    }
    /* --polar-gud-canary [n_codes [route_pairs [rows [batches [pairs [down_rows [act_rows]]]]]]]:
     * H1735 fused gate*silu*up*route_weight + down-projection in one
     * dispatch. Synthetic input where expected output = pairs^2 for all
     * cells (mag=0, phase=4, levels=1, hidden=1, down=1/act_rows, route_weight=1).
     * Tile policy hint (H1736/H1738): try down_rows=8 act_rows=16 at small
     * batches, down_rows=64 act_rows=32 at large batches. */
    if (argc >= 2 && !strcmp(argv[1], "--polar-gud-canary")) {
        uint32_t n_codes     = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 16;
        uint32_t route_pairs = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 8;
        uint32_t rows        = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 32;
        uint32_t batches     = (argc >= 6) ? (uint32_t)atoi(argv[5]) : 1;
        uint32_t pairs       = (argc >= 7) ? (uint32_t)atoi(argv[6]) : 2048;
        uint32_t down_rows   = (argc >= 8) ? (uint32_t)atoi(argv[7]) : 8;
        uint32_t act_rows    = (argc >= 9) ? (uint32_t)atoi(argv[8]) : 16;
        return ds4_gpu_mtl4_polar_gate_up_down_canary(n_codes, route_pairs, rows,
                                                      batches, pairs,
                                                      down_rows, act_rows) ? 0 : 1;
    }
    /* --polar-real-canary <polar_dir> [layer [expert [down_rows [act_rows]]]]:
     * #563 Phase B-2.2 real-data validation — load PLR2 files from
     * <polar_dir> for the given layer, copy expert <expert> rows into MTL
     * buffers, dispatch H1735 with hidden=1, route_weight=1, down=1/act_rows.
     * Compares GPU output to a CPU reference derived from polar decode.
     * Validates the PLR2 byte format → MTL4 GPU pipeline end-to-end. */
    if (argc >= 3 && !strcmp(argv[1], "--polar-real-canary")) {
        const char *dir   = argv[2];
        uint32_t layer    = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 0;
        uint32_t expert   = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 0;
        uint32_t down_rows = (argc >= 6) ? (uint32_t)atoi(argv[5]) : 8;
        uint32_t act_rows  = (argc >= 7) ? (uint32_t)atoi(argv[6]) : 16;
        return ds4_gpu_mtl4_polar_real_canary(dir, layer, expert, down_rows, act_rows) ? 0 : 1;
    }
    /* --vq-real-canary <vqb1_dir> [layer [expert [down_rows [act_rows]]]]:
     * VQ-2D codec validation. Reads VQB1 files from <vqb1_dir>/L{LL}_{kind}.vqb1,
     * loads expert codes + codebook into MTL buffers, dispatches gate_up_down_vq
     * kernel. Compares GPU output to a CPU reference computed via the same
     * codebook lookup. Validates VQB1 format → MTL4 GPU pipeline end-to-end.
     * Mirror of --polar-real-canary structure for the VQ codec arc. */
    if (argc >= 3 && !strcmp(argv[1], "--vq-real-canary")) {
        const char *dir   = argv[2];
        uint32_t layer    = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 0;
        uint32_t expert   = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 0;
        uint32_t down_rows = (argc >= 6) ? (uint32_t)atoi(argv[5]) : 8;
        uint32_t act_rows  = (argc >= 7) ? (uint32_t)atoi(argv[6]) : 16;
        return ds4_gpu_mtl4_vq_real_canary(dir, layer, expert, down_rows, act_rows) ? 0 : 1;
    }
    /* --softplus-sqrt-canary [n_rows [n_cols]] : silv 2026-05-27 task #670
     * MTL4 port of kernel_dsv4_softplus_sqrt_f32_4 (metal/unary.metal:290).
     * Feeds deterministic input through both MTL4 pipeline + CPU reference;
     * passes if max abs diff < 1e-4. First of 74-kernel classic → MTL4
     * sweep; validates the storage-class migration pattern (constant →
     * device const args_ptr) on the smallest viable kernel. */
    if (argc >= 2 && !strcmp(argv[1], "--softplus-sqrt-canary")) {
        const uint32_t n_rows = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 8;
        const uint32_t n_cols = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        return ds4_gpu_mtl4_softplus_sqrt_canary(n_rows, n_cols) ? 0 : 1;
    }
    /* --router-weights-one-canary : silv 2026-05-27 task #671. MTL4 port
     * of kernel_dsv4_router_weights_one (metal/dsv4_misc.metal:263). */
    if (argc >= 2 && !strcmp(argv[1], "--router-weights-one-canary")) {
        return ds4_gpu_mtl4_router_weights_one_canary() ? 0 : 1;
    }
    /* --topk-mask-canary [ne0 [ne1]] : silv 2026-05-27 task #672. MTL4 port
     * of kernel_dsv4_topk_mask (metal/dsv4_misc.metal:346). Verifies dst
     * filled with -INFINITY at every position. */
    if (argc >= 2 && !strcmp(argv[1], "--topk-mask-canary")) {
        const uint32_t ne0 = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 256;
        const uint32_t ne1 = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 16;
        return ds4_gpu_mtl4_topk_mask_canary(ne0, ne1) ? 0 : 1;
    }
    /* --topk-mask-scatter-canary [n_topk [n_tokens [n_comp]]] : silv 2026-05-27
     * task #673. MTL4 port of kernel_dsv4_topk_mask_scatter
     * (metal/dsv4_misc.metal:366). Scatters 0.0 at selected (idx, token)
     * positions in dst mask. Expected 3 zeroed per token (3 valid topk indices). */
    if (argc >= 2 && !strcmp(argv[1], "--topk-mask-scatter-canary")) {
        const uint32_t n_topk = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 6;
        const uint32_t n_tokens = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 16;
        const uint32_t n_comp = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 64;
        return ds4_gpu_mtl4_topk_mask_scatter_canary(n_topk, n_tokens, n_comp) ? 0 : 1;
    }
    /* --indexer-weighted-sum-canary [ne0 [ne1 [n_heads]]] : task #674
     * MTL4 port of kernel_dsv4_indexer_weighted_sum (dsv4_misc.metal:1340). */
    if (argc >= 2 && !strcmp(argv[1], "--indexer-weighted-sum-canary")) {
        const uint32_t ne0 = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 8;
        const uint32_t ne1 = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 16;
        const uint32_t n_heads = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 64;
        return ds4_gpu_mtl4_indexer_weighted_sum_canary(ne0, ne1, n_heads) ? 0 : 1;
    }
    /* --dir-steering-canary [width [rows]] : task #675 MTL4 port of
     * kernel_dsv4_directional_steering_project_f32 (dsv4_misc.metal:106). */
    if (argc >= 2 && !strcmp(argv[1], "--dir-steering-canary")) {
        const uint32_t width = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 4096;
        const uint32_t rows = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 16;
        return ds4_gpu_mtl4_dir_steering_canary(width, rows) ? 0 : 1;
    }
    /* --sort-i32-rows-canary [top_k [n_rows]] : task #676 MTL4 port of
     * kernel_dsv4_sort_i32_rows_asc (dsv4_misc.metal:388). top_k must be
     * power-of-2 and ≤ 256. */
    if (argc >= 2 && !strcmp(argv[1], "--sort-i32-rows-canary")) {
        const uint32_t top_k = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t n_rows = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        return ds4_gpu_mtl4_sort_i32_rows_canary(top_k, n_rows) ? 0 : 1;
    }
    /* --router-remap-canary [n_tokens] : task #677 MTL4 port of
     * kernel_dsv4_router_weights_with_remap (dsv4_misc.metal:210). Identity
     * remap; verifies weights renormalize to 0.25 per slot. */
    if (argc >= 2 && !strcmp(argv[1], "--router-remap-canary")) {
        const uint32_t n_tokens = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 4;
        return ds4_gpu_mtl4_router_remap_canary(n_tokens) ? 0 : 1;
    }
    /* --ratio4-shift-canary [width] : task #678 MTL4 port of
     * kernel_dsv4_ratio4_shift_f32 (dsv4_kv.metal:271). Tiny KV
     * ratio-4 state shift on 4*width elements. */
    if (argc >= 2 && !strcmp(argv[1], "--ratio4-shift-canary")) {
        const uint32_t width = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 128;
        return ds4_gpu_mtl4_ratio4_shift_canary(width) ? 0 : 1;
    }
    /* --amortized-canary [n_iter] : task #679 demonstrates ArgumentTable
     * pool amortization. Runs N back-to-back dispatches using the pool;
     * reports alloc_count vs acquire_count. Pool-hit rate should be
     * (N-1)/N after warm-up (1 alloc, N acquires). */
    if (argc >= 2 && !strcmp(argv[1], "--amortized-canary")) {
        const uint32_t n_iter = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 100;
        return ds4_gpu_mtl4_router_weights_one_amortized_canary(n_iter) ? 0 : 1;
    }
    /* --compressor-store-one-canary [width] : task #680 MTL4 port. */
    if (argc >= 2 && !strcmp(argv[1], "--compressor-store-one-canary")) {
        const uint32_t width = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 256;
        return ds4_gpu_mtl4_compressor_store_one_canary(width) ? 0 : 1;
    }
    /* --softmax-pool-canary [ne0 [ne1 [n_rows]]] : task #681 MTL4 port. */
    if (argc >= 2 && !strcmp(argv[1], "--softmax-pool-canary")) {
        const uint32_t ne0 = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 16;
        const uint32_t ne1 = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 8;
        const uint32_t n_rows = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 32;
        return ds4_gpu_mtl4_softmax_pool_canary(ne0, ne1, n_rows) ? 0 : 1;
    }
    /* --moe-matmul-init-canary : task #682 MTL4 port of
     * kernel_mul_mm_id_fp16_pair_swiglu_f32 (moe.metal:1282). Pipeline-
     * init smoke test; full output canary multi-turn. */
    if (argc >= 2 && !strcmp(argv[1], "--moe-matmul-init-canary")) {
        return ds4_gpu_mtl4_moe_matmul_init_canary() ? 0 : 1;
    }
    /* --moe-matmul-full-canary : task #682 full output validation.
     * 1-expert × 8-token × 16K test bench with identity weights. */
    if (argc >= 2 && !strcmp(argv[1], "--moe-matmul-full-canary")) {
        return ds4_gpu_mtl4_moe_matmul_full_canary() ? 0 : 1;
    }
    /* --router-finalize-one-canary [has_bias=0|1] : task #684 */
    if (argc >= 2 && !strcmp(argv[1], "--router-finalize-one-canary")) {
        const int has_bias = (argc >= 3) ? atoi(argv[2]) : 0;
        return ds4_gpu_mtl4_router_finalize_one_canary(has_bias) ? 0 : 1;
    }
    /* --qkv-rms-norm-canary [q_n [kv_n]] : task #685 per-layer Q+KV RMSNorm */
    if (argc >= 2 && !strcmp(argv[1], "--qkv-rms-norm-canary")) {
        const uint32_t q_n = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 1024;
        const uint32_t kv_n = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 512;
        return ds4_gpu_mtl4_qkv_rms_norm_canary(q_n, kv_n) ? 0 : 1;
    }
    /* --soft-max-4-canary [n] : task #678 row-softmax float4 vectorized */
    if (argc >= 2 && !strcmp(argv[1], "--soft-max-4-canary")) {
        const uint32_t n = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 512;
        return ds4_gpu_mtl4_soft_max_4_canary(n) ? 0 : 1;
    }
    /* --hc-expand4-canary [n_embd [n_tokens]] : task #679 HC=4 block-expand */
    if (argc >= 2 && !strcmp(argv[1], "--hc-expand4-canary")) {
        const uint32_t n_embd = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 128;
        const uint32_t n_tokens = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 4;
        return ds4_gpu_mtl4_hc_expand4_canary(n_embd, n_tokens) ? 0 : 1;
    }
    /* --indexer-score-one-direct-canary [n_comp] : task #680 decode-only DS4 indexer */
    if (argc >= 2 && !strcmp(argv[1], "--indexer-score-one-direct-canary")) {
        const uint32_t n_comp = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 32;
        return ds4_gpu_mtl4_indexer_score_one_direct_canary(n_comp) ? 0 : 1;
    }
    /* --soft-max-canary [n] : task #683 non-vectorized softmax */
    if (argc >= 2 && !strcmp(argv[1], "--soft-max-canary")) {
        const uint32_t n = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 511;  /* odd width to differ from soft_max_4 */
        return ds4_gpu_mtl4_soft_max_canary(n) ? 0 : 1;
    }
    /* --fp8-kv-quantize-canary [n_rows [n_full [n_rot]]] : task #684 FP8 KV round-trip */
    if (argc >= 2 && !strcmp(argv[1], "--fp8-kv-quantize-canary")) {
        const uint32_t n_rows = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 4;
        const uint32_t n_full = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 192;
        const uint32_t n_rot = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 64;
        return ds4_gpu_mtl4_fp8_kv_quantize_canary(n_rows, n_full, n_rot) ? 0 : 1;
    }
    /* --indexer-hadamard-fp4-canary [n_rows] : task #685b Walsh-Hadamard + FP4 */
    if (argc >= 2 && !strcmp(argv[1], "--indexer-hadamard-fp4-canary")) {
        const uint32_t n_rows = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 8;
        return ds4_gpu_mtl4_indexer_hadamard_fp4_canary(n_rows) ? 0 : 1;
    }
    /* --kv-fp8-store-canary [head_dim [n_rot]] : task #686 KV finalizer + FP16 mirror */
    if (argc >= 2 && !strcmp(argv[1], "--kv-fp8-store-canary")) {
        const uint32_t head_dim = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 192;
        const uint32_t n_rot = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 64;
        return ds4_gpu_mtl4_kv_fp8_store_canary(head_dim, n_rot) ? 0 : 1;
    }
    /* --moe-swiglu-weight-canary [rows [width]] : task #687 routed MoE activation */
    if (argc >= 2 && !strcmp(argv[1], "--moe-swiglu-weight-canary")) {
        const uint32_t rows = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 8;
        const uint32_t width = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        return ds4_gpu_mtl4_moe_swiglu_weight_canary(rows, width) ? 0 : 1;
    }
    /* --moe-sum6-canary [tokens [width]] : task #688 6-buffer MoE finalize */
    if (argc >= 2 && !strcmp(argv[1], "--moe-sum6-canary")) {
        const uint32_t tokens = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 4;
        const uint32_t width = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 512;
        return ds4_gpu_mtl4_moe_sum6_canary(tokens, width) ? 0 : 1;
    }
    /* --moe-swiglu-weight-f16-canary [rows [width]] : task #689 FP16 mid variant */
    if (argc >= 2 && !strcmp(argv[1], "--moe-swiglu-weight-f16-canary")) {
        const uint32_t rows = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 8;
        const uint32_t width = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        return ds4_gpu_mtl4_moe_swiglu_weight_f16_canary(rows, width) ? 0 : 1;
    }
    /* --hc-split-sinkhorn-canary [n_rows [iters]] : task #690 Sinkhorn 4×4 */
    if (argc >= 2 && !strcmp(argv[1], "--hc-split-sinkhorn-canary")) {
        const uint32_t n_rows = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t iters = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 1;
        return ds4_gpu_mtl4_hc_split_sinkhorn_canary(n_rows, iters) ? 0 : 1;
    }
    /* --get-rows-f32-canary [n_table_rows [row_width [n_ids]]] : task #691 */
    if (argc >= 2 && !strcmp(argv[1], "--get-rows-f32-canary")) {
        const uint32_t ntab = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 32;
        const uint32_t rw   = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 128;
        const uint32_t nid  = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 8;
        return ds4_gpu_mtl4_get_rows_f32_canary(ntab, rw, nid) ? 0 : 1;
    }
    /* --argsort-f32-i32-desc-canary [row_n [top_k]] : task #692 bitonic argsort */
    if (argc >= 2 && !strcmp(argv[1], "--argsort-f32-i32-desc-canary")) {
        const uint32_t row_n = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t top_k = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 8;
        return ds4_gpu_mtl4_argsort_f32_i32_desc_canary(row_n, top_k) ? 0 : 1;
    }
    /* --cpy-f32-f32-canary [n_rows [row_width]] : task #693 typed copy */
    if (argc >= 2 && !strcmp(argv[1], "--cpy-f32-f32-canary")) {
        const uint32_t nr = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 16;
        const uint32_t rw = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        return ds4_gpu_mtl4_cpy_f32_f32_canary(nr, rw) ? 0 : 1;
    }
    /* --hc-split-weighted-sum-canary [n_rows [n_embd]] : task #694 fused HC */
    if (argc >= 2 && !strcmp(argv[1], "--hc-split-weighted-sum-canary")) {
        const uint32_t nr = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 16;
        const uint32_t ne = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 512;
        return ds4_gpu_mtl4_hc_split_weighted_sum_canary(nr, ne) ? 0 : 1;
    }
    /* --rope-tail-f32-canary [head_dim [n_rot [mode]]] : task #695 partial RoPE+YaRN */
    if (argc >= 2 && !strcmp(argv[1], "--rope-tail-f32-canary")) {
        const uint32_t hd = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 192;
        const uint32_t nr = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 64;
        const uint32_t m  = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 2;  /* default NeoX */
        return ds4_gpu_mtl4_rope_tail_f32_canary(hd, nr, m) ? 0 : 1;
    }
    /* --concat-canary [n0 [n1 [n_rows]]] : task #696 concat along dim 0 */
    if (argc >= 2 && !strcmp(argv[1], "--concat-canary")) {
        const uint32_t n0 = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t n1 = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 128;
        const uint32_t nr = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 8;
        return ds4_gpu_mtl4_concat_canary(n0, n1, nr) ? 0 : 1;
    }
    /* --hc-split-weighted-sum-norm4-canary [n_rows] : task #697 fused HC+RMSNorm at 4096 */
    if (argc >= 2 && !strcmp(argv[1], "--hc-split-weighted-sum-norm4-canary")) {
        const uint32_t nr = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 8;
        return ds4_gpu_mtl4_hc_split_weighted_sum_norm4_canary(nr) ? 0 : 1;
    }
    /* --hc-expand-canary [n_embd [n_hc [n_tokens]]] : task #698 generic-HC variant */
    if (argc >= 2 && !strcmp(argv[1], "--hc-expand-canary")) {
        const uint32_t ne = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 128;
        const uint32_t nh = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 8;
        const uint32_t nt = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 4;
        return ds4_gpu_mtl4_hc_expand_canary(ne, nh, nt) ? 0 : 1;
    }
    /* --hadamard16-wide-canary [n_rows [blocks_per_row]] : task #699 wide batched Hadamard */
    if (argc >= 2 && !strcmp(argv[1], "--hadamard16-wide-canary")) {
        const uint32_t nr = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 4;
        const uint32_t bpr = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        return ds4_gpu_mtl4_hadamard16_wide_canary(nr, bpr) ? 0 : 1;
    }
    /* --argsort-merge-desc-canary [len [top_k]] : task #700 multi-block merge */
    if (argc >= 2 && !strcmp(argv[1], "--argsort-merge-desc-canary")) {
        const uint32_t len = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 32;
        const uint32_t top_k = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 16;
        return ds4_gpu_mtl4_argsort_merge_f32_i32_desc_canary(len, top_k) ? 0 : 1;
    }
    /* --cpy-f32-f16-canary [n_rows [row_width]] : task #702 f32→f16 typed copy */
    if (argc >= 2 && !strcmp(argv[1], "--cpy-f32-f16-canary")) {
        const uint32_t nr = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 16;
        const uint32_t rw = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        return ds4_gpu_mtl4_cpy_f32_f16_canary(nr, rw) ? 0 : 1;
    }
    /* --cpy-f16-f32-canary [n_rows [row_width]] : task #703 f16→f32 typed copy */
    if (argc >= 2 && !strcmp(argv[1], "--cpy-f16-f32-canary")) {
        const uint32_t nr = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 16;
        const uint32_t rw = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        return ds4_gpu_mtl4_cpy_f16_f32_canary(nr, rw) ? 0 : 1;
    }
    /* --sum-rows-canary [n_rows [row_width]] : task #704 row-sum reduction */
    if (argc >= 2 && !strcmp(argv[1], "--sum-rows-canary")) {
        const uint32_t nr = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 16;
        const uint32_t rw = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        return ds4_gpu_mtl4_sum_rows_f32_f32_canary(nr, rw) ? 0 : 1;
    }
    /* --set-rows-canary [n_src [row_width]] : task #705 KV-cache scatter */
    if (argc >= 2 && !strcmp(argv[1], "--set-rows-canary")) {
        const uint32_t nr = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 16;
        const uint32_t rw = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        return ds4_gpu_mtl4_set_rows_f32_i32_canary(nr, rw) ? 0 : 1;
    }
    /* --map0-ne20-8-canary [n_experts [n_tokens]] : task #706 MoE routing-table builder */
    if (argc >= 2 && !strcmp(argv[1], "--map0-ne20-8-canary")) {
        const uint32_t ne = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t nt = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        return ds4_gpu_mtl4_mul_mm_id_map0_ne20_8_canary(ne, nt) ? 0 : 1;
    }
    /* --repeat-canary [src_r [src_c [r_fac [c_fac]]]] : task #707 broadcast kernel */
    if (argc >= 2 && !strcmp(argv[1], "--repeat-canary")) {
        const uint32_t sr = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 8;
        const uint32_t sc = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 16;
        const uint32_t rf = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 4;
        const uint32_t cf = (argc >= 6) ? (uint32_t)atoi(argv[5]) : 2;
        return ds4_gpu_mtl4_repeat_f32_canary(sr, sc, rf, cf) ? 0 : 1;
    }
    /* --swiglu-canary [n_rows [row_width [alpha [limit]]]] : task #708 SwiGLU activation */
    if (argc >= 2 && !strcmp(argv[1], "--swiglu-canary")) {
        const uint32_t nr = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 8;
        const uint32_t rw = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 64;
        const float al = (argc >= 5) ? (float)atof(argv[4]) : 1.0f;
        const float li = (argc >= 6) ? (float)atof(argv[5]) : 0.0f;
        return ds4_gpu_mtl4_swiglu_f32_canary(nr, rw, al, li) ? 0 : 1;
    }
    /* --rms-norm-mul-canary [n_rows [row_width [eps]]] : task #709 RMSNorm + weight */
    if (argc >= 2 && !strcmp(argv[1], "--rms-norm-mul-canary")) {
        const uint32_t nr = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 8;
        const uint32_t rw = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        const float ep = (argc >= 5) ? (float)atof(argv[4]) : 1.0e-5f;
        return ds4_gpu_mtl4_rms_norm_mul_f32_4_canary(nr, rw, ep) ? 0 : 1;
    }
    /* --rms-norm-canary [n_rows [row_width [eps]]] : task #710 RMSNorm only */
    if (argc >= 2 && !strcmp(argv[1], "--rms-norm-canary")) {
        const uint32_t nr = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 8;
        const uint32_t rw = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        const float ep = (argc >= 5) ? (float)atof(argv[4]) : 1.0e-5f;
        return ds4_gpu_mtl4_rms_norm_f32_4_canary(nr, rw, ep) ? 0 : 1;
    }
    /* --get-rows-f16-canary [n_table [row_width [n_ids]]] : task #711 f16 embedding lookup */
    if (argc >= 2 && !strcmp(argv[1], "--get-rows-f16-canary")) {
        const uint32_t nt = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 32;
        const uint32_t rw = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 128;
        const uint32_t ni = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 8;
        return ds4_gpu_mtl4_get_rows_f16_canary(nt, rw, ni) ? 0 : 1;
    }
    /* --get-rows-i32-canary [n_table [row_width [n_ids]]] : task #712 i32 lookup */
    if (argc >= 2 && !strcmp(argv[1], "--get-rows-i32-canary")) {
        const uint32_t nt = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 32;
        const uint32_t rw = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 128;
        const uint32_t ni = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 8;
        return ds4_gpu_mtl4_get_rows_i32_canary(nt, rw, ni) ? 0 : 1;
    }
    /* --bin-add-canary [n_rows [row_width]] : task #713 residual-add */
    if (argc >= 2 && !strcmp(argv[1], "--bin-add-canary")) {
        const uint32_t nr = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 16;
        const uint32_t rw = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        return ds4_gpu_mtl4_bin_fuse_add_f32_canary(nr, rw) ? 0 : 1;
    }
    /* --bin-sub-canary [n_rows [row_width]] : task #714 subtract */
    if (argc >= 2 && !strcmp(argv[1], "--bin-sub-canary")) {
        const uint32_t nr = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 16;
        const uint32_t rw = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        return ds4_gpu_mtl4_bin_fuse_sub_f32_canary(nr, rw) ? 0 : 1;
    }
    /* --bin-mul-canary [n_rows [row_width]] : task #715 multiply */
    if (argc >= 2 && !strcmp(argv[1], "--bin-mul-canary")) {
        const uint32_t nr = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 16;
        const uint32_t rw = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        return ds4_gpu_mtl4_bin_fuse_mul_f32_canary(nr, rw) ? 0 : 1;
    }
    /* --bin-div-canary [n_rows [row_width]] : task #716 divide */
    if (argc >= 2 && !strcmp(argv[1], "--bin-div-canary")) {
        const uint32_t nr = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 16;
        const uint32_t rw = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        return ds4_gpu_mtl4_bin_fuse_div_f32_canary(nr, rw) ? 0 : 1;
    }
    /* --bin-add-cb-canary [n_rows [row_width]] : task #717 bias-add */
    if (argc >= 2 && !strcmp(argv[1], "--bin-add-cb-canary")) {
        const uint32_t nr = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 16;
        const uint32_t rw = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        return ds4_gpu_mtl4_bin_fuse_add_cb_f32_canary(nr, rw) ? 0 : 1;
    }
    /* --bin-mul-cb-canary [n_rows [row_width]] : task #718 per-channel scale */
    if (argc >= 2 && !strcmp(argv[1], "--bin-mul-cb-canary")) {
        const uint32_t nr = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 16;
        const uint32_t rw = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        return ds4_gpu_mtl4_bin_fuse_mul_cb_f32_canary(nr, rw) ? 0 : 1;
    }
    /* --map0-ne20-{10,16,22}-canary [n_experts [n_tokens]] : tasks #719/#720/#721 */
    if (argc >= 2 && !strcmp(argv[1], "--map0-ne20-10-canary")) {
        const uint32_t ne = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t nt = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        return ds4_gpu_mtl4_mul_mm_id_map0_ne20_10_canary(ne, nt) ? 0 : 1;
    }
    if (argc >= 2 && !strcmp(argv[1], "--map0-ne20-16-canary")) {
        const uint32_t ne = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t nt = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        return ds4_gpu_mtl4_mul_mm_id_map0_ne20_16_canary(ne, nt) ? 0 : 1;
    }
    if (argc >= 2 && !strcmp(argv[1], "--map0-ne20-22-canary")) {
        const uint32_t ne = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t nt = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        return ds4_gpu_mtl4_mul_mm_id_map0_ne20_22_canary(ne, nt) ? 0 : 1;
    }
    /* --map0-ne20-{1,2,4,5,6}-canary [n_experts [n_tokens]] : tasks #729-#733 */
    if (argc >= 2 && !strcmp(argv[1], "--map0-ne20-1-canary")) {
        const uint32_t ne = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t nt = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        return ds4_gpu_mtl4_mul_mm_id_map0_ne20_1_canary(ne, nt) ? 0 : 1;
    }
    if (argc >= 2 && !strcmp(argv[1], "--map0-ne20-2-canary")) {
        const uint32_t ne = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t nt = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        return ds4_gpu_mtl4_mul_mm_id_map0_ne20_2_canary(ne, nt) ? 0 : 1;
    }
    if (argc >= 2 && !strcmp(argv[1], "--map0-ne20-4-canary")) {
        const uint32_t ne = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t nt = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        return ds4_gpu_mtl4_mul_mm_id_map0_ne20_4_canary(ne, nt) ? 0 : 1;
    }
    if (argc >= 2 && !strcmp(argv[1], "--map0-ne20-5-canary")) {
        const uint32_t ne = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t nt = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        return ds4_gpu_mtl4_mul_mm_id_map0_ne20_5_canary(ne, nt) ? 0 : 1;
    }
    if (argc >= 2 && !strcmp(argv[1], "--map0-ne20-6-canary")) {
        const uint32_t ne = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t nt = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        return ds4_gpu_mtl4_mul_mm_id_map0_ne20_6_canary(ne, nt) ? 0 : 1;
    }
    /* --mul-mv-canary [M [N]] : task #722 first FC-aware MTL4 port (f32 matvec) */
    if (argc >= 2 && !strcmp(argv[1], "--mul-mv-canary")) {
        const uint32_t m = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t n = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        return ds4_gpu_mtl4_mul_mv_f32_f32_canary(m, n) ? 0 : 1;
    }
    /* --mul-mv-q8-0-canary [M [N]] : task #723 Q8_0 matvec */
    if (argc >= 2 && !strcmp(argv[1], "--mul-mv-q8-0-canary")) {
        const uint32_t m = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t n = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        return ds4_gpu_mtl4_mul_mv_q8_0_f32_canary(m, n) ? 0 : 1;
    }
    /* --gate-up-swiglu-q8-0-canary [M [N [clamp]]] : task #724 fused Q8_0 SwiGLU */
    if (argc >= 2 && !strcmp(argv[1], "--gate-up-swiglu-q8-0-canary")) {
        const uint32_t m = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t n = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        const float cv = (argc >= 5) ? (float)atof(argv[4]) : 0.0f;
        return ds4_gpu_mtl4_dsv4_shared_gate_up_swiglu_q8_0_canary(m, n, cv) ? 0 : 1;
    }
    /* --q8-hc-expand4-canary [M [N]] : task #725 Q8_0 matvec + 4-channel HC expand */
    if (argc >= 2 && !strcmp(argv[1], "--q8-hc-expand4-canary")) {
        const uint32_t m = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t n = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        return ds4_gpu_mtl4_dsv4_q8_hc_expand4_q8_0_canary(m, n) ? 0 : 1;
    }
    /* --mul-mv-f16-canary [M [N]] : task #727 FP16 matvec */
    if (argc >= 2 && !strcmp(argv[1], "--mul-mv-f16-canary")) {
        const uint32_t m = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t n = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        return ds4_gpu_mtl4_mul_mv_f16_f32_canary(m, n) ? 0 : 1;
    }
    /* --shared-down-hc-expand4-canary [M [N]] : task #728 */
    if (argc >= 2 && !strcmp(argv[1], "--shared-down-hc-expand4-canary")) {
        const uint32_t m = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t n = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        return ds4_gpu_mtl4_dsv4_shared_down_hc_expand4_q8_0_canary(m, n) ? 0 : 1;
    }
    /* --lane-diag [nthreads] : MTL4 lane-execution diagnostic */
    if (argc >= 2 && !strcmp(argv[1], "--lane-diag")) {
        const uint32_t n = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 128;
        return ds4_gpu_mtl4_lane_diag_canary(n) ? 0 : 1;
    }
    /* --mma-iso [n_simdgroups] : MTL4 simdgroup MMA isolation test */
    if (argc >= 2 && !strcmp(argv[1], "--mma-iso")) {
        const uint32_t n = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 1;
        return ds4_gpu_mtl4_mma_iso_canary(n) ? 0 : 1;
    }
    /* --indexer-scores-tiled-canary [n_tokens [n_comp [n_head]]] : #730 unblocks #701 */
    if (argc >= 2 && !strcmp(argv[1], "--indexer-scores-tiled-canary")) {
        const uint32_t nt = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 8;
        const uint32_t nc = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        const uint32_t nh = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 1;
        return ds4_gpu_mtl4_indexer_scores_tiled_f32_canary(nt, nc, nh) ? 0 : 1;
    }
    /* --indexer-scores-tiled-half-canary : #731 half-precision variant */
    if (argc >= 2 && !strcmp(argv[1], "--indexer-scores-tiled-half-canary")) {
        const uint32_t nt = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 8;
        const uint32_t nc = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        const uint32_t nh = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 1;
        return ds4_gpu_mtl4_indexer_scores_tiled_canary(nt, nc, nh) ? 0 : 1;
    }
    /* --mul-mm-f16-canary [M [N [K]]] : #732 dense FP16 matmul */
    if (argc >= 2 && !strcmp(argv[1], "--mul-mm-f16-canary")) {
        const uint32_t m = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t n = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        const uint32_t k = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 64;
        return ds4_gpu_mtl4_mul_mm_f16_f32_canary(m, n, k) ? 0 : 1;
    }
    /* --mul-mm-q8-0-canary [M [N [K]]] : #733 Q8_0 dense matmul */
    if (argc >= 2 && !strcmp(argv[1], "--mul-mm-q8-0-canary")) {
        const uint32_t m = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t n = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        const uint32_t k = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 64;
        return ds4_gpu_mtl4_mul_mm_q8_0_f32_canary(m, n, k) ? 0 : 1;
    }
    /* --mul-mm-id-q8-0-canary [M [N [K [E]]]] : #734 Q8_0 routed MoE matmul */
    if (argc >= 2 && !strcmp(argv[1], "--mul-mm-id-q8-0-canary")) {
        const uint32_t m = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t n = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        const uint32_t k = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 64;
        const uint32_t e = (argc >= 6) ? (uint32_t)atoi(argv[5]) : 2;
        return ds4_gpu_mtl4_mul_mm_id_q8_0_f32_canary(m, n, k, e) ? 0 : 1;
    }
    /* --mul-mm-id-iq2-xxs-canary [M [N [K [E]]]] : #735 IQ2_XXS routed MoE matmul */
    if (argc >= 2 && !strcmp(argv[1], "--mul-mm-id-iq2-xxs-canary")) {
        const uint32_t m = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t n = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        const uint32_t k = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 256;
        const uint32_t e = (argc >= 6) ? (uint32_t)atoi(argv[5]) : 2;
        return ds4_gpu_mtl4_mul_mm_id_iq2_xxs_f32_canary(m, n, k, e) ? 0 : 1;
    }
    /* --mul-mm-id-iq2-xxs-n64-canary [M [N [K [E]]]] : #739 IQ2_XXS routed n64 */
    if (argc >= 2 && !strcmp(argv[1], "--mul-mm-id-iq2-xxs-n64-canary")) {
        const uint32_t m = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t n = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        const uint32_t k = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 256;
        const uint32_t e = (argc >= 6) ? (uint32_t)atoi(argv[5]) : 2;
        return ds4_gpu_mtl4_mul_mm_id_iq2_xxs_f32_n64_canary(m, n, k, e) ? 0 : 1;
    }
    /* --mul-mm-id-iq2-xxs-n128-canary [M [N [K [E]]]] : #740 IQ2_XXS routed n128 */
    if (argc >= 2 && !strcmp(argv[1], "--mul-mm-id-iq2-xxs-n128-canary")) {
        const uint32_t m = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t n = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        const uint32_t k = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 256;
        const uint32_t e = (argc >= 6) ? (uint32_t)atoi(argv[5]) : 2;
        return ds4_gpu_mtl4_mul_mm_id_iq2_xxs_f32_n128_canary(m, n, k, e) ? 0 : 1;
    }
    /* #742-#747 wide-tile canaries for Q8_0 / Q4_K / Q2_K × {n64, n128} */
    if (argc >= 2 && !strcmp(argv[1], "--mul-mm-id-q8-0-n64-canary")) {
        const uint32_t m = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t n = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        const uint32_t k = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 64;
        const uint32_t e = (argc >= 6) ? (uint32_t)atoi(argv[5]) : 2;
        return ds4_gpu_mtl4_mul_mm_id_q8_0_f32_n64_canary(m, n, k, e) ? 0 : 1;
    }
    if (argc >= 2 && !strcmp(argv[1], "--mul-mm-id-q8-0-n128-canary")) {
        const uint32_t m = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t n = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        const uint32_t k = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 64;
        const uint32_t e = (argc >= 6) ? (uint32_t)atoi(argv[5]) : 2;
        return ds4_gpu_mtl4_mul_mm_id_q8_0_f32_n128_canary(m, n, k, e) ? 0 : 1;
    }
    if (argc >= 2 && !strcmp(argv[1], "--mul-mm-id-q4-k-n64-canary")) {
        const uint32_t m = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t n = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        const uint32_t k = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 256;
        const uint32_t e = (argc >= 6) ? (uint32_t)atoi(argv[5]) : 2;
        return ds4_gpu_mtl4_mul_mm_id_q4_K_f32_n64_canary(m, n, k, e) ? 0 : 1;
    }
    if (argc >= 2 && !strcmp(argv[1], "--mul-mm-id-q4-k-n128-canary")) {
        const uint32_t m = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t n = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        const uint32_t k = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 256;
        const uint32_t e = (argc >= 6) ? (uint32_t)atoi(argv[5]) : 2;
        return ds4_gpu_mtl4_mul_mm_id_q4_K_f32_n128_canary(m, n, k, e) ? 0 : 1;
    }
    if (argc >= 2 && !strcmp(argv[1], "--mul-mm-id-q2-k-n64-canary")) {
        const uint32_t m = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t n = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        const uint32_t k = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 256;
        const uint32_t e = (argc >= 6) ? (uint32_t)atoi(argv[5]) : 2;
        return ds4_gpu_mtl4_mul_mm_id_q2_K_f32_n64_canary(m, n, k, e) ? 0 : 1;
    }
    if (argc >= 2 && !strcmp(argv[1], "--mul-mm-id-q2-k-n128-canary")) {
        const uint32_t m = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t n = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        const uint32_t k = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 256;
        const uint32_t e = (argc >= 6) ? (uint32_t)atoi(argv[5]) : 2;
        return ds4_gpu_mtl4_mul_mm_id_q2_K_f32_n128_canary(m, n, k, e) ? 0 : 1;
    }
    /* #742 wide-tile audit: routes R tokens to 1 expert; tests all 3 widths */
    if (argc >= 2 && !strcmp(argv[1], "--wide-tile-audit")) {
        const uint32_t m = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t k = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        const uint32_t r = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 64;
        return ds4_gpu_mtl4_wide_tile_audit_iq2_xxs(m, k, r) ? 0 : 1;
    }
    /* Classic Metal counterpart — tests antirez upstream kernels */
    if (argc >= 2 && !strcmp(argv[1], "--wide-tile-audit-classic")) {
        const uint32_t m = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t k = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        const uint32_t r = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 64;
        return ds4_gpu_classic_wide_tile_audit_iq2_xxs(m, k, r) ? 0 : 1;
    }
    /* Per-quant wide-tile audits — verify n64/n128 fixes work at R>32 */
    if (argc >= 2 && !strcmp(argv[1], "--wide-tile-audit-q8-0")) {
        const uint32_t m = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t k = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 64;
        const uint32_t r = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 64;
        return ds4_gpu_mtl4_wide_tile_audit_q8_0(m, k, r) ? 0 : 1;
    }
    if (argc >= 2 && !strcmp(argv[1], "--wide-tile-audit-q4-k")) {
        const uint32_t m = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t k = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        const uint32_t r = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 64;
        return ds4_gpu_mtl4_wide_tile_audit_q4_K(m, k, r) ? 0 : 1;
    }
    if (argc >= 2 && !strcmp(argv[1], "--wide-tile-audit-q2-k")) {
        const uint32_t m = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t k = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        const uint32_t r = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 64;
        return ds4_gpu_mtl4_wide_tile_audit_q2_K(m, k, r) ? 0 : 1;
    }
    /* --mul-mm-id-q4-k-canary [M [N [K [E]]]] : #736 Q4_K routed MoE matmul */
    if (argc >= 2 && !strcmp(argv[1], "--mul-mm-id-q4-k-canary")) {
        const uint32_t m = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t n = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        const uint32_t k = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 256;
        const uint32_t e = (argc >= 6) ? (uint32_t)atoi(argv[5]) : 2;
        return ds4_gpu_mtl4_mul_mm_id_q4_K_f32_canary(m, n, k, e) ? 0 : 1;
    }
    /* --mul-mm-id-q2-k-canary [M [N [K [E]]]] : #737 Q2_K routed MoE matmul */
    if (argc >= 2 && !strcmp(argv[1], "--mul-mm-id-q2-k-canary")) {
        const uint32_t m = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t n = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32;
        const uint32_t k = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 256;
        const uint32_t e = (argc >= 6) ? (uint32_t)atoi(argv[5]) : 2;
        return ds4_gpu_mtl4_mul_mm_id_q2_K_f32_canary(m, n, k, e) ? 0 : 1;
    }
    /* --routed-mm-dispatch-probe : silv 2026-05-28 I-1 wiring smoke test.
     * Pair with DS4_METAL_LOG_ROUTED_MM=1 to see per-pick log lines confirming
     * the MTL4 redirect for wide-tile (n64/n128) fires correctly. */
    if (argc >= 2 && !strcmp(argv[1], "--routed-mm-dispatch-probe")) {
        return ds4_gpu_mtl4_routed_mm_dispatch_probe() ? 0 : 1;
    }
    /* --vqb2-pack-probe <pack> <index_csv> : silv 2026-05-28
     * Opens pack + index, verifies summary, materializes 5 views from across
     * the pack range, decodes a sample pair, prints results. Smoke test for
     * the pack-backed VQB2 view loader (codex H2116-H2125). */
    if (argc >= 2 && !strcmp(argv[1], "--vqb2-pack-probe")) {
        if (argc < 4) {
            fprintf(stderr, "usage: --vqb2-pack-probe <pack> <index_csv>\n");
            return 1;
        }
        extern int ds4_cli_vqb2_pack_probe(const char *pack, const char *index);
        return ds4_cli_vqb2_pack_probe(argv[2], argv[3]);
    }
    /* --vqb2-pack-layer-tour <pack> <index_csv> <layer> : silv 2026-05-28
     * H2125 layer-major iterator smoke test. Walks all 64 packets for the
     * given layer in (kind, row_start) order, decodes one pair per packet.
     * Expected: 16 gate + 16 up + 32 down = 64 packets, 64 successful decodes. */
    if (argc >= 2 && !strcmp(argv[1], "--vqb2-pack-layer-tour")) {
        if (argc < 5) {
            fprintf(stderr, "usage: --vqb2-pack-layer-tour <pack> <index_csv> <layer>\n");
            return 1;
        }
        extern int ds4_cli_vqb2_pack_layer_tour(const char *pack, const char *index, uint32_t layer);
        return ds4_cli_vqb2_pack_layer_tour(argv[2], argv[3], (uint32_t)atoi(argv[4]));
    }
    /* --vqb2-decode-fp16-canary [N [K]] : silv 2026-05-28 GPU-side VQB2 decoder.
     * Default N=4096, K=16. K∈{4,16,64,256}. Synthetic packet → GPU decode →
     * compare against ground truth. Foundation for pack-direct dispatch. */
    if (argc >= 2 && !strcmp(argv[1], "--vqb2-decode-fp16-canary")) {
        const uint32_t n = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 4096;
        const uint32_t k = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 16;
        return ds4_gpu_mtl4_vqb2_decode_fp16_canary(n, k) ? 0 : 1;
    }
    /* --vqb2-decode-icb-bench [N_PACKETS [N_CODES [K [ROUNDS]]]] : silv 2026-05-28
     * ICB Phase 8 bench: direct vs ICB record/replay timing. Defaults: 64 packets
     * × 32768 codes (one DS4 layer's gate/up shape) × K=16 × 100 rounds. */
    if (argc >= 2 && !strcmp(argv[1], "--vqb2-decode-icb-bench")) {
        const uint32_t np = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 64;
        const uint32_t nc = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 32768;
        const uint32_t k  = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 16;
        const uint32_t r  = (argc >= 6) ? (uint32_t)atoi(argv[5]) : 100;
        return ds4_gpu_mtl4_vqb2_decode_icb_bench(np, nc, k, r) ? 0 : 1;
    }
    /* --prefix-cache-test : silv 2026-05-27 Phase 1 self-test (cached prefix activations) */
    if (argc >= 2 && !strcmp(argv[1], "--prefix-cache-test")) {
        extern int ds4_prefix_cache_phase1_self_test(void);
        return ds4_prefix_cache_phase1_self_test() ? 0 : 1;
    }
    /* --hc-weighted-sum-canary [n_embd [n_hc [n_tokens]]] : task #683 */
    if (argc >= 2 && !strcmp(argv[1], "--hc-weighted-sum-canary")) {
        const uint32_t n_embd = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 128;
        const uint32_t n_hc = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 4;
        const uint32_t n_tokens = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 8;
        return ds4_gpu_mtl4_hc_weighted_sum_canary(n_embd, n_hc, n_tokens) ? 0 : 1;
    }
    cli_config cfg = parse_options(argc, argv);
    if (cfg.gen.dump_tokens) {
        if (cfg.gen.prompt == NULL) {
            fprintf(stderr, "ds4: --dump-tokens requires -p or --prompt-file\n");
            free(cfg.prompt_owned);
            return 2;
        }
        int rc = ds4_dump_text_tokenization(cfg.engine.model_path,
                                            cfg.gen.prompt,
                                            stdout);
        free(cfg.prompt_owned);
        return rc;
    }
    cfg.engine.inspect_only = cfg.inspect;
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &cfg.engine) != 0) {
        free(cfg.prompt_owned);
        return 1;
    }
    if (!cfg.inspect) {
        log_context_memory(cfg.engine.backend, cfg.gen.ctx_size);
        cli_warn_think_max_downgraded(&cfg.gen, "--think-max");
    }
    int rc = 0;
    if (cfg.inspect) {
        ds4_engine_summary(engine);
    } else if (cfg.gen.imatrix_output_path) {
        rc = ds4_engine_collect_imatrix(engine,
                                        cfg.gen.imatrix_dataset_path,
                                        cfg.gen.imatrix_output_path,
                                        cfg.gen.ctx_size,
                                        cfg.gen.imatrix_max_prompts,
                                        cfg.gen.imatrix_max_tokens);
    } else if (cfg.gen.perplexity_file_path) {
        rc = run_perplexity_file(engine, &cfg);
    } else if (cfg.gen.prompt == NULL) {
        rc = run_repl(engine, &cfg);
    } else {
        rc = run_generation(engine, &cfg);
    }
    ds4_engine_close(engine);
    free(cfg.prompt_owned);
    return rc;
}
