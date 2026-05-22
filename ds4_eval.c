#include "ds4.h"

/* ds4-eval: small built-in benchmark integration test.
 *
 * This program is deliberately not a unit test.  It loads the real model,
 * renders chat prompts, prefills them through ds4_session_sync(), samples the
 * continuation token by token, and grades the final answer.  The terminal UI is
 * also intentionally simple: no ncurses, just ANSI cursor movement, colors, and
 * a fixed two-pane layout.
 *
 * The embedded questions are small fixed subsets of GPQA Diamond, SuperGPQA,
 * AIME 2025, and COMPSEC.  The SuperGPQA slice is intentionally audited: rows
 * with wrong keys, missing figures, or underspecified prompts are replaced
 * instead of being locally re-keyed, because ds4-eval is a regression harness
 * and a bad target is worse than a merely hard target.  COMPSEC contains a
 * small audited subset of reduced C/C++ single-function vulnerability
 * localization questions derived from public CVE writeups; the CVE anchors and
 * private rationales are not rendered to the model.
 * GPQA is released under CC BY 4.0.  SuperGPQA is released under ODC-BY and
 * includes mostly original data plus a limited amount of transformed
 * third-party data.  The AIME 2025 mirror used here is MIT licensed.  Source
 * mirrors used while preparing this file:
 * https://huggingface.co/datasets/Wanfq/gpqa
 * https://huggingface.co/datasets/m-a-p/SuperGPQA
 * https://huggingface.co/datasets/test-time-compute/aime_2025
 */

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#define ANSI_RESET "\x1b[0m"
#define ANSI_DIM "\x1b[90m"
#define ANSI_RED "\x1b[31m"
#define ANSI_GREEN "\x1b[32m"
#define ANSI_YELLOW "\x1b[33m"
#define ANSI_BLUE "\x1b[34m"
#define ANSI_CYAN "\x1b[36m"
#define ANSI_BOLD "\x1b[1m"

#define EVAL_MAX_CHOICES 10
#define EVAL_ANSWER_MAX 32
#define EVAL_MAX_CONTEXT 1000000

typedef enum {
    EVAL_PENDING,
    EVAL_PREFILL,
    EVAL_THINKING,
    EVAL_SKIPPED,
    EVAL_STOPPED,
    EVAL_PASSED,
    EVAL_FAILED,
} eval_status;

typedef enum {
    EVAL_RUN_OK,
    EVAL_RUN_ERROR,
    EVAL_RUN_SWITCH,
    EVAL_RUN_QUIT,
} eval_run_result;

typedef enum {
    EVAL_THINK_CLOSE_NONE,
    EVAL_THINK_CLOSE_NATURAL,
    EVAL_THINK_CLOSE_SOFT,
    EVAL_THINK_CLOSE_HARD,
} eval_think_close_kind;

typedef struct {
    const char *source;
    const char *id;
    const char *domain;
    const char *title;
    const char *question;
    const char *choice[EVAL_MAX_CHOICES];
    const char *answer;
} eval_case;

#include "eval_cases_data.inc.c"

typedef struct {
    char *v;
    size_t len;
    size_t cap;
} byte_buf;

typedef struct {
    unsigned char *v;
    size_t len;
    size_t cap;
} style_buf;

typedef struct {
    const char *model_path;
    const char *mtp_path;
    const char *trace_path;
    ds4_backend backend;
    int threads;
    int ctx_size;
    int max_tokens;
    int question_limit;
    float temperature;
    float top_p;
    float min_p;
    uint64_t seed;
    int pause_ms;
    int soft_limit_reply_budget;
    int hard_limit_reply_budget;
    int soft_limit_think_close_rank;
    ds4_think_mode think_mode;
    bool plain;
    bool warm_weights;
    bool quality;
    bool cpu_moe;
    int  n_cpu_moe_layers;
    int  prefill_metal_phases;
} eval_config;

typedef struct {
    eval_think_close_kind kind;
    int token_index;
    int remaining_budget;
    int rank;
} eval_think_close_info;

typedef struct {
    int cols;
    int rows;
    int left_w;
    int right_x;
    int right_w;
    int body_y;
    int body_h;
    bool active;
    bool enabled;
    const eval_case *cases;
    int ncases;
    eval_status *status;
    char (*guess)[EVAL_ANSWER_MAX];
    int *prompt_tokens;
    int *generated_tokens;
    int active_case;
    int generated;
    int max_tokens;
    int think_max_tokens;
    int prefill_current;
    int prefill_total;
    double phase_start_sec;
    double speed_tps;
    double run_elapsed_sec;
    double run_last_sec;
    bool run_clock_active;
    bool selection_active;
    int selected_case;
    int requested_case;
    bool paused;
    bool quit_requested;
    byte_buf stream;
    style_buf styles;
    bool in_think;
    char pending_tag[16];
    size_t pending_tag_len;
} eval_ui;

static eval_ui *global_ui;

static double run_clock_sec(void);

static void tui_run_clock_tick(eval_ui *ui) {
    if (!ui || !ui->run_clock_active) return;
    double now = run_clock_sec();
    double delta = now - ui->run_last_sec;
    if (delta > 0.0) ui->run_elapsed_sec += delta;
    ui->run_last_sec = now;
}

static void tui_run_clock_start(eval_ui *ui) {
    if (!ui || ui->run_clock_active) return;
    ui->run_last_sec = run_clock_sec();
    ui->run_clock_active = true;
}

static void tui_run_clock_stop(eval_ui *ui) {
    if (!ui || !ui->run_clock_active) return;
    tui_run_clock_tick(ui);
    ui->run_clock_active = false;
}

static double tui_run_clock_visible_sec(const eval_ui *ui) {
    if (!ui) return 0.0;
    double elapsed = ui->run_elapsed_sec;
    if (ui->run_clock_active) {
        double delta = run_clock_sec() - ui->run_last_sec;
        if (delta > 0.0) elapsed += delta;
    }
    return elapsed;
}

static void format_run_elapsed(char *dst, size_t dstlen, double sec) {
    if (sec < 0.0) sec = 0.0;
    unsigned long long minutes = (unsigned long long)(sec / 60.0);
    unsigned long long hours = minutes / 60ull;
    minutes %= 60ull;
    snprintf(dst, dstlen, "%02lluh:%02llum", hours, minutes);
}

typedef struct {
    bool enabled;
    bool raw_mode;
    bool thread_started;
    volatile sig_atomic_t running;
    pthread_t thread;
    pthread_mutex_t mu;
    struct termios orig_termios;
    int move_delta;
    bool enter_pressed;
    bool pause_pressed;
    bool quit_pressed;
} eval_input;

static eval_input global_input = {
    .mu = PTHREAD_MUTEX_INITIALIZER,
};

static void buf_append(byte_buf *b, const char *p, size_t n) {
    if (n == 0) return;
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 1024;
        while (cap < b->len + n + 1) cap *= 2;
        char *v = realloc(b->v, cap);
        if (!v) {
            fprintf(stderr, "ds4-eval: out of memory\n");
            exit(1);
        }
        b->v = v;
        b->cap = cap;
    }
    memcpy(b->v + b->len, p, n);
    b->len += n;
    b->v[b->len] = '\0';
}

static void buf_appendf(byte_buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list copy;
    va_copy(copy, ap);
    int n = vsnprintf(NULL, 0, fmt, copy);
    va_end(copy);
    if (n < 0) {
        va_end(ap);
        return;
    }
    char *tmp = malloc((size_t)n + 1);
    if (!tmp) {
        fprintf(stderr, "ds4-eval: out of memory\n");
        exit(1);
    }
    vsnprintf(tmp, (size_t)n + 1, fmt, ap);
    va_end(ap);
    buf_append(b, tmp, (size_t)n);
    free(tmp);
}

static int eval_case_nchoices(const eval_case *tc) {
    int n = 0;
    while (n < EVAL_MAX_CHOICES && tc->choice[n]) n++;
    return n;
}

static bool eval_case_is_multiple_choice(const eval_case *tc) {
    return eval_case_nchoices(tc) > 0;
}

static bool eval_case_is_compsec(const eval_case *tc) {
    return tc->source && !strcmp(tc->source, "COMPSEC");
}

static void style_append(style_buf *b, unsigned char style, size_t n) {
    if (n == 0) return;
    if (b->len + n > b->cap) {
        size_t cap = b->cap ? b->cap : 1024;
        while (cap < b->len + n) cap *= 2;
        unsigned char *v = realloc(b->v, cap);
        if (!v) {
            fprintf(stderr, "ds4-eval: out of memory\n");
            exit(1);
        }
        b->v = v;
        b->cap = cap;
    }
    memset(b->v + b->len, style, n);
    b->len += n;
}

static void buf_free(byte_buf *b) {
    free(b->v);
    memset(b, 0, sizeof(*b));
}

static void style_free(style_buf *b) {
    free(b->v);
    memset(b, 0, sizeof(*b));
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static double run_clock_sec(void) {
    struct timespec ts;
#ifdef CLOCK_UPTIME_RAW
    /* On Darwin this clock excludes system sleep, which is exactly what the TUI
     * elapsed counter wants: benchmark runtime, not lid-closed wall time. */
    clock_gettime(CLOCK_UPTIME_RAW, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static int parse_int_arg(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-eval: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static uint64_t parse_u64_arg(const char *s, const char *opt) {
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v == 0) {
        fprintf(stderr, "ds4-eval: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (uint64_t)v;
}

static float parse_float_arg(const char *s, const char *opt, float min, float max) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v) || v < min || v > max) {
        fprintf(stderr, "ds4-eval: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-eval: %s requires an argument\n", opt);
        exit(2);
    }
    return argv[++*i];
}

static ds4_backend parse_backend(const char *s, const char *opt) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4-eval: invalid value for %s: %s\n", opt, s);
    fprintf(stderr, "ds4-eval: valid backends are: metal, cuda, cpu\n");
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

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: ds4-eval [options]\n"
        "\n"
        "Runs a small built-in GPQA Diamond/audited SuperGPQA/AIME2025/COMPSEC integration test.\n"
        "The TTY UI keeps the question list on the left and streams sampled\n"
        "tokens live on the right; thinking text is dim grey until </think>.\n"
        "In the TTY UI, Up/Down selects a question, Enter runs it next,\n"
        "p pauses or resumes evaluation, and q exits with a report.\n"
        "\n"
        "Model and backend:\n"
        "  -m, --model FILE       GGUF model path. Default: ds4flash.gguf\n"
        "  --mtp FILE             Optional MTP support GGUF.\n"
        "  --metal | --cuda | --cpu | --backend NAME\n"
        "  -t, --threads N        CPU helper threads.\n"
        "  --quality              Prefer exact kernels where applicable.\n"
        "  --warm-weights         Touch mapped tensor pages before evaluation.\n"
        "\n"
        "Evaluation:\n"
        "  -n, --tokens N         Max generated tokens per question. Default: 16000\n"
        "  --questions N          Run only the first N embedded questions.\n"
        "  --temp F               Sampling temperature. Default: 0\n"
        "  --top-p F              Nucleus sampling probability. Default: 1\n"
        "  --min-p F              Keep tokens scoring at least F times the top token. Default: 0.05\n"
        "  --seed N               Sampling seed. Default: time-based\n"
        "  --trace FILE           Write questions, outputs, and grading decisions.\n"
        "  --think                Enable thinking mode. Default\n"
        "  --nothink              Disable thinking mode.\n"
        "  --soft-limit-reply-budget N\n"
        "                         Inside the last N tokens, close thinking if\n"
        "                         </think> is already among the top close ranks.\n"
        "                         Default: 1024\n"
        "  --hard-limit-reply-budget N\n"
        "                         Force </think> with N tokens left for the answer.\n"
        "                         Default: 512\n"
        "  --soft-limit-think-close-rank N\n"
        "                         Soft-close when </think> is in the top N tokens.\n"
        "                         Default: 3\n"
        "  --pause-ms N           Pause after each result in the TTY UI. Default: 350\n"
        "  --plain                Disable split-screen ANSI UI.\n"
        "  -h, --help             Show this help.\n");
}

static eval_config parse_options(int argc, char **argv) {
    eval_config c = {
        .model_path = "ds4flash.gguf",
        .backend = default_backend(),
        .max_tokens = 16000,
        .top_p = DS4_DEFAULT_TOP_P,
        .min_p = DS4_DEFAULT_MIN_P,
        .pause_ms = 350,
        .soft_limit_reply_budget = 1024,
        .hard_limit_reply_budget = 512,
        .soft_limit_think_close_rank = 3,
        .think_mode = DS4_THINK_HIGH,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp")) {
            c.mtp_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) {
            c.max_tokens = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--questions")) {
            c.question_limit = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--temp")) {
            c.temperature = parse_float_arg(need_arg(&i, argc, argv, arg), arg, 0.0f, 100.0f);
        } else if (!strcmp(arg, "--top-p")) {
            c.top_p = parse_float_arg(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--min-p")) {
            c.min_p = parse_float_arg(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--seed")) {
            c.seed = parse_u64_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--trace")) {
            c.trace_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--soft-limit-reply-budget")) {
            c.soft_limit_reply_budget = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--hard-limit-reply-budget")) {
            c.hard_limit_reply_budget = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--soft-limit-think-close-rank")) {
            c.soft_limit_think_close_rank = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--pause-ms")) {
            c.pause_ms = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.threads = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--backend")) {
            c.backend = parse_backend(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--metal")) {
            c.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--cuda")) {
            c.backend = DS4_BACKEND_CUDA;
        } else if (!strcmp(arg, "--cpu")) {
            c.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "--quality")) {
            c.quality = true;
        } else if (!strcmp(arg, "--warm-weights")) {
            c.warm_weights = true;
        } else if (!strcmp(arg, "--think")) {
            c.think_mode = DS4_THINK_HIGH;
        } else if (!strcmp(arg, "--nothink")) {
            c.think_mode = DS4_THINK_NONE;
        } else if (!strcmp(arg, "--plain")) {
            c.plain = true;
        } else if (!strcmp(arg, "--cpu-moe")) {
            c.cpu_moe = true;
        } else if (!strcmp(arg, "--n-cpu-moe")) {
            const char *s = need_arg(&i, argc, argv, arg);
            c.n_cpu_moe_layers = parse_int_arg(s, arg);
        } else if (!strcmp(arg, "--prefill-metal-phases")) {
            const char *s = need_arg(&i, argc, argv, arg);
            if (!strcmp(s, "auto")) {
                c.prefill_metal_phases = -1;
            } else {
                c.prefill_metal_phases = parse_int_arg(s, arg);
            }
        } else {
            fprintf(stderr, "ds4-eval: unknown option: %s\n", arg);
            usage(stderr);
            exit(2);
        }
    }
    if (c.max_tokens > EVAL_MAX_CONTEXT) {
        fprintf(stderr,
                "ds4-eval: --tokens (%d) exceeds the %d token context cap\n",
                c.max_tokens, EVAL_MAX_CONTEXT);
        exit(2);
    }
    if (c.hard_limit_reply_budget >= c.max_tokens) {
        fprintf(stderr,
                "ds4-eval: --hard-limit-reply-budget (%d) must be smaller than --tokens (%d)\n",
                c.hard_limit_reply_budget, c.max_tokens);
        exit(2);
    }
    if (c.soft_limit_reply_budget < c.hard_limit_reply_budget) {
        fprintf(stderr,
                "ds4-eval: --soft-limit-reply-budget (%d) must be >= --hard-limit-reply-budget (%d)\n",
                c.soft_limit_reply_budget, c.hard_limit_reply_budget);
        exit(2);
    }
    return c;
}

static int terminal_size(int *cols, int *rows) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) != 0 || ws.ws_col == 0 || ws.ws_row == 0) {
        *cols = 80;
        *rows = 24;
        return -1;
    }
    *cols = ws.ws_col;
    *rows = ws.ws_row;
    return 0;
}

static void term_move(int row, int col) {
    printf("\x1b[%d;%dH", row, col);
}

static void term_clear_to_eol(void) {
    fputs("\x1b[K", stdout);
}

static int tui_right_text_w(const eval_ui *ui) {
    /* Avoid writing the physical last column.  Many terminals defer wrapping
     * until the next byte after the last column, which would spill right-pane
     * status text into the next row's left pane. */
    return ui->right_w > 1 ? ui->right_w - 1 : ui->right_w;
}

static void tui_clear_left_line(eval_ui *ui, int row) {
    term_move(row, 1);
    for (int i = 0; i < ui->left_w; i++) fputc(' ', stdout);
    term_move(row, ui->left_w + 1);
    fputs(ANSI_DIM "|" ANSI_RESET, stdout);
    term_move(row, 1);
}

static void term_set_color_for_status(eval_status st) {
    switch (st) {
    case EVAL_PENDING:  fputs(ANSI_DIM, stdout); break;
    case EVAL_PREFILL:  fputs(ANSI_CYAN, stdout); break;
    case EVAL_THINKING: fputs(ANSI_YELLOW, stdout); break;
    case EVAL_SKIPPED:  fputs(ANSI_DIM, stdout); break;
    case EVAL_STOPPED:  fputs(ANSI_YELLOW ANSI_BOLD, stdout); break;
    case EVAL_PASSED:   fputs(ANSI_GREEN ANSI_BOLD, stdout); break;
    case EVAL_FAILED:   fputs(ANSI_RED ANSI_BOLD, stdout); break;
    }
}

static const char *status_name(eval_status st) {
    switch (st) {
    case EVAL_PENDING: return "PEND";
    case EVAL_PREFILL: return "FILL";
    case EVAL_THINKING: return "RUN";
    case EVAL_SKIPPED: return "SKIP";
    case EVAL_STOPPED: return "STOP";
    case EVAL_PASSED: return "PASS";
    case EVAL_FAILED: return "FAIL";
    }
    return "?";
}

static bool eval_status_running(eval_status st) {
    return st == EVAL_PREFILL || st == EVAL_THINKING;
}

static void print_trimmed(const char *s, int width) {
    if (width <= 0) return;
    int len = (int)strlen(s);
    if (len <= width) {
        fputs(s, stdout);
        return;
    }
    if (width <= 3) {
        for (int i = 0; i < width; i++) fputc('.', stdout);
        return;
    }
    fwrite(s, 1, (size_t)width - 3, stdout);
    fputs("...", stdout);
}

static void tui_draw_question_preview(eval_ui *ui, const eval_case *tc) {
    const char *q = tc->question;
    size_t pos = 0;
    const int width = tui_right_text_w(ui);

    for (int row = 0; row < 3; row++) {
        term_move(2 + row, ui->right_x);
        term_clear_to_eol();
        fputs(ANSI_BLUE, stdout);

        int printed = 0;
        bool last_space = true;
        while (q[pos] && printed < width) {
            unsigned char c = (unsigned char)q[pos++];
            if (isspace(c)) {
                if (last_space) continue;
                c = ' ';
                last_space = true;
            } else {
                last_space = false;
            }
            fputc(c, stdout);
            printed++;
        }

        if (row == 2) {
            while (q[pos] && isspace((unsigned char)q[pos])) pos++;
            if (q[pos] && width >= 3) {
                int ell_col = ui->right_x + (printed >= 3 ? printed - 3 : 0);
                term_move(2 + row, ell_col);
                fputs("...", stdout);
            }
        }
        fputs(ANSI_RESET, stdout);
    }

    term_move(5, ui->right_x);
    term_clear_to_eol();
}

static void input_queue_key(int key) {
    pthread_mutex_lock(&global_input.mu);
    if (key == 'A') {
        global_input.move_delta--;
    } else if (key == 'B') {
        global_input.move_delta++;
    } else if (key == '\r' || key == '\n') {
        global_input.enter_pressed = true;
    } else if (key == 'p' || key == 'P') {
        global_input.pause_pressed = true;
    } else if (key == 'q' || key == 'Q') {
        global_input.quit_pressed = true;
    }
    pthread_mutex_unlock(&global_input.mu);
}

static void *input_thread_main(void *arg) {
    (void)arg;
    int esc_state = 0;
    char buf[64];

    while (global_input.running) {
        ssize_t n = read(STDIN_FILENO, buf, sizeof(buf));
        if (n <= 0) {
            usleep(5000);
            continue;
        }
        for (ssize_t i = 0; i < n; i++) {
            unsigned char c = (unsigned char)buf[i];
            if (esc_state == 0) {
                if (c == 27) esc_state = 1;
                else if (c == 'p' || c == 'P' || c == 'q' || c == 'Q') input_queue_key(c);
                else if (c == '\r' || c == '\n') input_queue_key(c);
            } else if (esc_state == 1) {
                esc_state = (c == '[' || c == 'O') ? 2 : 0;
            } else {
                if (c == 'A' || c == 'B') input_queue_key(c);
                esc_state = 0;
            }
        }
    }
    return NULL;
}

static void tui_start_input(void) {
    if (!isatty(STDIN_FILENO)) return;
    if (tcgetattr(STDIN_FILENO, &global_input.orig_termios) != 0) return;

    /* The input thread is intentionally boring: it never writes to the terminal,
     * it only queues navigation/control state.  Rendering remains owned by the
     * main thread and follows the same full-frame redraw path as the
     * noninteractive UI. Keep ISIG set so Ctrl-C still restores the alternate
     * screen. */
    struct termios raw = global_input.orig_termios;
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 1;
    if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) return;

    pthread_mutex_lock(&global_input.mu);
    global_input.move_delta = 0;
    global_input.enter_pressed = false;
    global_input.pause_pressed = false;
    global_input.quit_pressed = false;
    pthread_mutex_unlock(&global_input.mu);

    global_input.raw_mode = true;
    global_input.enabled = true;
    global_input.running = 1;
    if (pthread_create(&global_input.thread, NULL, input_thread_main, NULL) == 0) {
        global_input.thread_started = true;
    } else {
        global_input.running = 0;
        global_input.enabled = false;
        tcsetattr(STDIN_FILENO, TCSANOW, &global_input.orig_termios);
        global_input.raw_mode = false;
    }
}

static void tui_stop_input(void) {
    if (global_input.thread_started) {
        global_input.running = 0;
        pthread_join(global_input.thread, NULL);
        global_input.thread_started = false;
    }
    if (global_input.raw_mode) {
        tcsetattr(STDIN_FILENO, TCSANOW, &global_input.orig_termios);
        global_input.raw_mode = false;
    }
    global_input.enabled = false;
}

static void tui_consume_input(eval_ui *ui) {
    if (!ui->enabled || !global_input.enabled) return;

    pthread_mutex_lock(&global_input.mu);
    int move = global_input.move_delta;
    bool enter = global_input.enter_pressed;
    bool pause = global_input.pause_pressed;
    bool quit = global_input.quit_pressed;
    global_input.move_delta = 0;
    global_input.enter_pressed = false;
    global_input.pause_pressed = false;
    global_input.quit_pressed = false;
    pthread_mutex_unlock(&global_input.mu);

    if (quit) {
        ui->quit_requested = true;
        ui->paused = false;
        return;
    }

    if (pause) ui->paused = !ui->paused;

    if (move) {
        if (!ui->selection_active) {
            ui->selected_case = ui->active_case;
            ui->selection_active = true;
        }
        int selected = ui->selected_case + move;
        if (selected < 0) selected = 0;
        if (selected >= ui->ncases) selected = ui->ncases - 1;
        ui->selected_case = selected;
    }

    if (enter) {
        if (!ui->selection_active) {
            ui->selected_case = ui->active_case;
            ui->selection_active = true;
        }
        if (ui->selected_case != ui->active_case ||
            !eval_status_running(ui->status[ui->active_case]))
        {
            ui->requested_case = ui->selected_case;
        }
    }
}

static void tui_restore(void) {
    eval_ui *ui = global_ui;
    tui_stop_input();
    if (!ui || !ui->active) return;
    fputs(ANSI_RESET "\x1b[?25h\x1b[?1049l", stdout);
    fflush(stdout);
    ui->active = false;
}

static void tui_signal_restore(int sig) {
    /* Signal handlers cannot safely run the full tui_restore() path: that path
     * joins the input thread and uses stdio.  For Ctrl-C / termination, do the
     * minimal terminal repair directly, then re-raise the signal with its
     * default action so the process status remains correct. */
    eval_ui *ui = global_ui;
    global_input.running = 0;
    if (global_input.raw_mode) {
        tcsetattr(STDIN_FILENO, TCSANOW, &global_input.orig_termios);
        global_input.raw_mode = false;
    }
    if (ui && ui->active) {
        const char restore[] = ANSI_RESET "\x1b[?25h\x1b[?1049l";
        (void)write(STDOUT_FILENO, restore, sizeof(restore) - 1);
        ui->active = false;
    }
    signal(sig, SIG_DFL);
    raise(sig);
}

static void tui_draw_title(eval_ui *ui) {
    term_move(1, 1);
    tui_clear_left_line(ui, 1);
    char elapsed[32];
    format_run_elapsed(elapsed, sizeof(elapsed), tui_run_clock_visible_sec(ui));
    fputs("ds4-eval (" ANSI_BOLD "p" ANSI_RESET ")ause (" ANSI_BOLD "q" ANSI_RESET ")uit", stdout);
    printf(" %s", elapsed);
    if (ui->paused) {
        fputs(" " ANSI_RED ANSI_BOLD "PAUSED" ANSI_RESET, stdout);
    }
}

static void tui_draw_frame(eval_ui *ui) {
    fputs("\x1b[2J", stdout);
    tui_draw_title(ui);
    term_move(1, ui->right_x);
    fputs(ANSI_BOLD "live sampled tokens" ANSI_RESET, stdout);

    for (int row = 1; row <= ui->rows; row++) {
        term_move(row, ui->left_w + 1);
        fputs(ANSI_DIM "|" ANSI_RESET, stdout);
    }
}

static void tui_draw_left(eval_ui *ui) {
    int passed = 0;
    int failed = 0;
    for (int i = 0; i < ui->ncases; i++) {
        if (ui->status[i] == EVAL_PASSED) passed++;
        else if (ui->status[i] == EVAL_FAILED) failed++;
    }

    tui_draw_title(ui);

    term_move(2, 1);
    tui_clear_left_line(ui, 2);
    printf("score %s%d%s/%d  failed %s%d%s",
           ANSI_GREEN, passed, ANSI_RESET,
           ui->ncases,
           failed ? ANSI_RED : ANSI_DIM, failed, ANSI_RESET);

    const int first_row = 4;
    const int visible_rows = ui->rows >= first_row ? ui->rows - first_row + 1 : 0;
    const int shown = ui->ncases < visible_rows ? ui->ncases : visible_rows;
    int first = 0;
    if (shown > 0 && ui->ncases > shown) {
        /* The list follows the selection cursor, not the running test.  This
         * makes arrow navigation visible immediately.  When normal execution
         * advances, main() moves the selection to the new active case so the
         * viewport naturally follows the work again. */
        int anchor = ui->selected_case;
        if (anchor < 0) anchor = ui->active_case;
        if (anchor >= ui->ncases) anchor = ui->ncases - 1;
        first = anchor - shown / 2;
        if (first < 0) first = 0;
        if (first > ui->ncases - shown) first = ui->ncases - shown;
    }

    for (int row = 0; row < visible_rows; row++) {
        const int screen_row = first_row + row;
        const int i = first + row;
        term_move(screen_row, 1);
        tui_clear_left_line(ui, screen_row);
        if (i >= ui->ncases) continue;

        if (i == ui->active_case) fputs(ANSI_BOLD, stdout);
        term_set_color_for_status(ui->status[i]);
        printf("%c%2d ", ui->selection_active && i == ui->selected_case ? '>' : ' ', i + 1);
        printf("%-4s", status_name(ui->status[i]));
        fputs(ANSI_RESET, stdout);

        const int answer_w = 18;
        const int answer_col = ui->left_w - answer_w + 1;
        int title_w = answer_col - 11;
        if (title_w > 0) {
            fputc(' ', stdout);
            char title[512];
            snprintf(title, sizeof(title), "%s: %s",
                     ui->cases[i].source, ui->cases[i].title);
            print_trimmed(title, title_w);
        }
        if (ui->status[i] == EVAL_FAILED || ui->status[i] == EVAL_PASSED) {
            char answers[64];
            snprintf(answers, sizeof(answers), "%s/%s",
                     ui->guess[i][0] ? ui->guess[i] : "?",
                     ui->cases[i].answer);
            term_move(screen_row, answer_col);
            print_trimmed(answers, answer_w);
        }
        fputs(ANSI_RESET, stdout);
    }
}

static void tui_reset_stream(eval_ui *ui, const eval_case *tc, bool in_think) {
    ui->stream.len = 0;
    if (ui->stream.v) ui->stream.v[0] = '\0';
    ui->styles.len = 0;
    ui->in_think = in_think;
    ui->pending_tag_len = 0;
    ui->generated = 0;
    ui->prefill_current = 0;
    ui->prefill_total = 0;
    ui->phase_start_sec = now_sec();
    ui->speed_tps = 0.0;

    for (int row = 2; row <= ui->rows; row++) {
        term_move(row, ui->right_x);
        term_clear_to_eol();
    }
    tui_draw_question_preview(ui, tc);
}

static bool bytes_has_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n >= plen && memcmp(p, prefix, plen) == 0;
}

static bool bytes_is_partial_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n < plen && memcmp(prefix, p, n) == 0;
}

static void stream_append_visible(eval_ui *ui, const char *p, size_t n) {
    buf_append(&ui->stream, p, n);
    style_append(&ui->styles, ui->in_think ? 1 : 0, n);
}

static void stream_append_token_text(eval_ui *ui, const char *text, size_t len, bool finish) {
    const char *open = "<think>";
    const char *close = "</think>";
    size_t total = ui->pending_tag_len + len;
    char *tmp = malloc(total ? total : 1);
    if (!tmp) {
        fprintf(stderr, "ds4-eval: out of memory\n");
        exit(1);
    }
    if (ui->pending_tag_len) memcpy(tmp, ui->pending_tag, ui->pending_tag_len);
    if (len) memcpy(tmp + ui->pending_tag_len, text, len);
    ui->pending_tag_len = 0;

    size_t i = 0;
    while (i < total) {
        const char *cur = tmp + i;
        size_t rem = total - i;
        if (bytes_has_prefix(cur, rem, open)) {
            ui->in_think = true;
            i += strlen(open);
            continue;
        }
        if (bytes_has_prefix(cur, rem, close)) {
            ui->in_think = false;
            stream_append_visible(ui, "\n", 1);
            i += strlen(close);
            continue;
        }
        if (!finish && cur[0] == '<' &&
            (bytes_is_partial_prefix(cur, rem, open) ||
             bytes_is_partial_prefix(cur, rem, close)))
        {
            if (rem < sizeof(ui->pending_tag)) {
                memcpy(ui->pending_tag, cur, rem);
                ui->pending_tag_len = rem;
            }
            break;
        }
        stream_append_visible(ui, cur, 1);
        i++;
    }
    free(tmp);
}

typedef struct {
    size_t start;
    size_t end;
} line_span;

static void tui_draw_stream(eval_ui *ui) {
    if (!ui->enabled) return;
    const int width = tui_right_text_w(ui);

    line_span *lines = NULL;
    int line_len = 0;
    int line_cap = 0;
    size_t line_start = 0;
    int col = 0;

    for (size_t i = 0; i < ui->stream.len; i++) {
        char c = ui->stream.v[i];
        bool end_line = false;
        size_t end = i;
        if (c == '\n') {
            end_line = true;
            end = i;
        } else {
            col++;
            if (col >= width) {
                end_line = true;
                end = i + 1;
            }
        }
        if (end_line) {
            if (line_len == line_cap) {
                line_cap = line_cap ? line_cap * 2 : 64;
                line_span *v = realloc(lines, (size_t)line_cap * sizeof(*lines));
                if (!v) {
                    fprintf(stderr, "ds4-eval: out of memory\n");
                    exit(1);
                }
                lines = v;
            }
            lines[line_len++] = (line_span){line_start, end};
            line_start = i + 1;
            col = 0;
        }
    }
    if (line_start <= ui->stream.len) {
        if (line_len == line_cap) {
            line_cap = line_cap ? line_cap * 2 : 64;
            line_span *v = realloc(lines, (size_t)line_cap * sizeof(*lines));
            if (!v) {
                fprintf(stderr, "ds4-eval: out of memory\n");
                exit(1);
            }
            lines = v;
        }
        lines[line_len++] = (line_span){line_start, ui->stream.len};
    }

    int start = line_len > ui->body_h ? line_len - ui->body_h : 0;
    for (int row = 0; row < ui->body_h; row++) {
        term_move(ui->body_y + row, ui->right_x);
        term_clear_to_eol();
        int li = start + row;
        if (li >= line_len) continue;
        unsigned char cur_style = 255;
        int printed = 0;
        for (size_t i = lines[li].start; i < lines[li].end && printed < width; i++) {
            unsigned char st = ui->styles.v[i];
            if (st != cur_style) {
                fputs(st ? ANSI_DIM : ANSI_RESET, stdout);
                cur_style = st;
            }
            char c = ui->stream.v[i];
            if (c == '\r' || c == '\n') continue;
            if (c == '\t') c = ' ';
            fputc((unsigned char)c, stdout);
            printed++;
        }
        fputs(ANSI_RESET, stdout);
    }
    free(lines);
}

static void format_short_count(char *dst, size_t dstlen, int n) {
    if (n >= 1000000) {
        snprintf(dst, dstlen, "%dm", (n + 500000) / 1000000);
    } else if (n >= 10000) {
        snprintf(dst, dstlen, "%dk", (n + 500) / 1000);
    } else {
        snprintf(dst, dstlen, "%d", n);
    }
}

static const char *short_phase_name(const char *phase) {
    if (!strcmp(phase, "prefill")) return "FILL";
    if (!strcmp(phase, "thinking")) return "THINK";
    if (!strcmp(phase, "answer")) return "ANS";
    if (!strcmp(phase, "passed")) return "PASS";
    if (!strcmp(phase, "failed")) return "FAIL";
    return "RUN";
}

static void tui_draw_right_status(eval_ui *ui, const char *phase) {
    term_move(1, ui->right_x);
    term_clear_to_eol();
    const eval_case *tc = &ui->cases[ui->active_case];
    char id[13];
    char gen[16], max[16], cur[16], total[16];
    char line[512];
    snprintf(id, sizeof(id), "%.12s", tc->id);
    if (ui->prefill_total > 0 && ui->status[ui->active_case] == EVAL_PREFILL) {
        double pct = 100.0 * (double)ui->prefill_current / (double)ui->prefill_total;
        format_short_count(cur, sizeof(cur), ui->prefill_current);
        format_short_count(total, sizeof(total), ui->prefill_total);
        snprintf(line, sizeof(line),
                 "Speed %.2f t/s  %s %s/%s %.0f%%  %s/%s",
                 ui->speed_tps, short_phase_name(phase), cur, total, pct,
                 tc->source, id);
    } else {
        int phase_max = !strcmp(phase, "thinking") ? ui->think_max_tokens : ui->max_tokens;
        int phase_gen = ui->generated;
        if (phase_max < 0) phase_max = 0;
        if (!strcmp(phase, "thinking") && phase_gen > phase_max) phase_gen = phase_max;
        format_short_count(gen, sizeof(gen), phase_gen);
        format_short_count(max, sizeof(max), phase_max);
        snprintf(line, sizeof(line),
                 "Speed %.2f t/s  %s %s/%s  %s/%s",
                 ui->speed_tps, short_phase_name(phase), gen, max,
                 tc->source, id);
    }
    print_trimmed(line, tui_right_text_w(ui));
}

static void tui_refresh(eval_ui *ui, const char *phase) {
    if (!ui->enabled) return;
    tui_draw_left(ui);
    tui_draw_right_status(ui, phase);
    tui_draw_stream(ui);
    fflush(stdout);
}

static void tui_start(eval_ui *ui, const eval_case *cases, int ncases, int max_tokens,
                      bool enabled) {
    memset(ui, 0, sizeof(*ui));
    ui->enabled = enabled;
    ui->cases = cases;
    ui->ncases = ncases;
    ui->max_tokens = max_tokens;
    ui->selected_case = 0;
    ui->requested_case = -1;
    ui->quit_requested = false;
    ui->status = calloc((size_t)ncases, sizeof(*ui->status));
    ui->guess = calloc((size_t)ncases, sizeof(*ui->guess));
    ui->prompt_tokens = calloc((size_t)ncases, sizeof(*ui->prompt_tokens));
    ui->generated_tokens = calloc((size_t)ncases, sizeof(*ui->generated_tokens));
    if (!ui->status || !ui->guess || !ui->prompt_tokens || !ui->generated_tokens) {
        fprintf(stderr, "ds4-eval: out of memory\n");
        exit(1);
    }
    if (!enabled) return;

    terminal_size(&ui->cols, &ui->rows);
    if (ui->cols < 90 || ui->rows < 18) {
        ui->enabled = false;
        return;
    }
    ui->left_w = ui->cols / 2;
    if (ui->left_w < 42) ui->left_w = 42;
    if (ui->left_w > 72) ui->left_w = 72;
    ui->right_x = ui->left_w + 3;
    ui->right_w = ui->cols - ui->right_x + 1;
    ui->body_y = 6;
    ui->body_h = ui->rows - ui->body_y + 1;

    global_ui = ui;
    atexit(tui_restore);
    signal(SIGINT, tui_signal_restore);
    signal(SIGTERM, tui_signal_restore);
#ifdef SIGHUP
    signal(SIGHUP, tui_signal_restore);
#endif
#ifdef SIGQUIT
    signal(SIGQUIT, tui_signal_restore);
#endif

    fputs("\x1b[?1049h\x1b[?25l", stdout);
    ui->active = true;
    tui_start_input();
    tui_draw_frame(ui);
    tui_refresh(ui, "idle");
}

static void tui_free(eval_ui *ui) {
    if (ui->active) tui_restore();
    free(ui->status);
    free(ui->guess);
    free(ui->prompt_tokens);
    free(ui->generated_tokens);
    buf_free(&ui->stream);
    style_free(&ui->styles);
    memset(ui, 0, sizeof(*ui));
}

static void plain_set_thinking_color(bool use_color) {
    if (use_color) fputs(ANSI_DIM, stdout);
}

static void plain_reset_color(bool use_color) {
    if (use_color) fputs(ANSI_RESET, stdout);
}

static const char *eval_system_prompt(void) {
    return "You are solving a hard benchmark question. Reason carefully. "
           "The final answer must follow the requested format exactly.";
}

static char *build_question_prompt(const eval_case *tc) {
    byte_buf b = {0};
    int nchoices = eval_case_nchoices(tc);
    buf_appendf(&b, "%s\n", tc->question);
    if (nchoices > 0) {
        buf_append(&b, "\nChoices:\n", strlen("\nChoices:\n"));
        for (int i = 0; i < nchoices; i++) {
            buf_appendf(&b, "%c. %s\n", 'A' + i, tc->choice[i]);
        }
        buf_append(&b,
            "\nSolve the question. At the end, write exactly one final line in this "
            "format and do not write anything after it:\n"
            "Answer: <letter>",
            strlen("\nSolve the question. At the end, write exactly one final line in this "
                   "format and do not write anything after it:\n"
                   "Answer: <letter>"));
    } else if (eval_case_is_compsec(tc)) {
        buf_append(&b,
            "\nAt the end, write exactly one final line in this format and do not "
            "write anything after it:\n"
            "Answer: <line number or comma-separated line numbers>",
            strlen("\nAt the end, write exactly one final line in this format and do not "
                   "write anything after it:\n"
                   "Answer: <line number or comma-separated line numbers>"));
    } else {
        buf_append(&b,
            "\nSolve the problem. At the end, write exactly one final line in this "
            "format and do not write anything after it:\n"
            "Answer: <integer>",
            strlen("\nSolve the problem. At the end, write exactly one final line in this "
                   "format and do not write anything after it:\n"
                   "Answer: <integer>"));
    }
    if (b.v) return b.v;
    char *empty = malloc(1);
    if (empty) empty[0] = '\0';
    return empty;
}

static int eval_max_prompt_tokens(ds4_engine *engine,
                                  const eval_config *cfg,
                                  const eval_case *cases,
                                  int ncases,
                                  int ctx_for_think_mode,
                                  int *max_case_out)
{
    int max_prompt = 0;
    int max_case = -1;
    const ds4_think_mode think_mode =
        ds4_think_mode_for_context(cfg->think_mode, ctx_for_think_mode);

    for (int i = 0; i < ncases; i++) {
        char *question = build_question_prompt(&cases[i]);
        if (!question) {
            fprintf(stderr, "ds4-eval: failed to allocate prompt\n");
            exit(1);
        }
        ds4_tokens prompt = {0};
        ds4_encode_chat_prompt(engine, eval_system_prompt(), question, think_mode, &prompt);
        if (prompt.len > max_prompt) {
            max_prompt = prompt.len;
            max_case = i;
        }
        ds4_tokens_free(&prompt);
        free(question);
    }
    if (max_case_out) *max_case_out = max_case;
    return max_prompt;
}

static int eval_auto_context_size(ds4_engine *engine,
                                  eval_config *cfg,
                                  const eval_case *cases,
                                  int ncases,
                                  int *max_prompt_out,
                                  int *max_case_out)
{
    int ctx = EVAL_MAX_CONTEXT;
    int max_prompt = 0;
    int max_case = -1;

    /* Think Max downgrades to normal thinking under its minimum context.  Size
     * the prompts iteratively so the prompt tokenizer sees the same effective
     * thinking mode that the actual run will use. */
    for (int iter = 0; iter < 3; iter++) {
        max_prompt = eval_max_prompt_tokens(engine, cfg, cases, ncases, ctx, &max_case);
        long long required = (long long)max_prompt + (long long)cfg->max_tokens;
        if (required > EVAL_MAX_CONTEXT) {
            fprintf(stderr,
                    "ds4-eval: largest prompt (%d tokens, case %d) + --tokens (%d) exceeds the %d token context cap\n",
                    max_prompt, max_case + 1, cfg->max_tokens, EVAL_MAX_CONTEXT);
            exit(2);
        }
        if ((int)required == ctx) break;
        ctx = (int)required;
    }

    if (max_prompt_out) *max_prompt_out = max_prompt;
    if (max_case_out) *max_case_out = max_case;
    return ctx;
}

static void trace_write_block(FILE *trace, const char *label, const char *text) {
    if (!trace) return;
    size_t len = text ? strlen(text) : 0;
    fprintf(trace, "%s_BEGIN bytes=%zu\n", label, len);
    if (len) {
        fwrite(text, 1, len, trace);
        if (text[len - 1] != '\n') fputc('\n', trace);
    }
    fprintf(trace, "%s_END\n", label);
}

static const char *think_close_kind_name(eval_think_close_kind kind) {
    switch (kind) {
    case EVAL_THINK_CLOSE_NATURAL: return "natural";
    case EVAL_THINK_CLOSE_SOFT: return "soft_forced";
    case EVAL_THINK_CLOSE_HARD: return "hard_forced";
    case EVAL_THINK_CLOSE_NONE:
    default: return "none";
    }
}

static int token_rank_in_top(ds4_session *session, int token, int max_rank) {
    if (token < 0 || max_rank <= 0) return 0;
    ds4_token_score *top = malloc((size_t)max_rank * sizeof(*top));
    if (!top) return 0;
    int n = ds4_session_top_logprobs(session, top, max_rank);
    int rank = 0;
    for (int i = 0; i < n; i++) {
        if (top[i].id == token) {
            rank = i + 1;
            break;
        }
    }
    free(top);
    return rank;
}

static void trace_write_header(FILE *trace, const eval_config *cfg, int ncases, int max_prompt_tokens) {
    if (!trace) return;
    fprintf(trace,
            "# ds4-eval trace\n"
            "started_unix: %lld\n"
            "model: %s\n"
            "backend: %s\n"
            "ctx: %d\n"
            "max_tokens: %d\n"
            "max_prompt_tokens: %d\n"
            "questions: %d\n"
            "temperature: %.6g\n"
            "top_p: %.6g\n"
            "min_p: %.6g\n"
            "seed: %llu\n"
            "think_mode_requested: %s\n"
            "soft_limit_reply_budget: %d\n"
            "hard_limit_reply_budget: %d\n"
            "soft_limit_think_close_rank: %d\n"
            "\n",
            (long long)time(NULL),
            cfg->model_path,
            ds4_backend_name(cfg->backend),
            cfg->ctx_size,
            cfg->max_tokens,
            max_prompt_tokens,
            ncases,
            cfg->temperature,
            cfg->top_p,
            cfg->min_p,
            (unsigned long long)cfg->seed,
            ds4_think_mode_name(cfg->think_mode),
            cfg->soft_limit_reply_budget,
            cfg->hard_limit_reply_budget,
            cfg->soft_limit_think_close_rank);
    fflush(trace);
}

static void trace_write_case(FILE *trace,
                             const eval_config *cfg,
                             const eval_case *tc,
                             int idx,
                             int ncases,
                             const char *status,
                             const char *error,
                             const char *system_prompt,
                             const char *question_prompt,
                             const char *model_output,
                             ds4_think_mode effective_think_mode,
                             int prompt_tokens,
                             int generated_tokens,
                             double elapsed_sec,
                             const char *picked,
                             const eval_think_close_info *think_close) {
    if (!trace) return;
    int nchoices = eval_case_nchoices(tc);
    fprintf(trace,
            "===== CASE %d/%d %s/%s =====\n"
            "timestamp_unix: %lld\n"
            "source: %s\n"
            "id: %s\n"
            "domain: %s\n"
            "title: %s\n"
            "status: %s\n"
            "picked: %s\n"
            "expected: %s\n"
            "prompt_tokens: %d\n"
            "generated_tokens: %d\n"
            "elapsed_sec: %.3f\n"
            "temperature: %.6g\n"
            "top_p: %.6g\n"
            "min_p: %.6g\n"
            "think_mode_effective: %s\n",
            idx + 1, ncases, tc->source, tc->id,
            (long long)time(NULL),
            tc->source,
            tc->id,
            tc->domain,
            tc->title,
            status,
            picked && *picked ? picked : "?",
            tc->answer,
            prompt_tokens,
            generated_tokens,
            elapsed_sec,
            cfg->temperature,
            cfg->top_p,
            cfg->min_p,
            ds4_think_mode_name(effective_think_mode));
    if (think_close && think_close->kind != EVAL_THINK_CLOSE_NONE) {
        fprintf(trace,
                "think_close: %s\n"
                "think_close_token_index: %d\n"
                "think_close_remaining_budget: %d\n"
                "think_close_rank: %d\n",
                think_close_kind_name(think_close->kind),
                think_close->token_index,
                think_close->remaining_budget,
                think_close->rank);
    } else {
        fprintf(trace, "think_close: none\n");
    }
    if (error && *error) fprintf(trace, "error: %s\n", error);

    if (nchoices > 0) {
        fprintf(trace, "choices:\n");
        for (int i = 0; i < nchoices; i++) {
            fprintf(trace, "  %c. %s\n", 'A' + i, tc->choice[i]);
        }
    } else {
        fprintf(trace, "answer_kind: exact_integer\n");
    }
    trace_write_block(trace, "SYSTEM_PROMPT", system_prompt);
    trace_write_block(trace, "QUESTION_PROMPT", question_prompt);
    trace_write_block(trace, "MODEL_OUTPUT", model_output);
    fputc('\n', trace);
    fflush(trace);
}

static char *strcasestr_local(const char *hay, const char *needle) {
    size_t nlen = strlen(needle);
    if (nlen == 0) return (char *)hay;
    for (; *hay; hay++) {
        if (tolower((unsigned char)*hay) == tolower((unsigned char)needle[0]) &&
            strncasecmp(hay, needle, nlen) == 0)
            return (char *)hay;
    }
    return NULL;
}

static bool is_letter_boundary(char before, char after) {
    return !isalpha((unsigned char)before) && !isalpha((unsigned char)after);
}

static char find_answer_letter(const char *generated, int nchoices) {
    if (nchoices <= 0) return '?';
    const char *visible = strstr(generated, "</think>");
    visible = visible ? visible + 8 : generated;
    char max_answer = (char)('A' + nchoices - 1);

    char *answer = strcasestr_local(visible, "answer");
    if (answer) {
        const char *end = answer + strlen(answer);
        if (strlen(answer) > 96) end = answer + 96;
        for (const char *p = answer; p < end && *p; p++) {
            char c = (char)toupper((unsigned char)*p);
            if (c >= 'A' && c <= max_answer) {
                char before = p == visible ? ' ' : p[-1];
                char after = p[1];
                if (is_letter_boundary(before, after)) return c;
            }
        }
    }

    for (const char *p = visible + strlen(visible); p > visible; p--) {
        char c = (char)toupper((unsigned char)p[-1]);
        if (c >= 'A' && c <= max_answer) {
            char before = p - 1 == visible ? ' ' : p[-2];
            char after = p[0];
            if (is_letter_boundary(before, after)) return c;
        }
    }
    return '?';
}

static void normalize_integer_answer(const char *p, size_t len,
                                     char *dst, size_t dstlen) {
    while (len > 1 && *p == '0') {
        p++;
        len--;
    }
    if (dstlen == 0) return;
    size_t n = len < dstlen - 1 ? len : dstlen - 1;
    memcpy(dst, p, n);
    dst[n] = '\0';
}

static bool scan_first_integer(const char *start, const char *end,
                               char *dst, size_t dstlen) {
    const char *p = start;
    while (p < end && *p) {
        if (isdigit((unsigned char)*p)) {
            const char *q = p + 1;
            while (q < end && isdigit((unsigned char)*q)) q++;
            normalize_integer_answer(p, (size_t)(q - p), dst, dstlen);
            return true;
        }
        p++;
    }
    return false;
}

static void find_integer_answer(const char *generated, char *dst, size_t dstlen) {
    if (dstlen == 0) return;
    snprintf(dst, dstlen, "?");
    const char *visible = strstr(generated, "</think>");
    visible = visible ? visible + 8 : generated;

    char *answer = strcasestr_local(visible, "answer");
    if (answer) {
        const char *end = answer + strlen(answer);
        if (strlen(answer) > 160) end = answer + 160;
        if (scan_first_integer(answer, end, dst, dstlen)) return;
    }

    const char *last_start = NULL;
    const char *last_end = NULL;
    for (const char *p = visible; *p; p++) {
        if (isdigit((unsigned char)*p)) {
            const char *q = p + 1;
            while (isdigit((unsigned char)*q)) q++;
            last_start = p;
            last_end = q;
            p = q - 1;
        }
    }
    if (last_start && last_end) {
        normalize_integer_answer(last_start, (size_t)(last_end - last_start),
                                 dst, dstlen);
    }
}

static void normalize_compsec_line_spec(const char *p, const char *end,
                                        char *dst, size_t dstlen) {
    if (dstlen == 0) return;
    size_t n = 0;

    /* The model is instructed to finish with:
     *
     *     Answer: <line number or comma-separated line numbers>
     *
     * We still accept harmless surface variants in that final line: "line 9",
     * "9 and 15", "9, 15", or "20-22".  The hidden key decides what set of
     * exact line numbers is acceptable; this routine only normalizes the model
     * text into the same compact line-spec syntax.
     */
    for (; p < end && *p; p++) {
        if (!isdigit((unsigned char)*p)) continue;

        if (n > 0 && n + 1 < dstlen) dst[n++] = ',';
        while (p < end && isdigit((unsigned char)*p)) {
            if (n + 1 < dstlen) dst[n++] = *p;
            p++;
        }
        while (p < end && isspace((unsigned char)*p)) p++;

        if (p < end && *p == '-') {
            if (n + 1 < dstlen) dst[n++] = '-';
            p++;
            while (p < end && isspace((unsigned char)*p)) p++;
            while (p < end && isdigit((unsigned char)*p)) {
                if (n + 1 < dstlen) dst[n++] = *p;
                p++;
            }
        }
        if (p >= end || !*p) break;
    }
    while (n > 0 && (dst[n - 1] == ',' || dst[n - 1] == '-')) n--;
    dst[n] = '\0';
    if (n == 0) snprintf(dst, dstlen, "?");
}

static void find_compsec_answer(const char *generated, char *dst, size_t dstlen) {
    if (dstlen == 0) return;
    snprintf(dst, dstlen, "?");
    const char *visible = strstr(generated, "</think>");
    visible = visible ? visible + 8 : generated;

    char *answer = strcasestr_local(visible, "answer");
    if (answer) {
        const char *end = answer + strlen(answer);
        if (strlen(answer) > 160) end = answer + 160;
        const char *newline = memchr(answer, '\n', (size_t)(end - answer));
        if (newline) end = newline;
        normalize_compsec_line_spec(answer, end, dst, dstlen);
        if (strcmp(dst, "?") != 0) return;
    }
    find_integer_answer(generated, dst, dstlen);
}

static bool parse_line_spec(const char *spec, bool *set, size_t setlen) {
    bool any = false;
    const char *p = spec;
    while (p && *p) {
        while (*p && !isdigit((unsigned char)*p)) p++;
        if (!*p) break;
        char *end = NULL;
        long a = strtol(p, &end, 10);
        long b = a;
        p = end;
        if (*p == '-') {
            p++;
            b = strtol(p, &end, 10);
            p = end;
        }
        if (a > b) {
            long tmp = a;
            a = b;
            b = tmp;
        }
        if (a < 0) a = 0;
        if (b >= (long)setlen) b = (long)setlen - 1;
        for (long i = a; i <= b; i++) {
            set[i] = true;
            any = true;
        }
    }
    return any;
}

static bool compsec_answer_matches(const char *expected_spec, const char *got_spec) {
    bool expected[256] = {0};
    bool got[256] = {0};
    if (!parse_line_spec(expected_spec, expected, sizeof(expected))) return false;
    if (!parse_line_spec(got_spec, got, sizeof(got))) return false;

    bool hit = false;
    for (size_t i = 0; i < sizeof(got); i++) {
        if (!got[i]) continue;
        if (!expected[i]) return false;
        hit = true;
    }
    /* The prompt asks the model for the single best line, or the smallest exact
     * set when the bug cannot be localized to one line.  The hidden expected
     * answer may be a small audited range when adjacent lines are equivalent
     * locations for the same bug.  Any model-supplied line must be inside that
     * accepted set, and at least one accepted line must be present. */
    return hit;
}

static void find_case_answer(const eval_case *tc, const char *generated,
                             char *dst, size_t dstlen) {
    if (dstlen == 0) return;
    if (eval_case_is_multiple_choice(tc)) {
        dst[0] = find_answer_letter(generated, eval_case_nchoices(tc));
        if (dstlen > 1) dst[1] = '\0';
    } else if (eval_case_is_compsec(tc)) {
        find_compsec_answer(generated, dst, dstlen);
    } else {
        find_integer_answer(generated, dst, dstlen);
    }
}

static bool answer_matches(const eval_case *tc, const char *got) {
    if (eval_case_is_multiple_choice(tc)) {
        return got && got[0] && tc->answer && got[0] == tc->answer[0];
    }
    if (eval_case_is_compsec(tc)) {
        return got && tc->answer && compsec_answer_matches(tc->answer, got);
    }
    char expected[EVAL_ANSWER_MAX];
    normalize_integer_answer(tc->answer, strlen(tc->answer), expected, sizeof(expected));
    return got && strcmp(got, expected) == 0;
}

static bool tui_has_switch_request(eval_ui *ui, int running_idx) {
    return ui->enabled &&
           ui->requested_case >= 0 &&
           ui->requested_case < ui->ncases &&
           ui->requested_case != running_idx;
}

static bool tui_has_quit_request(eval_ui *ui) {
    return ui->enabled && ui->quit_requested;
}

static void mark_case_pending(eval_ui *ui, int idx) {
    ui->status[idx] = EVAL_PENDING;
    ui->guess[idx][0] = '\0';
}

static double tui_wait_if_paused(eval_ui *ui, const char *phase) {
    if (!ui->enabled || !ui->paused) return 0.0;
    bool was_running = ui->run_clock_active;
    if (was_running) tui_run_clock_stop(ui);
    double start = now_sec();
    tui_refresh(ui, phase);
    while (ui->paused) {
        usleep(50000);
        tui_consume_input(ui);
        tui_refresh(ui, phase);
    }
    if (was_running) tui_run_clock_start(ui);
    tui_refresh(ui, phase);
    return now_sec() - start;
}

static void eval_prefill_progress(void *ud, const char *event, int current, int total) {
    eval_ui *ui = ud;
    if (!ui || !event || strcmp(event, "prefill_chunk")) return;
    tui_consume_input(ui);
    tui_run_clock_tick(ui);
    ui->prefill_current = current;
    ui->prefill_total = total;
    double elapsed = now_sec() - ui->phase_start_sec;
    ui->speed_tps = elapsed > 0.001 ? (double)current / elapsed : 0.0;
    tui_refresh(ui, "prefill");
    double paused_sec = tui_wait_if_paused(ui, "prefill");
    if (paused_sec > 0.0) ui->phase_start_sec += paused_sec;
}

static eval_run_result run_one_case(ds4_engine *engine, ds4_session *session,
                                    const eval_config *cfg, eval_ui *ui,
                                    FILE *trace, int idx, uint64_t *rng) {
    const eval_case *tc = &eval_cases[idx];
    const bool tty = ui->enabled;
    const bool use_plain_color = !tty && isatty(STDOUT_FILENO);
    const ds4_think_mode think_mode = ds4_think_mode_for_context(cfg->think_mode, cfg->ctx_size);
    const char *system = eval_system_prompt();

    char *question = build_question_prompt(tc);
    if (!question) {
        fprintf(stderr, "ds4-eval: failed to allocate prompt\n");
        return EVAL_RUN_ERROR;
    }

    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, system, question, think_mode, &prompt);
    ui->prompt_tokens[idx] = prompt.len;
    ui->generated_tokens[idx] = 0;

    if (prompt.len >= cfg->ctx_size) {
        ui->active_case = idx;
        if (!ui->selection_active) ui->selected_case = idx;
        ui->guess[idx][0] = '\0';
        ui->status[idx] = EVAL_SKIPPED;
        tui_refresh(ui, "idle");
        trace_write_case(trace, cfg, tc, idx, ui->ncases, "SKIPPED",
                         "prompt does not fit context", system, question, "",
                         think_mode, prompt.len, 0, 0.0, "?", NULL);
        if (!tty) {
            printf("\n[%d/%d] SKIPPED %s/%s prompt=%d ctx=%d\n",
                   idx + 1, ui->ncases, tc->source, tc->id, prompt.len, cfg->ctx_size);
        }
        free(question);
        ds4_tokens_free(&prompt);
        return EVAL_RUN_OK;
    }

    int generation_limit = cfg->max_tokens;
    int ctx_generation_limit = cfg->ctx_size - prompt.len;
    if (ctx_generation_limit < generation_limit) generation_limit = ctx_generation_limit;
    if (generation_limit < 1) {
        ui->active_case = idx;
        if (!ui->selection_active) ui->selected_case = idx;
        ui->guess[idx][0] = '\0';
        ui->status[idx] = EVAL_SKIPPED;
        tui_refresh(ui, "idle");
        trace_write_case(trace, cfg, tc, idx, ui->ncases, "SKIPPED",
                         "prompt leaves no generation room", system, question, "",
                         think_mode, prompt.len, 0, 0.0, "?", NULL);
        if (!tty) {
            printf("\n[%d/%d] SKIPPED %s/%s prompt=%d ctx=%d\n",
                   idx + 1, ui->ncases, tc->source, tc->id, prompt.len, cfg->ctx_size);
        }
        free(question);
        ds4_tokens_free(&prompt);
        return EVAL_RUN_OK;
    }

    ui->active_case = idx;
    if (!ui->selection_active) ui->selected_case = idx;
    ui->requested_case = -1;
    ui->guess[idx][0] = '\0';
    ui->status[idx] = EVAL_PREFILL;
    ui->max_tokens = generation_limit;
    ui->think_max_tokens = generation_limit - cfg->hard_limit_reply_budget;
    if (ui->think_max_tokens < 0) ui->think_max_tokens = 0;
    tui_run_clock_start(ui);
    if (tty) {
        tui_reset_stream(ui, tc, ds4_think_mode_enabled(think_mode));
        tui_refresh(ui, "prefill");
    } else {
        printf("\n%s[%d/%d] %s%s%s/%s%s%s %s%s\n",
               use_plain_color ? ANSI_BOLD : "",
               idx + 1, ui->ncases,
               use_plain_color ? ANSI_CYAN : "", tc->source,
               use_plain_color ? ANSI_RESET : "",
               use_plain_color ? ANSI_CYAN : "", tc->domain,
               use_plain_color ? ANSI_RESET : "", tc->title,
               use_plain_color ? ANSI_RESET : "");
        fflush(stdout);
    }

    char err[256];
    ds4_session_set_progress(session, eval_prefill_progress, ui);
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
        ds4_session_set_progress(session, NULL, NULL);
        tui_run_clock_stop(ui);
        fprintf(stderr, "ds4-eval: prefill failed for %s: %s\n", tc->id, err);
        trace_write_case(trace, cfg, tc, idx, ui->ncases, "ERROR", err,
                         system, question, "", think_mode, prompt.len, 0, 0.0, "?", NULL);
        free(question);
        ds4_tokens_free(&prompt);
        return EVAL_RUN_ERROR;
    }
    ds4_session_set_progress(session, NULL, NULL);
    int prompt_tokens = prompt.len;
    ui->prompt_tokens[idx] = prompt_tokens;
    ds4_tokens_free(&prompt);

    tui_consume_input(ui);
    tui_wait_if_paused(ui, "prefill");
    if (tui_has_quit_request(ui)) {
        ui->status[idx] = EVAL_STOPPED;
        ui->generated_tokens[idx] = 0;
        tui_run_clock_stop(ui);
        tui_refresh(ui, "idle");
        trace_write_case(trace, cfg, tc, idx, ui->ncases, "STOPPED", NULL,
                         system, question, "", think_mode, prompt_tokens, 0, 0.0, "?", NULL);
        free(question);
        return EVAL_RUN_QUIT;
    }
    if (tui_has_switch_request(ui, idx)) {
        mark_case_pending(ui, idx);
        tui_run_clock_stop(ui);
        tui_refresh(ui, "idle");
        trace_write_case(trace, cfg, tc, idx, ui->ncases, "SWITCHED", NULL,
                         system, question, "", think_mode, prompt_tokens, 0, 0.0, "?", NULL);
        free(question);
        return EVAL_RUN_SWITCH;
    }

    ui->status[idx] = EVAL_THINKING;
    ui->generated = 0;
    ui->phase_start_sec = now_sec();
    ui->speed_tps = 0.0;
    byte_buf raw = {0};
    bool plain_in_think = ds4_think_mode_enabled(think_mode);
    bool generation_in_think = ds4_think_mode_enabled(think_mode);
    eval_think_close_info think_close = {0};
    ds4_tokens think_close_tokens = {0};
    if (generation_in_think) ds4_tokenize_text(engine, "</think>", &think_close_tokens);
    if (!tty && plain_in_think) plain_set_thinking_color(use_plain_color);
    tui_refresh(ui, "thinking");

    const int eos = ds4_token_eos(engine);
    double t0 = ui->phase_start_sec;
    int forced_close_pos = -1;
    for (int i = 0; i < generation_limit; i++) {
        if (tty) {
            tui_consume_input(ui);
            if (tui_has_quit_request(ui)) {
                ui->status[idx] = EVAL_STOPPED;
                ui->generated_tokens[idx] = ui->generated;
                tui_run_clock_stop(ui);
                tui_refresh(ui, "idle");
                trace_write_case(trace, cfg, tc, idx, ui->ncases, "STOPPED", NULL,
                                 system, question, raw.v ? raw.v : "", think_mode,
                                 prompt_tokens, ui->generated, now_sec() - t0, "?",
                                 &think_close);
                free(question);
                ds4_tokens_free(&think_close_tokens);
                buf_free(&raw);
                return EVAL_RUN_QUIT;
            }
            if (tui_has_switch_request(ui, idx)) {
                mark_case_pending(ui, idx);
                ui->generated_tokens[idx] = ui->generated;
                tui_run_clock_stop(ui);
                tui_refresh(ui, "idle");
                trace_write_case(trace, cfg, tc, idx, ui->ncases, "SWITCHED", NULL,
                                 system, question, raw.v ? raw.v : "", think_mode,
                                 prompt_tokens, ui->generated, now_sec() - t0, "?",
                                 &think_close);
                free(question);
                ds4_tokens_free(&think_close_tokens);
                buf_free(&raw);
                return EVAL_RUN_SWITCH;
            }
            double paused_sec = tui_wait_if_paused(ui, ui->in_think ? "thinking" : "answer");
            if (paused_sec > 0.0) {
                ui->phase_start_sec += paused_sec;
                t0 += paused_sec;
            }
            if (tui_has_quit_request(ui)) {
                ui->status[idx] = EVAL_STOPPED;
                ui->generated_tokens[idx] = ui->generated;
                tui_run_clock_stop(ui);
                tui_refresh(ui, "idle");
                trace_write_case(trace, cfg, tc, idx, ui->ncases, "STOPPED", NULL,
                                 system, question, raw.v ? raw.v : "", think_mode,
                                 prompt_tokens, ui->generated, now_sec() - t0, "?",
                                 &think_close);
                free(question);
                ds4_tokens_free(&think_close_tokens);
                buf_free(&raw);
                return EVAL_RUN_QUIT;
            }
            if (tui_has_switch_request(ui, idx)) {
                mark_case_pending(ui, idx);
                ui->generated_tokens[idx] = ui->generated;
                tui_run_clock_stop(ui);
                tui_refresh(ui, "idle");
                trace_write_case(trace, cfg, tc, idx, ui->ncases, "SWITCHED", NULL,
                                 system, question, raw.v ? raw.v : "", think_mode,
                                 prompt_tokens, ui->generated, now_sec() - t0, "?",
                                 &think_close);
                free(question);
                ds4_tokens_free(&think_close_tokens);
                buf_free(&raw);
                return EVAL_RUN_SWITCH;
            }
        }

        int remaining_budget = generation_limit - ui->generated;
        int close_rank = 0;
        int token = -1;
        eval_think_close_kind close_kind = EVAL_THINK_CLOSE_NONE;

        /* Benchmarks usually cap generation length, but DeepSeek can spend the
         * entire budget in <think>.  This controller only acts while the model is
         * still in thinking mode.  The soft limit is conservative: it accepts the
         * model's own desire to end thinking when </think> is already near the
         * top of the distribution.  The hard limit is a uniform benchmark rule:
         * leave a fixed answer reserve instead of failing a case because all
         * tokens were spent before the visible answer could start. */
        if (generation_in_think && think_close_tokens.len > 0) {
            if (forced_close_pos >= 0) {
                token = think_close_tokens.v[forced_close_pos++];
                if (forced_close_pos >= think_close_tokens.len) forced_close_pos = -1;
            } else if (remaining_budget <= cfg->hard_limit_reply_budget) {
                close_kind = EVAL_THINK_CLOSE_HARD;
                token = think_close_tokens.v[0];
                forced_close_pos = think_close_tokens.len > 1 ? 1 : -1;
            } else if (remaining_budget <= cfg->soft_limit_reply_budget &&
                       think_close_tokens.len == 1) {
                close_rank = token_rank_in_top(session, think_close_tokens.v[0],
                                               cfg->soft_limit_think_close_rank);
                if (close_rank > 0) {
                    close_kind = EVAL_THINK_CLOSE_SOFT;
                    token = think_close_tokens.v[0];
                }
            }
        }
        if (token < 0)
            token = ds4_session_sample(session, cfg->temperature, 0,
                                       cfg->top_p, cfg->min_p, rng);
        if (token == eos) break;
        if (close_kind != EVAL_THINK_CLOSE_NONE &&
            think_close.kind == EVAL_THINK_CLOSE_NONE) {
            think_close.kind = close_kind;
            think_close.token_index = ui->generated + 1;
            think_close.remaining_budget = remaining_budget;
            think_close.rank = close_rank;
        }
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            plain_reset_color(use_plain_color);
            ui->generated_tokens[idx] = ui->generated;
            tui_run_clock_stop(ui);
            fprintf(stderr, "ds4-eval: decode failed for %s: %s\n", tc->id, err);
            trace_write_case(trace, cfg, tc, idx, ui->ncases, "ERROR", err,
                             system, question, raw.v ? raw.v : "", think_mode,
                             prompt_tokens, ui->generated, now_sec() - t0, "?",
                             &think_close);
            free(question);
            ds4_tokens_free(&think_close_tokens);
            buf_free(&raw);
            return EVAL_RUN_ERROR;
        }

        size_t len = 0;
        char *text = ds4_token_text(engine, token, &len);
        buf_append(&raw, text, len);
        ui->generated++;
        ui->generated_tokens[idx] = ui->generated;
        tui_run_clock_tick(ui);
        if (generation_in_think && raw.v && strstr(raw.v, "</think>")) {
            generation_in_think = false;
            if (think_close.kind == EVAL_THINK_CLOSE_NONE) {
                think_close.kind = EVAL_THINK_CLOSE_NATURAL;
                think_close.token_index = ui->generated;
                think_close.remaining_budget = remaining_budget;
                think_close.rank = 0;
            }
        }
        double elapsed = now_sec() - ui->phase_start_sec;
        ui->speed_tps = elapsed > 0.001 ? (double)ui->generated / elapsed : 0.0;

        if (tty) {
            stream_append_token_text(ui, text, len, false);
            tui_refresh(ui, ui->in_think ? "thinking" : "answer");
        } else {
            if (plain_in_think && strstr(raw.v ? raw.v : "", "</think>")) {
                plain_in_think = false;
                plain_reset_color(use_plain_color);
            }
            fwrite(text, 1, len, stdout);
            fflush(stdout);
        }
        free(text);
    }
    if (tty) {
        stream_append_token_text(ui, NULL, 0, true);
        tui_draw_stream(ui);
    } else {
        plain_reset_color(use_plain_color);
        if (!raw.v || raw.len == 0 || raw.v[raw.len - 1] != '\n') fputc('\n', stdout);
    }

    char got[EVAL_ANSWER_MAX];
    find_case_answer(tc, raw.v ? raw.v : "", got, sizeof(got));
    snprintf(ui->guess[idx], EVAL_ANSWER_MAX, "%s", got);
    bool pass = answer_matches(tc, got);
    ui->status[idx] = pass ? EVAL_PASSED : EVAL_FAILED;
    ui->generated_tokens[idx] = ui->generated;
    tui_run_clock_stop(ui);
    double sec = now_sec() - t0;
    tui_refresh(ui, pass ? "passed" : "failed");
    trace_write_case(trace, cfg, tc, idx, ui->ncases, pass ? "PASSED" : "FAILED", NULL,
                     system, question, raw.v ? raw.v : "", think_mode, prompt_tokens,
                     ui->generated, sec, got, &think_close);

    if (!tty) {
        printf("%s%s%s got %s expected %s (%.1fs, %d tokens)\n",
               use_plain_color ? (pass ? ANSI_GREEN : ANSI_RED) : "",
               pass ? "PASSED" : "FAIL",
               use_plain_color ? ANSI_RESET : "",
               got, tc->answer, sec, ui->generated);
    }

    if (tty && cfg->pause_ms > 0) usleep((useconds_t)cfg->pause_ms * 1000);
    free(question);
    ds4_tokens_free(&think_close_tokens);
    buf_free(&raw);
    return EVAL_RUN_OK;
}

static int next_pending_case(eval_ui *ui, int start) {
    if (ui->ncases <= 0) return -1;
    for (int off = 0; off < ui->ncases; off++) {
        int i = (start + off) % ui->ncases;
        if (ui->status[i] == EVAL_PENDING) return i;
    }
    return -1;
}

static void log_context_memory(ds4_backend backend, int ctx_size) {
    ds4_context_memory m = ds4_context_memory_estimate(backend, ctx_size);
    fprintf(stderr,
            "ds4-eval: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            ctx_size,
            ds4_backend_name(backend),
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap);
}

static const char *report_status_name(eval_status st) {
    switch (st) {
    case EVAL_PASSED: return "PASSED";
    case EVAL_FAILED: return "FAILED";
    case EVAL_SKIPPED: return "SKIPPED";
    case EVAL_STOPPED: return "STOPPED";
    case EVAL_PREFILL: return "PREFILL";
    case EVAL_THINKING: return "RUNNING";
    case EVAL_PENDING:
    default: return "PENDING";
    }
}

static void print_eval_report(const eval_ui *ui, int ncases, int passed, int failed) {
    char elapsed[32];
    format_run_elapsed(elapsed, sizeof(elapsed), tui_run_clock_visible_sec(ui));

    printf("ds4-eval: %d/%d passed", passed, ncases);
    if (failed) printf(", %d failed", failed);
    printf(", runtime %s\n", elapsed);
    printf("%-3s %-8s %8s %8s %8s %-8s %-8s %s\n",
           "#", "state", "prompt", "gen", "total", "given", "correct", "test");
    for (int i = 0; i < ncases; i++) {
        int prompt_tokens = ui->prompt_tokens ? ui->prompt_tokens[i] : 0;
        int generated_tokens = ui->generated_tokens ? ui->generated_tokens[i] : 0;
        int total_tokens = prompt_tokens + generated_tokens;
        const char *given = ui->guess && ui->guess[i][0] ? ui->guess[i] : "-";
        printf("%3d %-8s %8d %8d %8d %-8s %-8s %s/%s\n",
               i + 1,
               report_status_name(ui->status[i]),
               prompt_tokens,
               generated_tokens,
               total_tokens,
               given,
               ui->cases[i].answer,
               ui->cases[i].source,
               ui->cases[i].id);
    }
}

int main(int argc, char **argv) {
    eval_config cfg = parse_options(argc, argv);
    int ncases = (int)(sizeof(eval_cases) / sizeof(eval_cases[0]));
    if (cfg.question_limit > 0 && cfg.question_limit < ncases) ncases = cfg.question_limit;
    if (cfg.question_limit > (int)(sizeof(eval_cases) / sizeof(eval_cases[0]))) {
        fprintf(stderr, "ds4-eval: only %zu questions are embedded\n",
                sizeof(eval_cases) / sizeof(eval_cases[0]));
        return 2;
    }
    if (!cfg.seed) {
        cfg.seed = (uint64_t)time(NULL) ^
                   ((uint64_t)getpid() << 32) ^
                   (uint64_t)clock();
    }

    FILE *trace = NULL;
    if (cfg.trace_path) {
        trace = fopen(cfg.trace_path, "w");
        if (!trace) {
            fprintf(stderr, "ds4-eval: cannot open trace '%s': %s\n",
                    cfg.trace_path, strerror(errno));
            return 2;
        }
    }

    ds4_engine_options opt = {
        .model_path = cfg.model_path,
        .mtp_path = cfg.mtp_path,
        .backend = cfg.backend,
        .n_threads = cfg.threads,
        .mtp_draft_tokens = 1,
        .mtp_margin = 3.0f,
        .warm_weights = cfg.warm_weights,
        .quality = cfg.quality,
        .cpu_moe = cfg.cpu_moe,
        .n_cpu_moe_layers = cfg.n_cpu_moe_layers,
        .prefill_metal_phases = cfg.prefill_metal_phases,
    };

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) {
        if (trace) fclose(trace);
        return 1;
    }

    int max_prompt_tokens = 0;
    int max_prompt_case = -1;
    cfg.ctx_size = eval_auto_context_size(engine, &cfg, eval_cases, ncases,
                                          &max_prompt_tokens, &max_prompt_case);
    fprintf(stderr,
            "ds4-eval: context auto-sized to %d tokens "
            "(largest prompt=%d tokens, case=%d, generation budget=%d)\n",
            cfg.ctx_size, max_prompt_tokens, max_prompt_case + 1, cfg.max_tokens);
    trace_write_header(trace, &cfg, ncases, max_prompt_tokens);
    log_context_memory(cfg.backend, cfg.ctx_size);

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg.ctx_size) != 0) {
        fprintf(stderr, "ds4-eval: failed to create session\n");
        if (trace) fclose(trace);
        ds4_engine_close(engine);
        return 1;
    }

    eval_ui ui;
    bool split_ui = !cfg.plain && isatty(STDOUT_FILENO);
    tui_start(&ui, eval_cases, ncases, cfg.max_tokens, split_ui);

    uint64_t rng = cfg.seed;
    int rc = 0;
    int next = 0;
    while (next >= 0) {
        tui_consume_input(&ui);
        tui_wait_if_paused(&ui, "idle");
        if (ui.quit_requested) break;
        if (ui.requested_case >= 0) {
            next = ui.requested_case;
            ui.requested_case = -1;
            ui.selection_active = false;
            ui.selected_case = next;
        }

        eval_run_result result = run_one_case(engine, session, &cfg, &ui, trace, next, &rng);
        if (result == EVAL_RUN_ERROR) {
            rc = 1;
            break;
        }
        if (result == EVAL_RUN_QUIT) break;
        /* A successful case should advance to the next pending benchmark.  If a
         * stale quit flag ever survives here, clear it; real q handling either
         * returns EVAL_RUN_QUIT from run_one_case() or is consumed at the top of
         * the next idle iteration. */
        ui.quit_requested = false;
        if (ui.requested_case >= 0) {
            next = ui.requested_case;
            ui.requested_case = -1;
            ui.selection_active = false;
            ui.selected_case = next;
            continue;
        }
        next = next_pending_case(&ui, next + 1);
        if (next >= 0) {
            ui.selected_case = next;
            ui.selection_active = false;
        }
    }

    int passed = 0;
    int failed = 0;
    for (int i = 0; i < ncases; i++) {
        if (ui.status[i] == EVAL_PASSED) passed++;
        else if (ui.status[i] == EVAL_FAILED) failed++;
    }

    if (ui.active) tui_restore();
    print_eval_report(&ui, ncases, passed, failed);
    if (trace) {
        fprintf(trace,
                "===== SUMMARY =====\n"
                "passed: %d\n"
                "failed: %d\n"
                "total: %d\n",
                passed, failed, ncases);
        fflush(trace);
    }

    tui_free(&ui);
    ds4_session_free(session);
    ds4_engine_close(engine);
    if (trace) fclose(trace);
    return rc || failed ? 1 : 0;
}
