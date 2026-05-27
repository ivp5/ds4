/* silv 2026-05-27 task #649 Phase A — organ-skip API smoke.
 *
 * Validates the per-(layer, expert, organ) skip array + loaders.
 * Does NOT exercise the dispatch path (Phase A.1, separate ship).
 *
 * Build (from ivp5_ds4):
 *   cc -O0 -g -Wall -Wextra -I. tmp/20260527_harm_scorer/organ_skip_smoke.c \
 *      ds4_expert_table.o ds4_vqb2_reader.o ds4_polar_reader.o ds4_vqb1_reader.o \
 *      -lm -o tmp/20260527_harm_scorer/organ_skip_smoke
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ds4_expert_table.h"

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); return 1; } \
    printf("  PASS: %s\n", msg); \
} while (0)

int main(void) {
    /* (1) Default state. */
    CHECK(!g_organ_skip_initialized, "default: not initialized");
    CHECK(g_organ_skip_count == 0, "default: count=0");
    CHECK(ds4_organ_should_skip(5, 17, DS4_ORGAN_GATE) == 0,
          "default: query returns 0");

    /* (2) Programmatic set. */
    CHECK(ds4_organ_skip_set(5, 17, DS4_ORGAN_GATE, 1) == 0,
          "set (5, 17, GATE) succeeds");
    CHECK(g_organ_skip_initialized, "now initialized");
    CHECK(g_organ_skip_count == 1, "count=1 after one set");
    CHECK(ds4_organ_should_skip(5, 17, DS4_ORGAN_GATE) == 1,
          "query (5, 17, GATE) returns 1");
    CHECK(ds4_organ_should_skip(5, 17, DS4_ORGAN_UP) == 0,
          "query (5, 17, UP) returns 0 (per-organ isolation)");
    CHECK(ds4_organ_should_skip(5, 17, DS4_ORGAN_DOWN) == 0,
          "query (5, 17, DOWN) returns 0");
    CHECK(ds4_organ_should_skip(5, 18, DS4_ORGAN_GATE) == 0,
          "query (5, 18, GATE) returns 0 (per-expert isolation)");
    CHECK(ds4_organ_should_skip(6, 17, DS4_ORGAN_GATE) == 0,
          "query (6, 17, GATE) returns 0 (per-layer isolation)");

    /* (3) Reset. */
    ds4_organ_skip_reset();
    CHECK(g_organ_skip_count == 0, "count=0 after reset");
    CHECK(ds4_organ_should_skip(5, 17, DS4_ORGAN_GATE) == 0,
          "(5, 17, GATE) cleared after reset");

    /* (4) Env loader. */
    int n = ds4_load_organ_skip_env("3,10,0;3,10,1;7,22,2");
    CHECK(n == 3, "env loader: 3 cells loaded");
    CHECK(g_organ_skip_count == 3, "count=3");
    CHECK(ds4_organ_should_skip(3, 10, DS4_ORGAN_GATE) == 1, "(3,10,GATE)");
    CHECK(ds4_organ_should_skip(3, 10, DS4_ORGAN_UP) == 1, "(3,10,UP)");
    CHECK(ds4_organ_should_skip(3, 10, DS4_ORGAN_DOWN) == 0, "(3,10,DOWN) NOT set");
    CHECK(ds4_organ_should_skip(7, 22, DS4_ORGAN_DOWN) == 1, "(7,22,DOWN)");

    /* (5) Env loader rejects malformed. */
    int err = ds4_load_organ_skip_env("garbage");
    CHECK(err == -1, "env loader: 'garbage' returns -1");

    /* (6) Empty env is OK (count=0, no error). */
    int empty = ds4_load_organ_skip_env("");
    CHECK(empty == 0, "env loader: '' returns 0");

    /* (7) Whitespace + semicolon tolerance. */
    ds4_organ_skip_reset();
    int spaced = ds4_load_organ_skip_env("  1,2,0 ;  ; 1,2,1; ");
    CHECK(spaced == 2, "env loader: whitespace tolerated, 2 cells");
    CHECK(ds4_organ_should_skip(1, 2, DS4_ORGAN_GATE) == 1, "(1,2,GATE)");
    CHECK(ds4_organ_should_skip(1, 2, DS4_ORGAN_UP) == 1, "(1,2,UP)");

    /* (8) Out-of-range refused. */
    int oor = ds4_load_organ_skip_env("999,0,0");
    CHECK(oor == -1, "env loader: layer=999 refused");

    /* (9) CSV loader. */
    const char *csv_path = "/tmp/organ_skip_test_csv.txt";
    FILE *fp = fopen(csv_path, "w");
    if (!fp) { perror("fopen csv"); return 1; }
    fprintf(fp,
        "# test csv\n"
        "0,0,0\n"
        "0,0,1\n"
        "0,0,2\n"
        "\n"
        "  # comment with leading space\n"
        "10,100,1\n");
    fclose(fp);
    int csv_n = ds4_load_organ_skip_csv(csv_path);
    CHECK(csv_n == 4, "csv loader: 4 cells (comments/blanks skipped)");
    CHECK(ds4_organ_should_skip(0, 0, DS4_ORGAN_GATE) == 1, "csv: (0,0,GATE)");
    CHECK(ds4_organ_should_skip(0, 0, DS4_ORGAN_UP) == 1, "csv: (0,0,UP)");
    CHECK(ds4_organ_should_skip(0, 0, DS4_ORGAN_DOWN) == 1, "csv: (0,0,DOWN)");
    CHECK(ds4_organ_should_skip(10, 100, DS4_ORGAN_UP) == 1, "csv: (10,100,UP)");
    remove(csv_path);

    /* (10) NULL/missing handled defensively. */
    CHECK(ds4_load_organ_skip_csv(NULL) == -1, "csv NULL path returns -1");

    /* (11) ds4_organ_should_skip is hot-path inline; out-of-range query
     *      must not crash. */
    CHECK(ds4_organ_should_skip(99999, 0, 0) == 0, "OOR layer returns 0");
    CHECK(ds4_organ_should_skip(0, 99999, 0) == 0, "OOR expert returns 0");
    CHECK(ds4_organ_should_skip(0, 0, 5) == 0, "OOR kind returns 0");

    ds4_organ_skip_print();
    printf("\nALL PASS — Phase A organ-skip data structure ready for dispatch wiring\n");
    return 0;
}
