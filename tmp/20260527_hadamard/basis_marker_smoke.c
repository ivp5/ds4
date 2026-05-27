/* silv 2026-05-27 task #647 — smoke test for hot-store basis-aware marker.
 *
 * Validates:
 *  1. Default state: domain_id == 0, basis bitmasks all 0.
 *  2. Marking a row block sets the underlying bit but getter still returns
 *     0 because domain_id is still 0 (Rule 6 defense-in-depth).
 *  3. After ds4_hot_store_set_calibration_domain(non-zero), getters return
 *     the marked bits.
 *  4. Per-organ isolation: marking gate doesn't affect up/down.
 *  5. Per-(layer, expert) isolation.
 *  6. Out-of-range args refused without crash.
 *
 * Build (from ivp5_ds4):
 *   cc -O0 -g -Wall -Wextra -I. tmp/20260527_hadamard/basis_marker_smoke.c \
 *      ds4_expert_table.o -o tmp/20260527_hadamard/basis_marker_smoke
 */
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include "ds4_expert_table.h"

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); return 1; } \
    printf("  PASS: %s\n", msg); \
} while (0)

int main(void) {
    /* Tiny budget — we don't pin anything, just exercise the bitmask fields. */
    ds4_hot_expert_store *store = ds4_hot_expert_store_alloc(1024 * 1024);
    CHECK(store != NULL, "alloc returned non-NULL");

    /* (1) Default state. */
    CHECK(ds4_hot_store_get_calibration_domain(store) == 0,
          "default domain_id == 0");
    CHECK(ds4_hot_get_gate_basis_hadamard16(store, 5, 17) == 0,
          "default gate basis at (5,17) == 0");
    CHECK(ds4_hot_get_up_basis_hadamard16(store, 5, 17) == 0,
          "default up basis == 0");
    CHECK(ds4_hot_get_down_basis_hadamard16(store, 5, 17) == 0,
          "default down basis == 0");

    /* (2) Mark row block 3 on gate at (5, 17) — domain still 0. */
    CHECK(ds4_hot_mark_basis_hadamard16(store, 5, 17, 0, 3) == 0,
          "mark gate (5,17) row_block 3 succeeds");
    CHECK(ds4_hot_get_gate_basis_hadamard16(store, 5, 17) == 0,
          "Rule 6 gate: domain_id == 0 ⇒ getter returns 0 even after mark");

    /* (3) Now set domain — bits should surface. */
    ds4_hot_store_set_calibration_domain(store, 0xdeadbeef);
    CHECK(ds4_hot_store_get_calibration_domain(store) == 0xdeadbeef,
          "domain id round-trips");
    uint64_t gate_mask = ds4_hot_get_gate_basis_hadamard16(store, 5, 17);
    CHECK(gate_mask == (1ULL << 3),
          "after domain set, gate basis at (5,17) has bit 3");

    /* (4) Per-organ isolation: gate mark didn't touch up/down. */
    CHECK(ds4_hot_get_up_basis_hadamard16(store, 5, 17) == 0,
          "up basis at (5,17) untouched by gate mark");
    CHECK(ds4_hot_get_down_basis_hadamard16(store, 5, 17) == 0,
          "down basis at (5,17) untouched by gate mark");

    /* Mark up + down at SAME (layer, expert), different row blocks. */
    CHECK(ds4_hot_mark_basis_hadamard16(store, 5, 17, 1, 7) == 0,
          "mark up (5,17) row_block 7");
    CHECK(ds4_hot_mark_basis_hadamard16(store, 5, 17, 2, 15) == 0,
          "mark down (5,17) row_block 15");
    CHECK(ds4_hot_get_gate_basis_hadamard16(store, 5, 17) == (1ULL << 3),
          "gate still bit 3 only");
    CHECK(ds4_hot_get_up_basis_hadamard16(store, 5, 17) == (1ULL << 7),
          "up has bit 7");
    CHECK(ds4_hot_get_down_basis_hadamard16(store, 5, 17) == (1ULL << 15),
          "down has bit 15");

    /* OR-into existing — mark same organ another row block. */
    CHECK(ds4_hot_mark_basis_hadamard16(store, 5, 17, 0, 5) == 0,
          "second gate mark at (5,17) row_block 5");
    CHECK(ds4_hot_get_gate_basis_hadamard16(store, 5, 17) ==
          ((1ULL << 3) | (1ULL << 5)),
          "gate bitmask OR'd into");

    /* (5) Per-(layer, expert) isolation. */
    CHECK(ds4_hot_get_gate_basis_hadamard16(store, 5, 18) == 0,
          "gate basis at neighbor expert (5,18) is 0");
    CHECK(ds4_hot_get_gate_basis_hadamard16(store, 6, 17) == 0,
          "gate basis at neighbor layer (6,17) is 0");

    /* (6) Out-of-range args refused without crash. */
    CHECK(ds4_hot_mark_basis_hadamard16(store, 999, 0, 0, 0) == -1,
          "out-of-range layer refused");
    CHECK(ds4_hot_mark_basis_hadamard16(store, 0, 9999, 0, 0) == -1,
          "out-of-range expert refused");
    CHECK(ds4_hot_mark_basis_hadamard16(store, 0, 0, 5, 0) == -1,
          "invalid kind refused");
    CHECK(ds4_hot_mark_basis_hadamard16(store, 0, 0, 0, 100) == -1,
          "out-of-range row_block refused");
    CHECK(ds4_hot_get_gate_basis_hadamard16(store, 999, 0) == 0,
          "out-of-range layer getter returns 0");

    /* (7) Domain reset to 0 should re-hide the flags (Rule 6 enforced). */
    ds4_hot_store_set_calibration_domain(store, 0);
    CHECK(ds4_hot_get_gate_basis_hadamard16(store, 5, 17) == 0,
          "domain reset to 0 ⇒ gate basis hidden again");

    /* (8) NULL store handling. */
    CHECK(ds4_hot_get_gate_basis_hadamard16(NULL, 0, 0) == 0,
          "NULL store ⇒ getter returns 0");
    CHECK(ds4_hot_store_get_calibration_domain(NULL) == 0,
          "NULL store ⇒ domain getter returns 0");
    CHECK(ds4_hot_mark_basis_hadamard16(NULL, 0, 0, 0, 0) == -1,
          "NULL store ⇒ mark returns -1");
    /* set_calibration_domain on NULL — must not crash. */
    ds4_hot_store_set_calibration_domain(NULL, 1);

    ds4_hot_expert_store_free(store);
    printf("\nALL PASS — basis-aware marker honors Rule 6 gate\n");
    return 0;
}
