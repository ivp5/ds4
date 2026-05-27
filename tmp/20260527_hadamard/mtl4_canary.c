/* silv 2026-05-27 task #653 — MTL4 Hadamard widened-kernel canary.
 *
 * Crash test: dispatches the MTL4-path widened Hadamard kernel twice on
 * deterministic FP16 random data, verifies H×H=I within FP16 precision,
 * reports GPU elapsed and dispatch reduction.
 *
 * If the kernel is broken, this surfaces it as a high max_abs_err.
 * If the MTL4 pipeline init fails, the canary prints the error and exits.
 *
 * Build (from ivp5_ds4):
 *   cc -O3 -ffast-math -fobjc-arc -I. tmp/20260527_hadamard/mtl4_canary.c \
 *      ds4.o ds4_neon_i8mm.o ds4_metal.o ds4_metal_vqb2_fp16.o \
 *      ds4_expert_table.o ds4_inflight.o ds4_moe_route_log.o \
 *      ds4_polar_reader.o ds4_vqb1_reader.o ds4_vqb2_reader.o \
 *      -lm -pthread -framework Foundation -framework Metal \
 *      -o tmp/20260527_hadamard/mtl4_canary
 */
#include <stdio.h>
#include <stdlib.h>
#include "ds4_gpu.h"

int main(int argc, char **argv) {
    uint32_t n_rows = 256;
    uint32_t n_in   = 7168;
    if (argc > 1) n_rows = (uint32_t)atoi(argv[1]);
    if (argc > 2) n_in   = (uint32_t)atoi(argv[2]);
    if (!ds4_gpu_init()) { fprintf(stderr, "ds4_gpu_init failed\n"); return 1; }
    fprintf(stderr, "ds4: dispatching MTL4 hadamard canary n_rows=%u n_in=%u\n", n_rows, n_in);
    const int rc = ds4_gpu_mtl4_hadamard16_canary(n_rows, n_in);
    fprintf(stderr, "ds4: canary result = %d\n", rc);
    return rc ? 0 : 2;
}
