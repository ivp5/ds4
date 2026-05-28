/* fp8_e4m3_lut_canary.c — silv 2026-05-28 #771 B+D FP8 kernel canary
 *
 * Validates FP8_E4M3 decode LUT against reference values before the Metal
 * kernel is written. FP8_E4M3 format (IEEE-style, asymmetric exponent bias):
 *   bit 7    = sign
 *   bits 6-3 = exponent (4 bits, bias = 7)
 *   bits 2-0 = mantissa (3 bits)
 *
 * Decode rules:
 *   v=0x00 or 0x80     → ±0
 *   exp=0, mant!=0     → denormal: (-1)^s × 2^-6 × (mant/8)
 *   exp>0, exp<15      → normal:   (-1)^s × 2^(exp-7) × (1 + mant/8)
 *   exp=15, mant=7     → ±NaN  (asymmetric — only one NaN encoding)
 *   exp=15, mant<7     → normal continued (max = 448 at exp=15,mant=6)
 *
 * Build: cc -O2 -Wall -o fp8_canary fp8_e4m3_lut_canary.c -lm
 * Run:   ./fp8_canary
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Build the 256-entry LUT once. */
static float g_fp8_e4m3_lut[256];

static void build_lut(void) {
 for (int i = 0; i < 256; i++) {
  uint8_t v = (uint8_t)i;
  uint32_t sign = (v >> 7) & 1;
  uint32_t exp  = (v >> 3) & 0xF;
  uint32_t mant = v & 0x7;
  float f;
  if (v == 0x00 || v == 0x80) {
   f = 0.0f;
  } else if (exp == 0) {
   /* Denormal: (-1)^s × 2^-6 × (mant/8)  [mant != 0 since v != 0/0x80] */
   f = (float)mant / 8.0f * 0.015625f;  /* 2^-6 = 1/64 */
  } else if (exp == 15 && mant == 7) {
   /* Single NaN encoding in FP8_E4M3 (asymmetric — different from IEEE). */
   f = NAN;
  } else {
   /* Normal: (-1)^s × 2^(exp-7) × (1 + mant/8) */
   const float p2 = ldexpf(1.0f, (int)exp - 7);
   f = p2 * (1.0f + (float)mant / 8.0f);
  }
  if (sign) f = -f;
  g_fp8_e4m3_lut[i] = f;
 }
}

struct ref {
 uint8_t v;
 float expected;
 const char *note;
};

/* Reference values from the FP8_E4M3 spec (Micikevicius et al. 2022,
 * "FP8 Formats for Deep Learning") + sanity-checks. */
/* FP8_E4M3 with bias=7. Normal value = (-1)^s × 2^(exp-7) × (1 + mant/8).
 * Reference values re-derived against spec — earlier table was wrong. */
static const struct ref refs[] = {
 {0x00,  0.0f,        "+0"},
 {0x80,  0.0f,        "-0 (decodes to 0.0)"},
 {0x01,  0.001953125f, "smallest +denormal (1/512 = 2^-6 × 1/8)"},
 {0x07,  0.013671875f, "largest +denormal (7/512)"},
 {0x08,  0.015625f,   "smallest +normal (2^-6, exp=1)"},
 {0x10,  0.03125f,    "exp=2, mant=0 (2^-5)"},
 {0x38,  1.0f,        "exp=7, mant=0 (2^0 = 1.0)"},
 {0x40,  2.0f,        "exp=8, mant=0 (2^1 = 2.0)"},
 {0x48,  4.0f,        "exp=9, mant=0 (2^2 = 4.0)"},
 {0x78,  256.0f,      "exp=15, mant=0 (2^8)"},
 {0x7E,  448.0f,      "max positive (exp=15, mant=6: 256 × 1.75)"},
 {0xFE, -448.0f,      "max negative"},
 {0xC0, -2.0f,        "exp=8, mant=0, neg (-2.0)"},
 {0xB8, -1.0f,        "exp=7, mant=0, neg (-1.0)"},
};

/* Decode UE8M0 scale byte: scale = 2^(byte - 127). FP8_E8M0 is "unsigned"
 * 8-bit exponent: no sign bit, no mantissa, just a 0-255 exponent with
 * bias 127. byte=127 → 1.0, byte=128 → 2.0, byte=126 → 0.5. */
static float decode_ue8m0(uint8_t v) {
 if (v == 0xFF) return NAN;  /* spec NaN encoding */
 return ldexpf(1.0f, (int)v - 127);
}

int main(void) {
 build_lut();
 int fails = 0;
 printf("=== FP8_E4M3 LUT validation ===\n");
 for (size_t i = 0; i < sizeof(refs)/sizeof(refs[0]); i++) {
  float got = g_fp8_e4m3_lut[refs[i].v];
  float diff = fabsf(got - refs[i].expected);
  int ok = diff < 1e-9f;
  if (!ok) fails++;
  printf("  0x%02x → %14.7e  (expected %14.7e)  %s%s\n",
   refs[i].v, got, refs[i].expected, ok ? "OK" : "FAIL", refs[i].note ? " — " : "");
  if (refs[i].note) printf("              %s\n", refs[i].note);
 }
 /* NaN sanity check */
 float nan_val = g_fp8_e4m3_lut[0x7F];
 printf("  0x7F → %f (isnan=%d) — expect NaN\n", nan_val, isnan(nan_val));
 if (!isnan(nan_val)) fails++;

 printf("\n=== UE8M0 scale decode validation ===\n");
 struct { uint8_t v; float expected; } scale_refs[] = {
  {127, 1.0f},      /* unity */
  {128, 2.0f},
  {126, 0.5f},
  {120, 1.0f/128.0f},  /* 2^-7 */
  {134, 128.0f},        /* 2^7 */
  {0,   ldexpf(1.0f, -127)},  /* smallest */
 };
 for (size_t i = 0; i < sizeof(scale_refs)/sizeof(scale_refs[0]); i++) {
  float got = decode_ue8m0(scale_refs[i].v);
  float diff = fabsf(got - scale_refs[i].expected);
  int ok = diff < 1e-9f * fabsf(scale_refs[i].expected) + 1e-30f;
  if (!ok) fails++;
  printf("  scale 0x%02x → %14.7e  (expected %14.7e)  %s\n",
   scale_refs[i].v, got, scale_refs[i].expected, ok ? "OK" : "FAIL");
 }

 /* Range check: dump the LUT extremes */
 printf("\n=== LUT range check ===\n");
 float lut_min = +1e30f, lut_max = -1e30f;
 int n_finite = 0;
 for (int i = 0; i < 256; i++) {
  float f = g_fp8_e4m3_lut[i];
  if (!isnan(f)) {
   n_finite++;
   if (f < lut_min) lut_min = f;
   if (f > lut_max) lut_max = f;
  }
 }
 printf("  n_finite=%d (expect 254 — two NaN encodings + 252 normal/denorm)\n", n_finite);
 printf("  range: [%.9g, %.9g] (expect [-448, +448])\n", lut_min, lut_max);
 if (n_finite != 254) { printf("  FAIL: expected 254 finite\n"); fails++; }
 /* Tight check: exact bit-equality against the reference extremes. */
 if (lut_min != -448.0f || lut_max != 448.0f) {
  printf("  FAIL: range not exactly [-448, +448]\n"); fails++;
 }

 printf("\n=== Verdict ===\n");
 printf("  FAILS: %d\n", fails);
 return fails == 0 ? 0 : 1;
}
