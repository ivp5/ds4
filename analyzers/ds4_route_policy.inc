// H1744 DS4 router pressure policy table generated from H1742.
// buckets: 0=high, 1=medium, 2=low. target: 100 means clean-top6 containment before rerank.
typedef struct { const char *profile; int bits; int axis_tensor; int target_pct; int width_high; int width_medium; int width_low; } ds4_route_policy_t;
static const ds4_route_policy_t H1744_DS4_ROUTE_POLICIES[] = {
  {"q8_row", 8, 0, 100, 7, 6, 6},
  {"q6_row", 6, 0, 100, 9, 8, 6},
  {"q4_row", 4, 0, 100, 32, 14, 11},
  {"q3_row", 3, 0, 100, 166, 62, 62},
  {"q8_tensor", 8, 1, 100, 8, 7, 7},
  {"q6_tensor", 6, 1, 100, 11, 12, 7},
  {"q4_tensor", 4, 1, 100, 91, 47, 14},
  {"q3_tensor", 3, 1, 100, 253, 224, 161},
};
