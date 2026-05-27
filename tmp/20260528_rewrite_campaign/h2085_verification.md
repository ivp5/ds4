# H2085 codex IQ2_XXS scale-bug claim — FALSE ALARM

## Codex H2085 claim (CODEX_SHIFTS.md line 6821)

> "ds4_metal.m's MTL4 IQ2_XXS dequant helper appears to scale every
> group by exactly 0.5× versus the CPU/hot-store formula: CPU uses
> d*(2*h+1)*0.125, while MTL4 uses d*0.125*(0.5+h)"

## Actual MTL4 source (ds4_metal.m line 31727)

```
const float dl = d * (0.5f + (float)(aux32_s >> 28)) * 0.25f;
```

Note the multiplier is **`* 0.25f`** (not `* 0.125f`).

## Algebraic verification

Let h = aux32_s >> 28.

CPU formula:
  d * (2h+1) * 0.125
= d * (2h+1) / 8

MTL4 formula:
  d * (h + 0.5) * 0.25
= d * (h + 0.5) / 4
= d * (2h + 1) / 8       (multiply both num/denom by 2)

Same.

## Conclusion

Codex misread `* 0.25f` as `* 0.125f`. MTL4 IQ2_XXS dequant scale is
CORRECT. No fix required.

Filing as audit-finding for the audit-graph: codex direct-read claims
need source verification before action.

