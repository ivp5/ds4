# Codex Journal — 2026-06-03 Fidelity Runtime

## Sequential agent2 read

Read `/Users/silv/cl/tlp/montyneg/agent2/agent2_shifts.md` from the prior frontier at line 939 through EOF line 993. A39-A42 overturn A38: the earlier 6.7x claim was a wrong-path comparison against prefill materialization, while the real decode matmul is already fused, vectorized over packed codes, and cooperative-X optimized. Treat the naive agent2 VQ-GEMV integration as obsolete, not as an implementation target.

## Runtime patch

Changed the existing shared gate/up FP8 fused SwiGLU path from env-enable to default-on outside `--quality`, gated by native FP8 pack support and explicit disable env. This is not overfitting AIME or importing the A38 loophole; it is a measured hot-path fusion already present in the runtime.

## Measurements

- Baseline n6 split2 before default fusion: 5.00 t/s, total 199.560 ms mean, 197.778 ms excluding first.
- Env-enabled shared gate/up n6: 5.02 t/s, total 198.823 ms mean, 196.438 ms excluding first.
- Default-on shared gate/up n12: 4.93 t/s, total 202.450 ms mean, 195.227 ms excluding first.
- Explicitly disabled n12 after patch: 4.69 t/s, total 213.126 ms mean, 201.532 ms excluding first.

## Verdict

Keep shared gate/up FP8 fused SwiGLU default-on with `DS4_METAL_DISABLE_SHARED_GATE_UP_FP8_SWIGLU_FUSION=1` as the escape hatch. Keep CBSRAM, preweight-mid, shared-down FP8 HC, and A38-style VQ-GEMV integration out of defaults.

## FP8 attention output one-CB probe

The existing `--fp8-attn-out-icb-canary` showed one-CB barrier headroom at DS4 dimensions (`1.28x`, bit-exact), so I wired `ds4_gpu_attention_output_fp8_e4m3_e8m0_hc_onecb_tensor` and tested it in full H3355 decode. Full-runtime n12 regressed versus the current gate/up default path (`4.75 t/s` active one-CB vs `4.93 t/s` baseline; total excluding first `196.536 ms` vs `195.227 ms`). Decision: keep the wrapper opt-in behind `DS4_ENABLE_FP8_ATTN_OUT_ONECB_HC=1`, not default.

## D8F rank1 split packet ICB

Implemented a four-command packet ICB for the rank1 split sidecar sequence and gated it behind `DS4_D8F_RANK1_SPLIT_PACKET_ICB=1`. L40 isolated canary improved selected path from `9.147 ms` to `5.318 ms` with zero mismatch, but full H3355 decode stayed mixed (`4.95 t/s` packet vs `5.04 t/s` no-packet split vs old default `4.93 t/s`). Decision: keep opt-in; do not default rank1 split or packet split.

## D8F tile knobs

Full H3355 n12 confirmed current D8F tile defaults. Disabling gate/up tile4 regressed to `4.59 t/s`; disabling down tile16 regressed to `4.79 t/s`; tile8 fallback regressed to `4.60 t/s`. Current defaults remain correct.

## Agent2 shifts A44-A46

Continued the sequential read of `/Users/silv/cl/tlp/montyneg/agent2/agent2_shifts.md` from the recorded frontier after line 993 through EOF line 1056. A44 revises the earlier A43 conclusion: agent2's parallel-reduction MoE mapping is not a regression, but the fair measured gain over codex's actual six-expert uint32/cooperative-X MoE kernel is only `1.23x`, implying roughly `1.1x` whole-decode speedup under the stated MoE fraction. A45 says attention already uses the parallel-reduction mapping in codex's FP8 projection kernel, so attention is not another agent2 lever. A46 closes the kernel-mapping lane and selects spec-decode/MTP as the next multiplicative direction, with acceptance rate as the key measurement.

## H3355 MTP acceptance stats

Added session-level MTP counters and CLI summaries (`DS4_MTP_STATS=1`) so acceptance can be measured directly instead of inferred from noisy interleaved token logs. Build passed. H3355 short `Hi` probe hit rate was `2/7 = 0.286`; forced spec sanity reported `calls=5 ready=5 first_hit=1 first_miss=4 drafted=5 committed=1`; AIME-style probe `tmp/20260527_dsml_aime/aime_p01_easy/prompt.txt` hit `19/31 = 0.613`. Decision: keep current MTP speculation disabled by default on external D8F packs. The MTP head has useful agreement on AIME-style text, but current verifier cost is still too high; next MTP work must reduce verifier cost or batch verification, not default-enable forced spec.

## MTP decode2 batch-output canary

Added opt-in `DS4_MTP_DECODE2_BATCH_OUTPUT=1` to test whether strict decode2 could keep exact one-token layer/cache order while batching only the final two output heads. Build passed, but the AIME-style H3355 strict A/B rejected it: baseline strict exact produced `3.85 t/s` with `decode2=5`, while batch-output produced `3.77 t/s` with `decode2_batch_out=3` and diverged continuation text. Decision: keep this diagnostic opt-in only; output-head batching is not a free verifier-cost cut unless a row-exact variant is proven.

## MTP decode2 fused-output canary

Added opt-in `DS4_MTP_DECODE2_FUSED_OUTPUT=1`, a row-exact variant that keeps the one-row output head but fuses both output heads/top-k reductions/row0 logits copy into one command buffer. Build passed and continuation matched the strict baseline over the sampled AIME-style span, but speed regressed: baseline strict exact `3.82 t/s`, fused-output `3.60 t/s`. Command buffers dropped `62 -> 57`, so the rejected hypothesis is concrete: command-buffer reduction alone does not offset resource-hazard/scratch-reuse pressure in strict decode2. Keep fused-output opt-in only.

## D8F rank1 split long A/B

Ran the longer same-build H3355 `n=24` A/B that had been pending. Default path produced `4.83 t/s`, token total excluding first `203.938 ms`, execute excluding first `197.887 ms`, with `packet_icb_hit=1032 miss=43`. `DS4_D8F_RANK1_SPLIT_SIDECAR=1` regressed to `4.48 t/s`, token total excluding first `216.792 ms`, execute excluding first `211.284 ms`, with `packet_icb_hit=360 miss=715`. Decision: do not default rank1 split or packet split; keep both opt-in diagnostics.

## verifier-branch tool + q-fusion verdict
- Q/head-norm/RoPE fusion canary stayed opt-in: `full_decode_h3355_q_head_norm_rope_n12_20260603T122438Z.log` produced 4.80 t/s versus 4.86 t/s baseline, same continuation, no default change.
- Integrated `tools/ds4_verifier_branch.py` as a self-contained arithmetic verifier/selector for verifier-guided branch tests.
- The tool infers scalar bindings from prior trace clauses, flags codec-shaped arithmetic contradictions such as `v = 18; t = 14; D = v*t = 18`, and selects the first verifier-clean alternative.
- Validation: `python3 tools/ds4_verifier_branch.py --self-test` passed; `python3 -m py_compile tools/ds4_verifier_branch.py` passed.

## verifier-branch run harness
- Added `tools/ds4_verifier_branch_run.py`, a shell-free severe-test harness around `./ds4` that implements the agent2 branch contract without native KV branching: initial greedy generation, scan, cut at first verifier flag, rerun `prompt + clean_prefix` at branch temperature, select the first verifier-clean branch, then greedy-continue from the selected prefix.
- This is deliberately slower than native branch-from-position because it re-prefills for every branch, but it directly tests the pass-axis hypothesis before core graph-state surgery.
- The harness defaults to raw prompt replay so generated assistant prefix is replayed as text, not rewrapped as a new user message; `--chat-template` remains available for explicit chat-template experiments.
- Selection now requires positive verifier evidence in the replacement branch, not merely absence of flags, so empty evasive branches cannot pass the selector.
- Validation: `python3 tools/ds4_verifier_branch_run.py --self-test` passed; `python3 -m py_compile tools/ds4_verifier_branch.py tools/ds4_verifier_branch_run.py` passed.

## verifier-branch orchestration stub validation
- Added `tmp/20260603_fidelity_runtime/fake_ds4_verifier_branch.py` to exercise the full wrapper without GPU/model load.
- End-to-end wrapper validation passed: initial fake generation emitted `D = v * t = 18`; branch 0 repeated the bad step and was rejected; branch 1 emitted `D = v * t = 252`, was selected, and greedy continuation produced `final answer is 252`.
- Run artifact: `tmp/20260603_fidelity_runtime/fake_branch_runs/20260603T123532Z/final.txt`; `selected.scan.json` shows `flags=[]` and inferred bindings `v=18`, `t=14`, `D=252`.

## verifier linear-solve hardening
- Hardened `tools/ds4_verifier_branch.py` beyond chained numeric arithmetic: it now normalizes implicit multiplication (`14v`, `2x`) and solves simple one-variable linear equalities from prior clauses.
- This catches wrong solve-step cascades such as `14v = 252. v = 19` and `2x + 3 = 17. x = 8`, while accepting `v = 18` and `x = 7` respectively.
- Fixed a false-flag found by the new tests: suffix extraction was treating `2x + 3` as the trailing numeric `3`; arithmetic extraction now stops instead of stripping into a misleading constant when an expression contains unresolved variables.
- Validation: expanded `--self-test` passed; wrapper `--self-test` passed; fake end-to-end branch run `tmp/20260603_fidelity_runtime/fake_branch_runs/20260603T123814Z/final.txt` selected the corrected branch and finished with zero verifier flags.

## rendered replay + live H3355 no-branch canary
- Patched `tools/ds4_verifier_branch_run.py` with `--render-chat`, `--system`, and `--think-mode` so the harness can render the DeepSeek chat template once, then replay `rendered_chat + assistant_prefix` without wrapping generated assistant text as a new user prompt.
- Validation: rendered-chat fake run `tmp/20260603_fidelity_runtime/fake_branch_runs/20260603T124030Z/` selected the corrected branch and finished with `flags=[]`.
- Ran a guarded real H3355 p01 canary: `tmp/20260603_fidelity_runtime/h3355_verifier_branch_runs/20260603T124126Z/`. It completed safely at `generation: 4.97 t/s`, but did not branch because `initial.scan.json` had no verifier claims/flags; the p01 file is an answer-prefix/confusion probe, not yet a useful branch trace at 128 generated tokens.
- Reconstructed agent2 margin trace `margins_20260603T053350.json` into `tmp/20260603_fidelity_runtime/margins_20260603T053350_selected_text.txt`; this exposed a detector false positive on relative arithmetic (`7*11=77, then +2=79`).
- Hardened `tools/ds4_verifier_branch.py` for relative arithmetic members that begin with prose plus an operator (`then +2=79`), while preserving variable expressions like `v+2`.
- Validation: expanded detector self-test passed; wrapper self-test passed; rescanning the reconstructed margin trace now yields `flags=0`, `claims=3`.

## agent2_shifts sequential tail A61
- Continued `/Users/silv/cl/tlp/montyneg/agent2/agent2_shifts.md` from the prior verified frontier at line 1283 and read through EOF line 1297.
- New A61 corrects the earlier A53 map: codex has not abandoned speed; current q_head_norm_rope, rank1_split/recbuf/counter, packet fusion, and flatpack runtime probes are active speed work but remain stuck at roughly `4.41-4.99 t/s`.
- A61 strengthens the codec-bound conclusion: three independent lanes now point below runtime dispatch as the bottleneck, so the open performance seam is re-quantization / algorithmic decode structure that reduces op-count or serial chain rather than more dispatch tuning.

## H3369-H3373 projection split of rejected final-answer rollbacks
- Added `tools/ds4_sparse_projection_splice.py`, a projection-specific D8F splicer and hardlink pack builder, because H3366-H3368 full expert rollbacks were too coarse: they changed gate/up/down together and did not isolate which projection moved the compact `m+n=277` failure.
- Materialized self-contained hardlink packs under 52GB logical: H3369 top-four gate-only rollback (`51966975628` bytes), H3372 top-four up-only rollback (`51967434380` bytes), and H3373 top-four down-only rollback (`51966918284` bytes). All have 43 D8F files and no symlinks.
- Focused H3365 compact final-answer gate results: H3364 route trace had top1 `0`, `277` absent, margin `0.084154`; H3369 gate-only worsened wrong-token margin to `0.168468` and kept `277` absent; H3372 up-only nearly tied `0`/`2` with margin `0.000240` but still kept `277` absent.
- Offline comparison of prior full rollbacks: H3366 full L33/E250 surfaced `277` at rank 20, H3367 top4 full at rank 18 with top1 `2`, H3368 top8 full at rank 19 with top1 `2`, while H3304 base had `277` at rank 8. This means full rollback can surface the target but also damages H3364’s local distribution; projection split shows the top4 up projection is the main `0`→`2` pressure, not a solution.
- H3373 down-only is built but not GPU-tested yet because an agent2-owned DS4 run was live (`ivp5_ds4/ds4 ... aime_p03_easy ...`); do not kill it. Next safe action when GPU is free: run the same H3365 compact final-answer gate on H3373, then decide whether down projection carries the target-rank improvement from full rollback.

## H3373-H3377 split completion
- Completed H3373 after the GPU window reopened: top4 down-only still missed `277` with top1 `0` and margin `0.067698`.
- Built and tested the remaining top4 projection-pair splits. H3374 up+down moved the wrong top1 to `2` with margin `0.195728` but kept `277` absent; H3375 gate+up missed with top1 `0`, margin `0.120373`; H3376 gate+down missed with top1 `0`, margin `0.050699`.
- Projection split verdict: no single projection or projection pair recovers `277`; only full gate+up+down rollback surfaces it, weakly at rank 18-19. The up projection carries most of the `0` vs `2` movement, but the target token needs full-projection nonlinearity and remains too low.
- Built H3377 as the reverse-anchor experiment: base H3304, then splice only H3364's known clean gate repairs L25/E237 gate and L26/E165 gate. H3377 is self-contained, no symlinks, and `51939492885` logical bytes, lower than H3364.
- H3377 compact final-answer validation is pending because agent2 reacquired the GPU for a DS4 run. Next safe gate: run H3365 `h3365_fail_free_equations_277`; if H3377 keeps H3304's `277` rank-8 signal, then run H3326/H3325 algebra gates against H3377.

## H3377-H3379 reverse-anchor rejection
- Ran H3377 after the GPU reopened. It did not preserve H3304's compact `277` rank-8 signal: target became absent from top-64, top1 remained `0`, margin `0.084570`.
- Isolated the two clean gate repairs on the H3304 anchor. H3378 (L25/E237 gate only) also made `277` absent with margin `0.082596`; H3379 (L26/E165 gate only) also made `277` absent with margin `0.087816`.
- This rejects the simple reverse-anchor path: the same gate repairs that helped H3353/H3355 algebra locally destroy H3304's compact final-answer rank signal. H3364 remains the active balanced pack; H3304 is useful as a diagnostic anchor, not as a promotion base.
- Updated `/Users/silv/cl/tlp/montyneg/ds4/codec_registry_20260603/reports/H3369_H3377_PROJECTION_SPLIT_REPORT.md` with the full H3369-H3379 projection/reverse-anchor matrix.

## H3380 — agent2_shifts sequential read + VQB2 parallel-reduction runtime patch

Sequentially read `/Users/silv/cl/tlp/montyneg/agent2/agent2_shifts.md` from line 1 through line 1370 because no prior counted frontier was found. Relevant implementation signal: the only concrete, bounded speed patch from agent2's handoffs is the VQB2 decode MoE parallel-reduction mapping (`PARALLEL_REDUCTION_MOE_HANDOFF.md`): measured +1.23x on the MoE GEMV at real six-expert occupancy, expected small total gain, default must stay off.

Implemented `DS4_VQB2_PARALLEL_DECODE=1` in `ds4_metal.m` as an opt-in VQB2 decode matmul sibling path. The new Metal kernel maps one threadgroup per output row/expert, splits `n_pairs` across 256 threads, reduces with `simd_sum`, preserves packed VQB2 decoding for K=4/16/64/256, supports packet/output stride and per-slot X stride, and is wired through standalone direct dispatch, PATH_FUSED direct MTL4 dispatch, and batched ICB replay. Defaults are unchanged when the env is unset.

Validation: `make ds4_metal.o` passed; extracted runtime MSL compiled through `newLibraryWithSource`; standalone correctness canary passed for K=16 and K=256 (`bad=0`, max_rel < 4e-4); `make ds4` linked; `./ds4 --help` startup smoke passed; `git diff --check -- ds4_metal.m` passed. No full model run was launched.
## 2026-06-03T22:55 JST — real D8F MPSGraph LUT selected-down canary

- Read agent2 handoffs fully: dispatch headroom is M1 classic/residency plus MPSGraph/ANE loophole, not MTL4-ML on M1; LUT decomposition is the concrete route: `x[groups,8] @ codebook[8,k]`, gather small table by VQ indices, reduce groups.
- Implemented `ds4_mpsgraph.m` and `--d8f-mpsgraph-lut-down-canary <d8f> [EXPERTS_CSV [rows [rounds [mode]]]]`. It opens real D8F records, applies down activation scales before LUT contraction, builds real codebook/indices constants, verifies against CPU reference, then times compiled/preallocated `MPSGraphExecutable`.
- Validation on H3364/H3355 active D8F L26/E165:
  - fp32 exact full-row: `bad=0`, `max_abs=1.78814e-07`, `us/op=872.267`.
  - fp16 full-row: `bad=0`, `max_abs=0.000645339`, `us/op=710.133`.
  - selected six experts `165,0,1,2,3,4`: `bad=0`, `us/op=2302.492`, `logical_MB/op=33.858`.
- Same selected-six classic baseline: `33.426 ms / 5 rounds = 6685.2 us/op`; MPSGraph selected LUT is ~2.9x faster for full down projection. This is the first real-codec evidence that the LUT loophole beats the current classic down organ on M1 full-row shape.
- Small-row caveat: at 128 rows, classic remains faster because MPSGraph dispatch/constant gather overhead dominates. The production target should therefore fuse full selected experts/layers or otherwise amortize graph dispatch; do not adopt the one-expert/partial-row form as a runtime endpoint.

## 2026-06-03T23:10 JST — MPSGraph D8F gate/up canary and memory-floor boundary

- Extended the real-D8F MPSGraph LUT canary from down to gate+up with on-graph clamp, sigmoid/SwiGLU, and selected-expert concatenation. The shared LUT builder now supports both 2048-input down and 4096-input gate/up projections.
- Corrected MPSGraph index handling to preserve D8F sentinel semantics: invalid codes map to an added zero codebook column, so MPSGraph gather and the CPU reference both skip sentinel blocks instead of indexing out of range.
- Validation after correction: `make ds4` passed; selected-six down full-row fp16 passed with `bad=0`, `us/op=2448.067`, `logical_MB/op=33.861`; selected-six gate/up fp16 passed with `bad=0`, `us/op=2534.725`, `logical_MB/op=72.741`; prior single-expert fp32 gate/up exactness was `bad=0`, `max_abs=1.49012e-07`.
- Performance boundary: MPSGraph down remains a real win versus classic selected-six down (`2448 us/op` now, prior classic `6685 us/op`). Gate/up is exact but not conclusively ahead of the current classic selected-six gate/up (`2535 us/op` after sentinel correction versus prior classic `2710 us/op`, within noisy/local-run range and still burdened by expanded int32 indices plus two projections).
- Runtime implication: MPSGraph standard gather does not reach the packed-index memory floor. The immediate production candidate is selected/full-row down graph caching; gate/up needs either packed-index/custom gather removal, ANE/CoreML palettized table execution, or overlap with another engine before it is a defensible hot-path replacement.

## 2026-06-03T23:21 JST — iOS26 ANE LUT boundary and prefill overlap

- Corrected the agent2 ANE gather probe locally: CoreML MIL `gather_along_axis` requires the iOS17+ `validate_indices` field, and this host is macOS 26.6, so the local canary now defaults to `opset=iOS26` and `deployment=macOS26`.
- Explicit DS4-shaped CoreML gather/reduce does run at B=1, but it is decode-dead: `tmp/20260603_mpsgraph_ane/ane_lut_gather_fixed_b1_ios26_py313_20260603T231947.log` measured `17515.448 us/op` for `4096 -> 4096`.
- Batched explicit gather is worse structurally, not merely by tuning: CoreML does not broadcast gather indices across batch, so B=16/N=2048 requires `67.109 MB` of materialized indices and measured `196979.927 us/op`.
- Therefore "ANE LUT viable" means CoreML's palettized matmul/LUT backend is viable, not raw DS4 MIL gather/reduce. Exact D8F gather on ANE is not a hot path unless CoreML gains packed/shared index semantics or the runtime supplies a custom zero-copy primitive.
- The useful ANE seam is high-B prefill overlap: local iOS26 CoreML 4-bit-palettized matmul plus MLX GPU split-expert canary at `B=2048, D=4096, N=2048, experts=6` measured ANE `91.190 ms`, GPU `46.049 ms`, serial `137.239 ms`, concurrent `105.123 ms`, speedup `1.306x`.
- Runtime implication: keep MPSGraph for exact D8F down/gateup graph canaries; pursue ANE as a parallel prefill organ for palettized/converted expert subsets, not as direct decode gather. The next falsifier is input/output movement: if CoreML cannot consume DS4 hidden states without a GPU-to-CPU copy, overlap loses most of the theoretical gain.

## 2026-06-03T23:28 JST — CoreML IOSurface ingress from DS4-style MTLBuffer

- Read the macOS 26.5 SDK headers: CoreML exposes no public `MTLBuffer` feature input, but `MLMultiArray initWithPixelBuffer:shape:` wraps IOSurface-backed one-component FP16 pixel buffers, and `MLPredictionOptions.outputBackings` can use `MLMultiArray`/`CVPixelBuffer` backing when the model accepts it.
- Added `tmp/20260603_mpsgraph_ane/make_coreml_prefill_model_ios26.py` and `tmp/20260603_mpsgraph_ane/coreml_iosurface_ingress_canary.m` to test the runtime seam directly: Metal writes a DS4-shaped FP16 `MTLBuffer` into an IOSurface texture, CoreML consumes that IOSurface as an `MLMultiArray`, and output uses an IOSurface-backed `MLMultiArray`.
- B=256 canary: output backing was used and `predict_ms=0.760`, `fill_predict_ms=1.083`, showing the ObjC/IOSurface path is much lower overhead than Python `numpy` input.
- B=2048 canary with post-run output sample: output backing was used, first output halfwords were nonzero (`0xb55d 0x2e2a 0xb22d 0xb78a`), and 20-round timing measured `predict_ms=4.593`, `fill_predict_ms=4.729`, `copy_predict_ms=4.705`.
- The DS4-style buffer-to-IOSurface ingest cost is therefore about `0.111 ms` for a `2048 x 4096` FP16 hidden-state copy into CoreML input. This does not block ANE prefill overlap; the harder remaining boundary is model generation/caching for real per-layer expert subsets and merging ANE outputs back into the DS4 prefill accumulation without serializing GPU work.

## 2026-06-03T23:36 JST — ANE/MPSGraph scheduling and cache-state probe

- Added `tmp/20260603_mpsgraph_ane/ane_mpsgraph_schedule_canary.m`, which wraps one shared `MTLBuffer.contents` pointer as a CoreML `MLMultiArray` while also feeding the same `MTLBuffer` to MPSGraph. This directly tests unified-memory scheduling/cache effects instead of comparing unrelated Python buffers.
- Correct architecture model: ANE, GPU/MPSGraph, and CPU share DRAM and some system-level memory/cache/TLB state, but not a simple CPU-style private-cache hierarchy. The useful levers are shared allocation residency, SLC/page/TLB warmth, command scheduling, and avoiding staging copies.
- B=256 schedule canary passed with output backing used and showed launch-noisy but strong overlap: ANE `1.003 ms`, MPSGraph `2.008 ms`, concurrent `0.831 ms`.
- B=2048 r5: ANE `4.601 ms`, MPSGraph `6.230 ms`, ANE→MPSGraph inner `5.248 ms`, MPSGraph→ANE inner `10.201 ms`, concurrent `8.651 ms`, speedup `1.252x`.
- B=2048 r20: ANE `4.561 ms`, MPSGraph `6.026 ms`, ANE→MPSGraph inner `8.066 ms`, MPSGraph→ANE inner `12.381 ms`, concurrent `5.944 ms`, speedup `1.781x`.
- Added CPU eviction controls. Evicting 64/256/512 MiB after ANE before MPSGraph pushed ANE→MPSGraph inner time into the `9.1-9.7 ms` range, while no-evict ANE→MPSGraph often stayed `5-7 ms`. This is evidence that cache/scheduling state matters, but the current canary has order contamination; the next rigorous test must counterbalance/randomize permutation order and record per-round distributions.
- Queued the cache work in `tmp/20260603_mpsgraph_ane/CACHE_ARCHITECTURE_EXPLORATION_QUEUE.md`. Current production direction remains: ANE high-B prefill shared-expert organ first, MPSGraph exact D8F down/gateup graph-cache exploration second, raw ANE D8F gather rejected for now.

## 2026-06-03T23:43 JST — real H3355 shared expert CoreML/ANE canary

- Added `tmp/20260603_mpsgraph_ane/make_coreml_shared_expert_from_nrpk.py`, which reads the H3355 non-routed pack, dequants real shared-expert FP8 E4M3 weights with E8M0 tile scales, and emits a CoreML shared-expert graph for layer-local `w1`, `w3`, and `w2`.
- Added `tmp/20260603_mpsgraph_ane/validate_coreml_shared_expert_ones.py`, a cheap fidelity canary that compares CoreML output against a direct dequant reference on an all-ones hidden-state batch.
- 4-bit CoreML palettization is not fidelity-safe on this canary: `max_abs=0.369388`, `rms=0.0841229`, `ref_rms=0.427915`, roughly `19.7%` RMS/reference RMS.
- 8-bit CoreML palettization is the viable candidate so far: `max_abs=0.0198364`, `rms=0.0050416`, `ref_rms=0.427915`, roughly `1.18%` RMS/reference RMS.
- Real layer-0 shared expert B=2048 8-bit CoreML/ANE timing measured `predict_ms=11.618`, `fill_predict_ms=12.181`, `copy_predict_ms=12.053`, with CoreML output backing accepted.
- Real shared expert ANE overlapped with same-shape MPSGraph dense work at B=2048: ANE `12.200 ms`, MPSGraph `14.896 ms`, serial `27.096 ms`, concurrent `15.980 ms`, speedup `1.696x`.
- Production implication: shared-expert ANE prefill is not blocked by model construction or ingress. The only fidelity-plausible CoreML shared path so far is 8-bit; the next hard gate is comparing real `batch_ffn_norm` samples and merging ANE shared output with GPU/D8F routed output without CPU serialization.

## 2026-06-03T23:50 JST — counterbalanced ANE/MPSGraph cache canary

- Added `tmp/20260603_mpsgraph_ane/ane_mpsgraph_counterbalanced_canary.m`, a randomized per-trial scheduler harness for real CoreML/ANE shared-expert models. It records p50/p90 for ANE-only, MPS-only, evicted MPS, ANE→MPS, ANE→evict→MPS, MPS→ANE, serial, and concurrent cases.
- The harness also has `same` and `separate` input modes. `same` feeds CoreML and MPSGraph from the same `MTLBuffer`; `separate` gives MPSGraph a distinct but equal buffer. This directly tests whether the win is shared-allocation residency rather than generic ANE/GPU overlap.
- Real H3355 layer-0 B=2048 8-bit shared model, no eviction: same-buffer p50 MPS-only `23.452 ms`, same-buffer ANE→MPS p50 `17.840 ms`; separate-buffer MPS-only `15.075 ms`, separate-buffer ANE→MPS `19.674 ms`.
- That pattern supports the user's virtual-bandwidth hypothesis in the narrow form: ANE work can leave shared allocation/cache/page/scheduler state that makes the immediately following MPSGraph pass cheaper. It is not a private-cache warmup story because the effect disappears or reverses when the MPS input is a separate allocation.
- Overlap remains useful even when cache behavior is noisy: p50 overlap speedups were `1.882x` same/no-evict, `1.566x` same/256MiB-evict, `1.608x` separate/no-evict, and `1.495x` separate/256MiB-evict.
- The eviction results are not yet promotion-grade. Separate-buffer 256MiB eviction behaved coherently (`mps_after_ane` `16.944 ms` → `mps_after_ane_evicted` `22.850 ms`), but same-buffer 256MiB had order/load noise (`17.294 ms` → `14.847 ms`). Next evidence must run more trials under lower load, include Metal eviction, and pair ANE shared with real MPSGraph D8F down/gateup instead of synthetic dense.

## 2026-06-04T00:00 JST — ANE shared + exact D8F routed MPSGraph canary

- Added `tmp/20260603_mpsgraph_ane/ane_d8f_routed_counterbalanced_canary.m`, which composes real D8F gate, up, SwiGLU, and down LUTs into one MPSGraph executable and schedules it against the real 8-bit CoreML/ANE shared-expert model.
- The harness is same-layer shaped: CoreML/ANE consumes the B=2048 hidden-state buffer for the shared expert while MPSGraph consumes the first row of either the same hidden-state `MTLBuffer` or a separate equal buffer for exact routed D8F. This is the production dependency boundary: shared and routed branches can run in parallel before merge.
- Exactness passed. L26/E165 single-expert routed graph had `bad=0`, `rms=1.45568e-05`; selected six experts `165,0,1,2,3,4` had `bad=0`, `rms=3.30178e-05`.
- Single-expert routed D8F is too small for major overlap: same/no-evict p50 D8F `2.028 ms`, ANE `12.886 ms`, concurrent `12.719 ms`, overlap speedup `1.172x`.
- Six-expert routed D8F is the useful prefill scale. Same/no-evict p50: ANE `12.269 ms`, D8F `7.270 ms`, serial measured `21.422 ms`, concurrent `14.133 ms`, speedup `1.383x`.
- Six-expert separate/no-evict p50: ANE `14.749 ms`, D8F `6.914 ms`, serial `19.865 ms`, concurrent `13.535 ms`, speedup `1.600x`. Same/256MiB p50 speedup was `1.613x`; separate/256MiB was `1.355x`.
- The synthetic-dense cache warmup result does not transfer cleanly to exact D8F routed. For six experts, same/no-evict D8F-only `7.270 ms` versus D8F-after-ANE `7.760 ms`; separate/no-evict `6.914 ms` versus `7.273 ms`. Treat ANE+D8F as a parallel-overlap win first, not a proven D8F cache warmup.
- Next runtime move: build an opt-in layer-local scheduler that launches ANE shared expert and GPU/MPSGraph routed D8F from the same hidden-state buffer, then measures GPU merge from CoreML output backing. The merge/fence cost now decides whether the overlap survives integration.
