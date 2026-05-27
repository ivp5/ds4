# Spec-decode tree-search — concrete implementation plan

silv 2026-05-27: "Spec-decode tree-search activation". Existing entry
points wired (#418 task #454 completed, #633 MTP bench validated).

## Existing infrastructure (already shipped)

| component | file:line | purpose |
|-----------|-----------|---------|
| MTP engine state | ds4.c:17102-17113 | `mtp_draft_tokens`, `mtp_ready` |
| MTP per-step draft | ds4.c:15394 `metal_graph_eval_mtp_draft_from_hc` | one draft token per call |
| Spec-decode loop | ds4.c:21786 `ds4_session_eval_speculative_argmax` | linear K-token speculative loop |
| Linear draft array | ds4.c:21829 `drafts[16]` | stored draft tokens |
| Strict verify | ds4.c:21833 `DS4_MTP_STRICT` | force re-verify every draft |
| Margin gate | ds4.c:21834 `mtp_margin` | confidence-based abort |
| MTP cache state | ds4.c:21864 `mtp_n_raw` + accept macros | rollback management |

## What's missing for tree variant

The current decoder:
```
draft[0] = first_token (from base eval)
for i in 1..K:
    draft[i] = greedy_argmax(mtp_logits at i)
verify drafts against target in one batched pass
accept longest matching prefix
```

The tree variant expands the per-step `greedy_argmax` into top-B branching:
```
draft_tree.root.tok = first_token
for i in 1..K:
    for each leaf node L at depth i-1:
        branches = top_B(mtp_logits | path to L)
        for b in branches:
            L.children.append(b)
verify ALL leaves at once
accept longest matching root-to-leaf path
```

## Code changes (3-turn implementation)

### Turn 1: engine + state

**ds4.c (engine struct, ~line 17102)**:
```c
typedef struct ds4_engine {
    /* ... existing fields ... */
    int mtp_draft_tokens;        /* current: linear depth */
    int mtp_draft_tree_width;    /* NEW: branching factor B (1 = linear) */
    bool mtp_ready;
    /* ... */
} ds4_engine;
```

**Init around ds4.c:20722**:
```c
e->mtp_draft_tree_width = opt->mtp_draft_tree_width > 0
    ? opt->mtp_draft_tree_width : 1;
if (e->mtp_draft_tree_width > 4) e->mtp_draft_tree_width = 4;
```

**ds4_cli.c** add `--mtp-tree-width N` (default 1 = linear).

### Turn 2: tree-shape mtp_draft + verifier

**New function in ds4.c** (parallel to `ds4_session_eval_speculative_argmax`):

```c
int ds4_session_eval_speculative_tree(ds4_session *s, int first_token,
        int max_tokens, int eos_token,
        int *accepted, int accepted_cap,
        char *err, size_t errlen) {
    /* Same setup as linear variant for first token */
    if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
    if (max_tokens <= 1 || !e->mtp_ready) {
        accepted[0] = first_token; return 1;
    }

    const int B = e->mtp_draft_tree_width;
    const int D = e->mtp_draft_tokens;  /* depth */
    if (B == 1) {
        /* Fall back to linear (existing code) */
        return ds4_session_eval_speculative_argmax(s, first_token,
            max_tokens, eos_token, accepted, accepted_cap, err, errlen);
    }

    /* Allocate tree:
     *   tree[i] = depth-i layer, has B^i nodes
     *   node[d][i] = { tok, parent_idx, mtp_hc (GPU tensor handle) }
     * Cap nodes per depth at 16 to keep memory bounded. */
    typedef struct {
        int tok;
        int parent_idx;   /* index in prior depth's array */
        ds4_gpu_tensor *hc;
    } tree_node;
    tree_node nodes[D][16];  /* alloca on stack if D, B small */
    int node_count[D];
    node_count[0] = 1;
    nodes[0][0] = (tree_node){first_token, -1, s->graph.mtp_state_hc};

    /* Build tree breadth-first */
    for (int d = 1; d < D; d++) {
        node_count[d] = 0;
        for (int p = 0; p < node_count[d-1]; p++) {
            tree_node *parent = &nodes[d-1][p];
            /* For each branch from this parent */
            for (int b = 0; b < B && node_count[d] < 16; b++) {
                int draft_tok;
                ds4_gpu_tensor *out_hc = /* acquire next slot */;
                if (!metal_graph_eval_mtp_draft_from_hc(&s->graph,
                        &e->model, &e->weights,
                        &e->mtp_model, &e->mtp_weights,
                        parent->hc, out_hc,
                        parent->tok,
                        (uint32_t)(s->checkpoint.len + d - 1),
                        s->mtp_logits, &draft_tok)) {
                    return n_accept;  /* fall back */
                }
                /* Get top-B from mtp_logits — NEW helper needed */
                int top_b_tokens[4];
                logits_top_k(s->mtp_logits, DS4_N_VOCAB, B, top_b_tokens);
                nodes[d][node_count[d]++] = (tree_node){
                    top_b_tokens[b], p, out_hc
                };
            }
        }
    }

    /* Verifier: evaluate target model on ALL leaf candidates in parallel.
     * Returns greedy-argmax per candidate. */
    int verify_top[16 * D];
    if (!metal_graph_eval_tree_verify(&s->graph, &e->model, &e->weights,
            nodes, node_count, D, verify_top)) {
        return n_accept;
    }

    /* Walk tree: pick the longest root-to-leaf path where
     *   nodes[d][i].tok == verify_top[parent's verify position] */
    /* ... acceptance logic ... */
    return n_accept;
}
```

**New helpers needed**:
- `logits_top_k(logits, n_vocab, k, out_tokens)` — top-K extraction
  from logits, ~15 lines using a heap or partial sort.
- `metal_graph_eval_tree_verify(...)` — batch verifier that takes
  multiple (token, parent_hc) pairs and produces per-candidate top
  logit. Most complex new function.

### Turn 3: AIME corpus A/B

Run with K=4 (depth) × B={1, 2, 3, 4} (width) on AIME 2026 P01-P10.
Measure:
- avg accepted-per-iteration (compares to linear baseline of ~2)
- t/s (wall-clock)
- low-entropy vs high-entropy decode positions

Activation gate: only use tree (B > 1) when at low-entropy positions.
Detect via `anchor confidence > 0.8` heuristic OR via `tree_width=B`
env var override.

## Expected acceptance scaling

| config | linear acceptance/iter | tree acceptance/iter | speedup vs linear |
|--------|------------------------|----------------------|-------------------|
| K=4, B=1 (linear) | 2.0 (50% accept) | 2.0 | 1.0× |
| K=4, B=2 | 2.0 | 2.5 (tree caches 50% retry) | 1.25× |
| K=4, B=3 | 2.0 | 2.8 | 1.4× |
| K=4, B=4 | 2.0 | 3.0 | 1.5× |

Compute cost: B× more drafter forwards but B× more candidates verified
in single target pass. Net wall-clock should improve if:
- target eval cost ≪ B× drafter cost (DS4: target = 1.7t/s, drafter
  = 5-10× faster on MTP = ~10-17 t/s effective for drafter)
- low-entropy positions where multiple branches CAN agree with target

## Risk: tree may NOT win at decode

Per the DEEP_OPS_DESIGN analysis: tree only wins at LOW-ENTROPY positions.
At HIGH-ENTROPY positions, branches diverge → low acceptance per
branch → wasted compute.

AIME math decode is mostly LOW-ENTROPY (next token usually heavily
favored). This is the regime where tree shines.

But: gen 4 t/s at K=4 (M1 measured) gives ~4 × 0.25 = 1.0s per
token-batch. Going to B=4 tree doesn't change that floor — the
target verify is what makes the time. Tree only helps if its
parallelism amortizes the target pass over more candidates.

## Recommended next concrete ship

Turn 1 (this session): add `mtp_draft_tree_width` field + CLI plumbing.
No tree-construction code yet. CI smoke that linear path still works
with `mtp_draft_tree_width=1`.

Turn 2: tree construction + `metal_graph_eval_tree_verify` (deferred
to MTL4 path since it's the biggest leverage).

Turn 3: AIME A/B + activation gate.

## What this is NOT

- Not a full Medusa-style tree decoder (which uses multiple parallel
  heads, not MTP draft chain).
- Not a guarantee of 2× t/s — that requires the target verifier to
  amortize well, which depends on low-entropy decode positions.
- Not deployable without VQB2 GGUF fit (per Wire Level-1 A/B blocker).

The honest scope: scaffold the variant + measure. If wins are not
visible at AIME, the dynamic gate (anchor confidence > 0.8) keeps
linear as the default and tree as opt-in for specific low-entropy
generation positions (e.g. inside `\boxed{...}`).
