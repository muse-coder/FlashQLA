# Summary512 partial primitive validation

Decision: **SUMMARY_ONLY_PARTIAL_PRIMITIVES_PASS** for Python cf09f63f / CUDA12af7752, compared with c346e430/c04fc96c. Job `dv_dd817d692fa1` exited0; no timeout, performance, full113 or promotion. The original c346 performance rejection remains unchanged.

Independent archive audit recomputed all4 natural clear-false helper fixtures (zero, nonzero and two part identities): original4-warp/new16-warp outputs, production/capture, every operand fragment and relative shared address are raw-equal. Actual addresses are16B aligned/in-bounds. Independent FP64 products have zero error. Three short summary fixtures retain raw B/M equality, all18 unchanged-main recurrence comparisons and inactive sentinels.

All3 long cases have identical B/M, prefix boundary, main output/final and both production forwards. One common raw O backing is hash-bound to both outputs; complete B/M/boundary/final arrays are saved. Long inputs are recorded by frozen generator/shape/seed1701, pre-generation RNG states and every dtype/shape/rawSHA; any later regeneration must match every hash. Runtime oracle and immutable checks passed; the audit did not rerun GPU computations.

|128K case|Output relative L2|Final relative L2|Actual nonterminal metadata|
|---|---:|---:|---|
|Hv8 blog75|0.0051624100|0.0039843475|heads0–5 each7 TAIL; heads6–7 each7 FULL|
|Hv16 blog75|0.0051650835|0.0040250350|heads0–11 each7 TAIL; heads12–15 each7 FULL|
|Hv8 zero|0.0052921154|0.0041489411|all8 heads each7 FULL|

All use P8/L256. TAIL warmups are4–5 chunks; FULL histories are256 chunks. Terminal partition is omitted from summaries. Per-head arrays are retained in JSON. Both actual loaded .so→native contiguous slices were independently checked. Both main dtype text sections match the original timed native exactly; the new summary helper does not change main text.

- Original native: `d29e6cb549ab15b28c13f3d26073ef317ac1bc96e0a1b6bf1628c6aadc3f85ed`.
- Summary512 diagnostic native: `65f7d27c86dedb4b39b5ee0b702952c0443a8ad8ff7edecce15c0d50af9fc129`.
- Archive: `e333a1117b02e7a5d8aff4e0e6414b13f91cf1a98b42b61bc7e93f8d5810c241`,952,722,394 bytes.

Next authorized scope can reuse the frozen same-input3case clean-event ABBA protocol to diagnose summary B+M and whole effect versus c346, with actual7/7/2 stage identities and unchanged-stage protection. This is not a replacement for whole11/carry/supplementals/streamGraph/full113 acceptance before adoption. All source and actual-array bindings are in `summary_cta512_primitives_validation.json` and `profiles/summary_cta512_primitives/attempt-1/remote/summary_results`.
