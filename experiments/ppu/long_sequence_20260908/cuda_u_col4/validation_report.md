# Stage4 PASS — private U layout, full113 correctness verified

**Retainable iteration improvement; overall GDN objective remains unmet.** Frozen candidate `9e82b0b77f8e4e831a9df2f15b4f2f1aedf34fc0c6d038f4523af6c3b0c7bc57` passes every declared bounded gate against exact7968 control and the complete54+59 audits. It still wins **0/113 against frozen GDN**. No best index, canonical/active episode, kernel source or Git commit was changed by this validator.

## Actual job results and scope

|Scope|Job|Command exit|Correctness / trace completeness|GDN wins|
|---|---|---:|---|---:|
|Primitive + bounded9 paired|dv_6b1dd8c299a7|0|8 primitive cases,9/9 operator cases, full RAW bits and5/5/2 traces PASS|0/9|
|Complete long54|dv_be6c7c1160d0|1|54/54 candidate+baseline correct, carry/immutable and all5/2 stage arrays PASS|0/54|
|Complete regression59|dv_a1873d7cc90f|1|59/59 candidate+baseline correct, carry/immutable and all5/2 stage arrays PASS|0/59|

All gateway jobs have status succeeded. Full-audit commandexit1 is caused solely by the unchanged requirement to beat GDN on every case; it is not a correctness failure. Job limits remain600s; the bounded compilation/primitive/case guard is120s. No timeout, width fallback, parameter sweep, failed-gate rescue or repeated timing run occurred.

## Correctness and exact mechanism

Actual PPU producer diagnostics execute the exact original and candidate basis-store bodies on512 threads, both tiles and packed offsets0/2/5. Distinct fragment tags, finite random inputs, mixed sign, cancellation, signed zeros/subnormals/ties and validC1/63/64 cover all8192 logical U words per selected chunk. Inverse permutation restores every FP32 word exactly; write counts, sentinels and16B U alignment pass. Consumer diagnostics compare actual read quartets[e,e+1,e+4,e+5], actual scalar FP32 subtraction bits, all global/shared BF16 words for both state parts, sentinels, original4B BF16 alignment and input immutability. Eight cases have zero raw-bit mismatches. The saved NPZ arrays were independently checked again locally after OSS hash validation.

Bounded9 operator cases use two independent FP32 oracle seeds1701/2903 and four own-carry seeds31/47/71/89. Full integer element views now count raw-bit mismatches including signed-zero encodings, alongside maxabs/relativeL2; **all raw bits, signed-zero counts, maxabs and control-relativeL2 are exactly equal**. Independent oracle/carry remain within the frozen.01 contract. Full113 retain the original two-oracle/four-own-carry numerical checks; full maximum oracle relativeL2=.005616472104520575 and own-carry=.0056254837465677535. Strict control raw-bit statistics apply to the bounded9 scope, not an invented113-case control bit audit.

Both actual loaded whole-model native PPU binaries were inspected once without recompiling a replacement. In both state specializations, residual U reads change **4×vmem.ld.b32x2 →2×vmem.ld.b32x4**, retaining four packed BF16 conversions and logical BF16 publication. Basis U stores change **16×vmem.st.b32 →4×vmem.st.b32x4**. Thus the intended widening appears in actual machine code. Native ELF SHA control17053c60389a13b27aa0cd021f796af8dc341afa7cdde3d946644e4053412c57, candidatefbdc2bcb69128e098f6b37ef01d61c6c19d83e16da2b2adaed7b80ace0c34e26. Raw `.hg_info` resource records are preserved; undocumented absolute register/spill fields are not guessed. No bandwidth/occupancy/peak-utilization percentage is inferred from widths or timing.

## Clean paired improvement

Bounded timing preserves5 warmups,256MiB flush,4 event ABBA pairs,2 device ABBA pairs and complete5/5/2 traces. Whole is the caller-stream event median; state/basis are their individual device-event medians. All values below are milliseconds.

|Case|Whole control→candidate|State control→candidate|Basis control→candidate|Whole speedup|
|---|---:|---:|---:|---:|
|3|18.859840 → 18.027184|9.252250 → 8.826679|4.462750 → 4.052913|1.046189×|
|7|28.715240 → 27.451301|9.038204 → 8.641622|9.005173 → 8.171555|1.046043×|
|tail_32769|3.617645 → 3.442715|1.136466 → 1.087741|1.145865 → 1.029211|1.050812×|
|128k_hv64|115.059845 → 109.771839|36.732509 → 34.870661|36.735985 → 33.240307|1.048173×|
|small64|0.086840 → 0.086960|0.008000 → 0.007590|0.024871 → 0.024675|0.998620×|
|packed_tail65|0.088885 → 0.088675|0.008055 → 0.007951|0.024160 → 0.024015|1.002368×|
|zero_128k|19.150434 → 18.231115|9.258949 → 8.832983|4.465141 → 4.055370|1.050426×|
|decay_32k|4.815220 → 4.576820|2.330512 → 2.200641|1.134045 → 1.020574|1.052089×|
|packed_32k_hv16|7.219405 → 6.879305|2.282700 → 2.177311|2.265401 → 2.045555|1.049438×|

Main shape3/7 are128K Hv8/Hv16: whole1.046189×/1.046043×, state1.048214×/1.045892×, basis1.101121×/1.102015×. Both exceed1.01×whole and1.03×state. All nine independent basis gates pass max(1%,1us); other whole gates pass<=1% long regression or max(1%,1us) tiny allowance. Small64 is0.120us slower within its allowance. Complete113 separate-run event medians are all below7968’s previous respective audit medians; this cross-run diagnostic is not substituted for paired acceptance.

## Complete long-audit current-source stage data

New full audits retain every actual per-stage event array; their only harness changes are observability/storage and the declared OSS export (`harness_observability.diff`). The shapes, generator, oracle, baseline, timed calls, samples and comparison rules are unchanged. Columns below are **milliseconds**, drawn entirely from the full54 job. Whole remains independent event timing, not the sum of stages.

|Shape|Prepare|Inverse|Basis|State|Outputs|QLA Whole|GDN prologue|GDN scan|GDN Whole|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|T131072 Hq2/Hv8|0.739808|3.057430|4.046938|8.813048|1.458114|18.306835|0.608677|5.511663|6.123195|
|T131072 Hq4/Hv16|1.512297|6.069806|8.166391|8.637178|2.973175|27.340640|1.201431|5.513547|6.714940|
|T131072 Hq16/Hv64|5.627132|24.432133|33.418428|34.823942|11.212710|109.891724|4.875928|21.981566|26.854521|
|T262144 Hq2/Hv8|1.467593|6.086832|8.087617|17.554130|2.817239|36.075531|1.192937|11.022157|12.211445|
|T262144 Hq4/Hv16|2.955715|12.107005|16.348391|17.194745|5.755548|54.768309|2.381191|11.019275|13.406570|

These measurements still show substantial inverse/basis and other non-state work. They support considering the missing AutoCP algorithm and the whole-stage portfolio rather than expecting residual micro-optimization alone to achieve the all-shape GDN goal. No new algorithm or performance prediction is claimed by this validation.

## Actual-input gate diagnostic

A read-only diagnostic was added to the same bounded job after creating case3 seed1701 inputs and before any timed region, without changing RNG/input order. Actual FP32 g bits are retained; CPUfloat64 analysis uses **hypothetical64-chunk partitions**, threshold strictly<-10. Heads0–5 have mean g about-.0403 and chunk64 sum median about-2.57 to-2.58; all31 preprocessing partitions per decaying head cross in4–5chunks with0fallback. Heads6/7 are exactly zero: all31 partitions need64chunks/fallback. The final32nd partition is ht_mask=True and needs0 preprocessing chunks; fallback is marked not-required/undefined. This is a mathematical diagnostic, not an official main warmup-kernel bit-equivalence test, an AutoCP heuristic decision, or AutoCP execution by this candidate. Current candidate and7968 do not implement AutoCP. Output chunks are not skipped. Details: gate_report.md and remote/gate_diagnostic.json.

## Artifact validation and retention

- Bounded OSS SHA3c96198aa478c3b03e1db006d244523ee511354ea82a12aaac7a111875e70a3d.
- Long54 OSS SHAbcf11f3106feb5bd837d3f8a65f03b0bcd7bf694d37a61be94fe2825a4db5300.
- Regression59 OSS SHAa6660fc15c8da9d202518e0813e6605735b42e9b3fb0ee2b040bb59f90d4fb93.

All downloads match gateway metadata/stdout hashes, and extracted source/harness hashes match exact local submitted bytes. Whole source SHA9e82b0b7 binds all three jobs. Full113 raw per-kernel name/duration arrays remain in each audit’s acceptance.json and remote archive; per-stage medians/raw samples are also in summary.json. Primitive arrays, loaded binaries, generated CUDA/CPP/build/ninja records, native ISA/resources and actual gate bits remain in this candidate’s remote directory. See artifact_verification.json, raw_artifact_audit.json, stage4_summary.json and both full audit reports.

Quality gate **PASS for this iteration**. Overall user objective **not met (0/113 GDN wins)**. The earlier accepted7968 fine receipt remains historical control evidence only; no changed-source accepted_ppu_diagnostics receipt is fabricated or reused. Hardware utilization stays unavailable because exact validated peaks/traffic accounting are not supplied. Parent owns any Stage5 commit or best/canonical promotion; no validator commit was made.
