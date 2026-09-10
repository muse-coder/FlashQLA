# Bounded Stage4 PASS: private U column layout

Frozen candidate9e82b0b77f8e4e831a9df2f15b4f2f1aedf34fc0c6d038f4523af6c3b0c7bc57 versus7968fe26 control, plan95c7119793e7bb9d12e66740faceaf006ced43a732879487d9b02a3253a08354. Actual job **dv_6b1dd8c299a7**, infrastructure succeeded, commandexit0. One bounded AKA→agate custom_harness after typed-route dry-run established the explicit-input limitation. All17 dependencies were declared; job600s and compile/case120s guards remained in force. Frozen kernel/control were not modified.

Eight actual PPU primitive cases pass: distinct fragment tags, finite random, mixed sign, cancellation, signed-zero/subnormal/tie inputs, and tail1/63/64. Exact old/new basis store bodies operate on all512×2×8 fragments for packed0/2/5. All8192 logical FP32 words per selected chunk are restored bit-exactly, unique write counts are1, unused chunks preserve sentinels, and U is16B aligned. Actual old/new read quartets recover[e,e+1,e+4,e+5] and actual FP32 subtraction bits match. Both512-thread state parts publish4096 shared BF16 words each/8192 global BF16 per chunk; strict raw bits, integrated-vs-recorded-subtraction conversion, shared/global mapping, sentinels,4B destination alignment and input immutability all pass. No host-only layout proof substitutes for this actual PPU test; no host FTZ assumption or arithmetic/rounding change was introduced.

Nine public cases pass the fixed.01 independent oracle contract for two seeds1701/2903, four own-state carries31/47/71/89, input/incoming-state immutability and complete5/5/2 traces. All output/state comparisons now use full int16/int32 raw element views, with a separate signed-zero mismatch counter, alongside numeric maxabs/relativeL2. **All raw bit mismatches, signed-zero mismatches, maxabs and control-relativeL2 are zero.** Max oracle relativeL2=.005607015948959898; max own-carry=.005615506786574918.

## Same-allocation timings

Five warmups,256MiB flush, four event ABBA pairs and two device ABBA pairs. Whole is the independent current-stream event median, never a sum of kernel times. Basis/state are medians from complete device traces. All values below are milliseconds.

|Case|Whole control→candidate|State control→candidate|Basis control→candidate|Whole speedup|All gates|
|---|---:|---:|---:|---:|---|
|3|18.859840 → 18.027184|9.252250 → 8.826679|4.462750 → 4.052913|1.046189×|PASS|
|7|28.715240 → 27.451301|9.038204 → 8.641622|9.005173 → 8.171555|1.046043×|PASS|
|tail_32769|3.617645 → 3.442715|1.136466 → 1.087741|1.145865 → 1.029211|1.050812×|PASS|
|128k_hv64|115.059845 → 109.771839|36.732509 → 34.870661|36.735985 → 33.240307|1.048173×|PASS|
|small64|0.086840 → 0.086960|0.008000 → 0.007590|0.024871 → 0.024675|0.998620×|PASS|
|packed_tail65|0.088885 → 0.088675|0.008055 → 0.007951|0.024160 → 0.024015|1.002368×|PASS|
|zero_128k|19.150434 → 18.231115|9.258949 → 8.832983|4.465141 → 4.055370|1.050426×|PASS|
|decay_32k|4.815220 → 4.576820|2.330512 → 2.200641|1.134045 → 1.020574|1.052089×|PASS|
|packed_32k_hv16|7.219405 → 6.879305|2.282700 → 2.177311|2.265401 → 2.045555|1.049438×|PASS|

Both128K Hv8/Hv16 exceed1.03× state and1.01× whole. Every case meets the independent basis regression allowance max(1%,1us); other long whole regressions stay<=1%, tiny whole withinmax(1%,1us). Small64 is0.120us slower in whole, within its predeclared allowance. GDN still wins9/9; this is iteration progress, not the user’s all-shape objective.

## Actual native mechanism

Both **actually loaded** host extensions and generated CUDA/CPP/build.ninja/object/.ninja_log are retained. One hgobjdump call per extracted native PPU ELF completedexit0; no alternative source recompilation supplied the evidence. Both FP32 and BF16 state variants show residual U **four vmem.ld.b32x2 → two vmem.ld.b32x4**, retaining four paired BF16 conversions and the logical BF16 stores. The basis function’s final U stores change **sixteen vmem.st.b32 → four vmem.st.b32x4**. Thus the intended producer/consumer widening is actually present. Complete native resource sections are retained as bytes and hashes, but undocumented fields are not decoded into invented register/spill figures. Device ELF hashes: control17053c60389a13b27aa0cd021f796af8dc341afa7cdde3d946644e4053412c57; candidatefbdc2bcb69128e098f6b37ef01d61c6c19d83e16da2b2adaed7b80ace0c34e26. Absolute hardware utilization/traffic/peak ratios remain unavailable; widths and timing do not prove bandwidth saturation.

OSS archive SHA **3c96198aa478c3b03e1db006d244523ee511354ea82a12aaac7a111875e70a3d** was downloaded and verified against metadata/stdout. Extracted candidate/control/probe/driver/gate-diagnostic/native-inspector sources match exact local bytes. Raw producer, quartet, subtraction, BF16 and actual-gate arrays remain in remote/*.npz; validation.json preserves complete samples and trace arrays.

The actual-input gate diagnostic is documented separately in gate_report.md. It is outside timing and makes no AutoCP execution claim. Full54+59 audits were authorized only after all bounded gates and source/artifact checks passed: dv_be6c7c1160d0 and dv_a1873d7cc90f. They retain per-stage event arrays and export the complete result through OSS; timed calls/thresholds/shapes/baselines/oracles remain unchanged. Final whole-scope Stage4 status awaits these two audits. No best/canonical/active-episode edits or commits were performed by this validator.
