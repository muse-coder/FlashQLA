# Pair residual publication: bounded validation

PASS: isolated publication diagnostic and all9 predeclared bounded gates in job `dv_f3cfd2f9c226` (remote exit0). Source remained frozen `f81f7b2e92f1a273b1564e3a955f3a0769e25d37a36bafbd8b4c8f8ea9d088df`; control `d2a12745625851896e836088c63e1ac8cbf06b19282498f72bab582fa71bb503`. This authorizes separate full54/original59 audits, not formal promotion or GDN completion. All9 still lose to pinned GDN.

Independent static review reproduced exact candidate→control roundtrip by replacing only the residual body, and enumerated512threads×4pairs×2parts:4096 unique pairs covering8192 global values,4096 shared values perCTA,8-byte float2 input and4-byte BF16-pair destinations,with swizzle adjacency preserved. Actual old_residual.cuh/new_residual.cuh bodies and the control helper/coord prefix were snapshotted into residual_probe.cu.

Isolated PPU publication tests:zero,signed,BF16 halfway/ties-to-even plus each adjacent FP32 ULP,cancellation with nonzero prediction,andvalidC1/63/64. Every case exported8192 global and8192 reconstructed shared BF16 values across both parts for old,new and independent host expected. All mismatch counts are0. Host expected first rounds U-pred to FP32 then performs integer round-to-nearest-even BF16 conversion, independently of candidate pack/store. Actual up/vn data_ptr and actual shared-b address checks passed8/4-byte alignment. Full raw values,FP32 input/prediction bits and addresses are retained under remote/raw_*.json; all raw hashes were verified after download.

Archive SHA256 `e3f4c11a4ba77a2b2f84bfedbdc0a4e28b342c40203eaf69a79fb471ddb97947`. Actual diagnostic binary SHA `574005f6f84f326a6834eee174e95a491e52096841087b571591097268a2285c`. Whole control binary SHA `7120279fe3aad58c87de4846d97656d42a4134c534231865adede4cbf4cd0c63`; whole candidate binary SHA `78ec84fc23d500dfc88e54ff7b770cbf0fd7bb61857282ba9d61d335f92087a7`. Actual .so files and owned extension build artifacts are preserved; no separate diagnostic binary is substituted for a timed binary.

| Public case | Control ms | Candidate ms | GDN ms | State control→candidate ms |
|---|---:|---:|---:|---|
| 3 | 23.331290 | 22.870460 | 6.121765 | 13.623108 → 13.021369 |
| 7 | 32.284479 | 31.863550 | 6.716455 | 12.721593 → 12.254812 |
| tail_32769 | 4.090925 | 4.035930 | 0.874230 | 1.607234 → 1.551304 |
| 128k_hv64 | 129.087723 | 126.978050 | 26.875080 | 50.798249 → 48.692608 |
| small64 | 0.089800 | 0.090170 | 0.027305 | 0.009241 → 0.009245 |
| packed_tail65 | 0.090250 | 0.090500 | 0.026170 | 0.009580 → 0.009225 |
| zero_128k | 23.456205 | 22.870630 | 6.124300 | 13.601164 → 13.050022 |
| decay_32k | 5.920605 | 5.761885 | 1.550840 | 3.434452 → 3.277727 |
| packed_32k_hv16 | 8.154970 | 8.019265 | 1.705035 | 3.213401 → 3.083791 |

At128K Hv8,state falls4.42% andwhole1.98%; atHv16,state falls3.67% andwhole1.30%. Both pass>=3% state and>=1% whole requirements. All other long/packed cases improve. Tiny64 andpacked65 medians rise.37us and.25us,which remain below predeclared max(1%,1us); these observed regressions are retained rather than omitted. The iteration passes its declared gate,so the conditional neutral/slower-iteration ISA investigation is not triggered. No automatic additional profile,ISA or rescue sweep was performed.

All9 public shapes are byte-identical to cuda_ktd_natural/public_shapes.json. Each passed2 independent FP32 oracle seeds1701/2903,4 own-state carries31/47/71/89, input and incoming-carried-state immutability,finite results and unchanged .01 relativeL2. Max oracle/carry L2=.005607015948959898/.005615506786574918. Candidate/control output andfinalstate are bitwise identical on every tested oracle/carry trial; control-relativeL2 review trigger1e-4 never fired. Blog75 head subsets retain separate error diagnostics.

Same allocation,5warmups,256MiB flush,4eventABBA+2deviceABBA pairs. Every repetition has complete5/5/2 kernel traces; single caller-stream timing includes the full operator. Stage medians are actual measurements. No bandwidth/occupancy/peak-utilization or compiler instruction-width cause is inferred from source width; no stale d2a fine receipt is reattributed to f81.

Explicit-input dry-run retained13 files. The actual AKA sandbox used agate custom_harness and a declared OSS artifact output; accepted jobID callback was observed without changing repository tools.600s outer/120s operation guards applied. Source/harness/input and artifact hashes were verified. All evidence is in job.json,validation.json,summary.json,residual_source_manifest.json,remote/,source.diff and frozen files. Fullscope audits are in residual_pairs_full_audit andresidual_pairs_regression59. No source,best index,canonical memory or active episode was modified by Stage4.
