# Summary512 three-case diagnostic performance validation

Decision: **SUMMARY_ONLY_THREECASE_DIAGNOSTIC_PASS**. No promotion or full113. Original best9e82 and c346 REJECTED_PERFORMANCE remain unchanged. Job `dv_47ec6209698c` exited0 without timeout; 600/570/120-second guards retained.

The independent audit recomputed all72 clean event samples and36 complete device traces (7/7/2 kernels), all nine unchanged-stage gates, source/input hashes, three actual .so/native contiguous slices and both native ISA outputs. All three input sets exactly reproduce the prior primitive raw tensor hashes at seed1701. Nine light oracle checks pass; maximum reported output/final relative L2 is0.005292115405909038, with finite outputs and unchanged caller inputs. Numerical checks are authenticated execution evidence, not a second local GPU evaluation.

|128K case|B+M control→new ms|Summary speedup|Whole control→new ms|Whole speedup|GDN ms|
|---|---:|---:|---:|---:|---:|
|Hv8 blog75|31.152 → 9.196|3.388×|73.889 → 51.778|1.427×|6.1215|
|Hv16 blog75|32.374 → 9.312|3.477×|115.603 → 92.476|1.250×|6.7133|
|Hv8 zero|32.069 → 17.741|1.808×|74.877 → 60.275|1.242×|6.1204|

All three meet the predeclared1.05× B+M and1.05× whole diagnostic gates. Prepare, inverse and main each remain within max(2%,1µs) regression. B+M is a sum of separately measured stage medians; it is not substituted for whole latency. There are **0/3 GDN wins**.

Both original main dtype text sections remain exact; both summary text sections match the previously validated primitive binary. The timed summary512 native ELF differs from the primitive diagnostic ELF because the latter also contains helper diagnostic entry points; required production text matches were independently verified.

- kernel timed native SHA256: `cde1717f867bf3993992905885d245a9cfe5d5408fd4631edc910b375537a7fc`.
- control timed native SHA256: `d29e6cb549ab15b28c13f3d26073ef317ac1bc96e0a1b6bf1628c6aadc3f85ed`.
- baseline timed native SHA256: `b8028a0d8c75e898d7e4872e426cc8648224242a1c0eea331281abd3b860144f`.

The evidence supports combining summary512 with a separately validated main512 candidate and performing full acceptance. It does not satisfy the original eleven-shape/full113 objective. No utilization, occupancy, bandwidth or hardware-counter explanation is inferred. Raw samples, every stage event and binary/source details remain in `profiles/summary_cta512_performance/attempt-1/remote/summary_performance_results`; full recomputation and hashes are in `summary_cta512_performance_validation.json`.
