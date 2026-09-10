# Natural KTD bounded validation

PASS: actual PPU raw-fragment, standalone GEMM, and all9 bounded numerical/performance gates passed in job `dv_199febb2c808` (remote exit0). The fixed source was not changed during validation. Full54/original59 audits were separately authorized and submitted after this pass; these bounded results are not formal promotion or the user's all-shape GDN completion.

Candidate SHA256 `d2a12745625851896e836088c63e1ac8cbf06b19282498f72bab582fa71bb503`; control45de `45de21e78c1796fde2c84106ca9038b471f4fab5a73c2adcaa4951bd951c85e7`; actual dedicated helper SHA `ef59b7e21238f6d9ee520edf5b92cf3023681dbb213523b0227d77e2779c4e32`. Diagnostic snapshot contains the exact control prefix/generic GEMM/coord and exact candidate helper. Raw A-load text is extracted from those actual bodies, preserving the corrected x4 quadrants.

Primitive checks executed before whole correctness:

-4 inputs: finite unique raw BF16 tags0x2000+index,allzero,validC1,andvalidC63. Every16384 uint32 word was exported for old A,new A,and independent host-derived expected fragments. Each case has0 old/new,0 old/host and0 new/host mismatches.
-5 standalone GEMM inputs: tag-derived finite,signed small binary random,allzero,tail1,tail63. All8192 FP32 outputs are bitwise equal old/new and finite.
-The well-conditioned random operands are integer multiples of1/8 in[-1,1]. An independent CPU integer dot product converted to FP32 by/64 is exact for this range. Observed GPU-reference maxabs0,within predeclared1e-6 bound. This tests the mapping and accumulation output; it is not a performance measurement or substitute for the full oracle.

The artifact archive was downloaded and SHA256-verified as `b7ce517eba2ba562f192306dbdf7fb0639aaf2b138d0ad760055f31292dda3d5`. All raw word arrays and expected tags are retained under remote/raw_*.json. The actual loaded diagnostic binary is remote/fragment_loaded.so, SHA256 `e87ef336ba896470553dde77447d4d69c43defb9bc243c2733f63364ac9e76f2`; source,helper and binary hashes are in fragment_proof.json/fragment_source_manifest.json. The diagnostic binary does not stand in for the timed whole-operator binary.

| Public case | Control ms | Candidate ms | GDN ms | State control→candidate ms |
|---|---:|---:|---:|---|
| 3 | 27.552685 | 23.398314 | 6.121260 | 17.859268 → 13.653811 |
| 7 | 36.563471 | 32.387865 | 6.720595 | 16.904440 → 12.794603 |
| tail_32769 | 4.612355 | 4.084965 | 0.870375 | 2.127597 → 1.598038 |
| 128k_hv64 | 145.971893 | 129.161804 | 26.874735 | 67.641141 → 50.919193 |
| small64 | 0.089815 | 0.087815 | 0.027380 | 0.011440 → 0.009225 |
| packed_tail65 | 0.091840 | 0.090025 | 0.026415 | 0.011870 → 0.009635 |
| zero_128k | 27.510740 | 23.315524 | 6.121365 | 17.821960 → 13.714218 |
| decay_32k | 6.885130 | 5.832635 | 1.550195 | 4.492368 → 3.428077 |
| packed_32k_hv16 | 9.137550 | 8.095100 | 1.705030 | 4.272250 → 3.198786 |

Both128K Hv8/Hv16 exceed the required>=5% state and>=2% whole latency reductions. Every other case also improves, satisfying the1% long/packed limit and max(1%,1us) tiny limit. All9 remain slower than pinned GDN. Final newer parent instruction explicitly overrides the frozen research plan's6-case list to9: retain the prepared6,addzero128K Hv8,default-decaying32K Hv8,andpacked24K+8K Hv16 blog75. approved_candidate.json records this scope override and the generator's actual default=decaying key; original research plan is preserved.

All9 cases passed2 independent FP32 oracle seeds1701/2903,4 own-state carries31/47/71/89,finite output/state,.01 relativeL2,input and incoming-carried-state immutability. Max candidate oracle/carry L2=.005607015948959898/.005615506786574918. Candidate and control output+final-state were bitwise identical for every oracle/carry trial,so quantified control difference is0 and the1e-4 review trigger was never reached. Blog75 decaying andzero head error slices are recorded separately. No hidden inputs,history approximation or threshold change was introduced.

Five warmups,256MiB flush,4 symmetric event ABBA rounds and2 device ABBA rounds on one allocated device; every repetition has complete5/5/2 kernel traces. The state gain is measured, but hardware bandwidth saturation,shared-bank behavior,occupancy or compiler-resource causes are not inferred. Exact-product peak utilization remains unavailable. The control45de coarse timeline is historical control evidence and is not reattributed to this new source.

Explicit-input dry-run retained12 files. submit_observe.py invokes the actual AKA tools/sandbox.py,observes accepted jobID,and appends declared agate custom_harness OSS artifact output; repository tools and candidate are unchanged.600s outer budget and120s primitive/compile/case guards applied. Raw result/source/input/harness hashes and artifact contents were checked. No rescue,parameter sweep or failed initial attempt occurred. See sibling ktd_natural_full_audit and ktd_natural_regression59 for actual fullscope results when available; no canonical memory or episode was edited.
