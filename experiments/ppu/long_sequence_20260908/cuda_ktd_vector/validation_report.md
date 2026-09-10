# Stage4 validation — PASS, overall GDN objective remains unmet

Frozen candidate `7968fe26cbb269c930e070943d5b907fb1df1446631d55581cac17b01b0eb9e8`, exact f81 control `f81f7b2e92f1a273b1564e3a955f3a0769e25d37a36bafbd8b4c8f8ea9d088df`, plan SHA `da7d0f2ee2737d64a6f66ac38f92d00f1e0706774294d562a31febe1a511228a`.

The natural KTD uint4 publication change passes its entire predeclared iteration gate. Exact source roundtrip confirms only the targeted loop and new JIT identity changed. Primitive proof passed10 cases and491520rawBF16word comparisons with no mismatches, covering both part CTAs, multiple packed offsets, signedzero, rounding-sensitive/tiny/subnormal/underflow inputs, and1/63/64 tails. Whole output and final state are strictly bitwise equal to f81 in all9 paired cases and all2oracle+4carry trials. These cases also pass input/incoming-state immutability and complete5/5/2 traces.

| Job | Scope | Command exit | Result |
|---|---|---:|---|
| dv_4604fd4728f9 | primitive +9same-allocationpairedcases | 0 | All declared correctness and performance gates PASS |
| dv_14a9276109af | unchanged full54long audit | 1 | 54/54numerical/carry/immutability/trace PASS;0/54GDNwins |
| dv_309d6f444229 | unchanged original59 audit | 1 | 59/59numerical/carry/immutability/trace PASS;0/59GDNwins |

The two full-audit exit1 results reflect the unchanged every-shape-faster-than-GDN requirement. They are not compile/accuracy/trace failures. Overall113/113 candidate and frozenGDN numerical checks pass; all traces are complete. Maximum fullscope oracle relativeL2=.005616472104520575; maximum carry relativeL2=.0056254837465677535, both below.01.

Paired9-case evidence:128K Hq2/Hv8 whole22.774956→18.987690ms (1.199459x), state13.060762→9.342008ms; Hq4/Hv16 whole31.849259→28.641390ms (1.112001x), state12.272370→9.059033ms. Both meet>=1%whole and>=3%state reductions. Every other long/tiny case satisfies its no-regression gate; all9 whole medians improve. GDN still wins all9.

Full-audit absolute measurements:128K Hq2/Hv8=19.224595ms versusGDN6.123665ms;128K Hq4/Hv16=28.567980ms versus6.717810ms;128K Hq16/Hv64=115.345329ms versus26.881475ms;256K Hq2/Hv8=37.636786ms versus12.213300ms. The256K measurement comes from full54, not extrapolation of bounded cases. All113 medians are lower than their previousf81 audit medians across separate allocations, but that observation is only diagnostic; it does not replace paired or formal performance verification.

No failed gate required conditionalISA escalation. The actual loaded probe/model binaries, exact source/build files and every rawword/timing are preserved; no native-vectorwidth, register/spill, occupancy, bandwidth or peak-utilization claim is made. Historicalf81 fine receipt0a46... selected the edit and is explicitly not a candidate accepteddiagnostic after this source change. No new candidate receipt is asserted.

ArchiveSHA `2de92f3c5fa8f0a2e013261ec3df831cc681d86be8cdd6f6660ba7b7dedad743` matches both actual OSS metadata and remote emitted hash; downloaded candidate SHA matches frozen7968fe26. No further GPU jobs, threshold changes or sweep occurred. The chosen shared-state publication arithmetic and scalarBF16 conversions remain unchanged; no loader-cache change was combined.

Artifacts: bounded_report.md and comparison.json cover the primitive/9-case gate; validation.json has raw accuracy and timing rows; remote/raw_*.json contains the complete publication words and actual addresses. ../ktd_vector_full_audit/validation_report.md and ../ktd_vector_regression59/validation_report.md cover all113 shapes, with original acceptance.json/job.json and frozen audit_scope.json. stage4_summary.json and manifest.json identify the final gate; memory/v1.json is updated through the memory manager. No best index, canonical memory, active episode source or git commit was modified. Formal independent policy and authoritative canonicalABBA remain required before promotion. The user's overall every-shape GDN objective is not achieved.
