# Compiled resource validation

Decision: RESOURCE_FACTS_ATTRIBUTABLE_TO_TIMED_BINARIES. The candidate remains REJECTED_PERFORMANCE.

Job `dv_4a99933b1e0e` completed with exit 0 in 49.29 seconds. This host-only diagnostic requested zero GPU kernel launches. It appended a resource query and compiler verbosity (`-Xptxas -v`); it did not measure performance. The 600/570/120-second job/worker/phase guards were retained.

Independent parsing matched all 10 candidate and 7 control compiler entries to their native kernel symbols. All 17 report zero stack-frame bytes, zero spilled SRegs, and zero spilled VRegs. All 11 queried runtime entries report `localSizeBytes=0`. Thus spilling is not supported as the cause of the observed slowdown.

| Target | Runtime numRegs | Compiler SRegs / VRegs | Static shared bytes | Production dynamic shared bytes |
|---|---:|---:|---:|---:|
| kernel/fused_svo_float | 144 | 61 / 138 | 0 | 82688 |
| kernel/fused_svo_bf16 | 144 | 61 / 138 | 0 | 82688 |
| kernel/summary_b | 168 | 61 / 165 | 0 | 58112 |
| kernel/summary_m | 168 | 61 / 163 | 0 | 58112 |
| kernel/prepare_float | 40 | 59 / 40 | 0 | 0 |
| kernel/inverse | 64 | 61 / 64 | 65536 | 0 |
| control/state_scan_float | 160 | 62 / 160 | 32768 | 0 |
| control/basis | 80 | 61 / 74 | 32768 | 0 |
| control/outputs | 56 | 59 / 51 | 32768 | 0 |
| control/prepare_float | 40 | 59 / 40 | 0 | 0 |
| control/inverse | 64 | 61 / 64 | 65536 | 0 |

Runtime register counts and compiler VRegs are distinct reported fields; their difference does not establish allocation granularity. Zero static shared storage does not mean zero shared storage: fused SVO requests 82,688 dynamic bytes and each summary requests 58,112 bytes. Occupancy, dynamic register pressure, bandwidth, utilization and stall-time fractions remain UNKNOWN.

Both resource native ELFs were independently compared byte-for-byte with the exact timed native ELFs. Actual loaded `.so` to embedded ELF contiguous slices were also verified. Entire native SHA256:

- kernel: `d29e6cb549ab15b28c13f3d26073ef317ac1bc96e0a1b6bf1628c6aadc3f85ed`
- control: `fbdc2bcb69128e098f6b37ef01d61c6c19d83e16da2b2adaed7b80ace0c34e26`

The collector verified all 12 input hashes and archive `ab3bfd9ac84a86de36df4a9edbff49ec20999ef53dda0dc43fd6d04039d212cd`. Full exact-symbol compiler fields, binary/log/source hashes and runtime queries are in [compiled_resources_validation.json](compiled_resources_validation.json).
Raw evidence is under `profiles/compiled_resources/attempt-1/remote/resource_results`; compiler logs are `probe/builds/{kernel,control}/compile.log`.

No production source, performance decision, or full113 eligibility changed.
