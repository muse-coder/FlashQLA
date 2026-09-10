# PPU FlashQLA long-sequence optimization snapshots

This directory preserves CUDA kernel experiments from the September 8–9, 2026
FlashQLA PPU optimization run. They are standalone experiment modules, not a
replacement for the public `flash_qla` dispatch. Importing the package does not
select these kernels.

## Retained implementations

| Directory | Original local commit | Status |
| --- | --- | --- |
| `cuda_ktd_natural` | `3266b32` | KTD shared layout and transposed matrix loads |
| `cuda_ktd_vector` | `7e79119` | Vectorized KTD publication |
| `cuda_residual_pairs` | `97c695a` | Paired residual publication |
| `cuda_u_col4` | `a7110d3` | Best fully audited retained development version |
| `cuda_gate_autocp/rejected` | `63dd5a2` | Archived candidate; performance gate failed |
| `cuda_residual_fusion/rejected` | `d91042f` | Archived candidate; performance gate failed |
| `cuda_structured_inverse/rejected` | `122e8b9` | Archived candidate; performance gate failed |
| `cuda_structural_rebuild` | Working-tree snapshot after `fbd9665` | Unfinished rebuild; not promoted |

Original commits belong to separate local experiment repositories. This is a
source import, not a claim that those commit objects are ancestors of this
repository. `manifest.json` records the full source revisions and SHA-256 hashes.
A null source revision denotes a file copied from the experiment working tree.

## Validation and performance

The retained `cuda_u_col4/kernel.py` has SHA-256
`9e82b0b77f8e4e831a9df2f15b4f2f1aedf34fc0c6d038f4523af6c3b0c7bc57`.
Its historical PPU correctness audits passed 54 long-sequence cases and 59
original regression cases. It did **not** achieve the goal of beating PPU GDN:
the reported GDN win count was 0/113.

In its bounded paired measurement, 128K Hq2/Hv8 took 18.027 ms, versus 18.860 ms
for its immediate control. These numbers describe the archived experiment and
measurement environment, not newly measured performance of this repository.
See `cuda_u_col4/validation_report.md`, `bounded_report.md`, and `evidence/`.

The structural rebuild includes the current `kernel.py` and `kernel.cu`, plus
selected historical resource and performance reports. Those reports describe
their recorded candidates; they do not validate the current snapshot merely
because they are in the same directory. No successful full-scope acceptance is
claimed for the rebuild.

## Source and harness conventions

- `kernel.py` contains the original PyTorch `Model` wrapper and embedded CUDA
  source. `kernel.cu` is its readable CUDA counterpart. The manifest verifies
  their equality without importing or compiling the module.
- The experiment interface is
  `Model.forward(q, k, v, g, beta, initial_state, cu_seqlens)`; it is distinct
  from the public FlashQLA API. Consult each input generator for layouts and
  supported shapes before invoking a candidate.
- `input.py`, `oracle.py`, and `public_shapes.json` preserve the original
  bounded harness. Full-audit shape lists and result summaries are under
  `cuda_u_col4/evidence/`. The GDN baseline is `cuda_u_col4/baseline.py`.
- The wrappers retain their original CUDA JIT flags and rely on the PPU SDK
  compatibility/translation setup used by AKA. They are not a portable NVIDIA
  performance benchmark or a CPU implementation. No GPU job runs during import
  of this archive into Git.
- Source files are preserved byte-for-byte. Historical reports may refer to
  local run paths and artifacts that are not included here. Full remote logs,
  compiled binaries, credentials, and nested Git repositories are not imported.

The September 10 import performed source-hash, Python syntax, and repository
CPU-test checks. It did not rerun PPU compilation or GPU correctness/performance
tests, and it did not start another optimization run.
