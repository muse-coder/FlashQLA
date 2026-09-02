#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

arch="${PPU_ARCH:-ppu_15}"
output="flash_qla/ops/gated_delta_rule/chunk/ppu/libflash_qla_ppu.so"
source_root="flash_qla/ops/gated_delta_rule/chunk/ppu/csrc"
sources=("$source_root/ppu_warmup.cu")
include_flags=()
deep_gemm_root="${FLASHQLA_DEEP_GEMM_ROOT:-${HOME}/DeepGEMM-for-sail}"
if [[ -f "$deep_gemm_root/third-party/actlize_v1.0.0/include/cute/tensor.hpp" ]]; then
  sources+=(
    "$source_root/ppu_kkt.cu"
    "$source_root/ppu_flash_qla_fwd.cu"
  )
  include_flags+=(
    -I"$deep_gemm_root/third-party/actlize_v1.0.0/include"
    -I"$deep_gemm_root/third-party/actlize_v1.0.0/tools"
    -I"$deep_gemm_root/deep_gemm/include"
  )
fi
hgcc "${sources[@]}" -o "$output" \
  "${include_flags[@]}" -shared -x hg -O3 -std=c++17 \
  -ftemplate-depth=8192 -Xcompiler -fPIC -arch="$arch"
