#include <hggc_runtime.h>

#include "ppu_include.hpp"
#include "cute/tensor.hpp"
#include "cutlass/gemm/config/gemm_operands.hpp"

using namespace cute;

namespace {

constexpr int kChunk = 64;
constexpr int kDim = 128;
constexpr float kLog2E = 1.4426950408889634f;

CUTLASS_DEVICE float stable_exp_difference(float difference) {
  return exp2f(difference * kLog2E);
}

using Element = cutlass::bfloat16_t;
using ArchTag = cutlass::arch::PPU0015;
using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<
    ArchTag, Element, Element, float>::type;
using TiledMma = TiledMMA<
    MMA_Atom<MmaInst>,
    Layout<Shape<_1, _4, _1>>,
    Tile<_16, _64, _16>>;

using TileShape = Shape<Int<kChunk>, Int<kChunk>, Int<kDim>>;
using StrideK = Stride<int, _1>;
using OperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<
    ArchTag, Element, false, Int<kChunk>, Int<kDim>, false>;
using OperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<
    ArchTag, Element, false, Int<kChunk>, Int<kDim>, true>;
using SmemLayoutA = decltype(tile_to_shape(
    typename OperandA::SmemLayoutAtom{},
    Shape<Int<kChunk>, Int<kDim>, _1>{}));
using SmemLayoutB = decltype(tile_to_shape(
    typename OperandB::SmemLayoutAtom{},
    Shape<Int<kChunk>, Int<kDim>, _1>{}));

struct FlashQlaAiuKktInverse {
  struct Arguments {
    const Element* k;
    const float* beta;
    const float* g;
    float* output_fp32;
    Element* output_bf16;
    int groups;
    int tokens;
    int q_heads;
    int value_heads;
    int chunks;
    bool bf16_output;
    bool gated_output;
  };
  using Params = Arguments;

  struct SharedStorage {
    // The two BF16 MMA operands and the two FP32 recurrence matrices have
    // identical aggregate size and disjoint lifetimes. Overlay them so this
    // kernel consumes 32 KiB rather than 64 KiB of TSM.
    cute::array_aligned<Element, 2 * kChunk * kDim> workspace;
  };

  static constexpr int MaxThreadsPerBlock = size(TiledMma{});
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    const int group = blockIdx.x;
    if (group >= params.groups) {
      return;
    }
    const int thread = threadIdx.x;
    int warp = cutlass::canonical_warp_idx_sync();
    auto& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Element* raw_a = storage.workspace.data();
    Element* raw_b = raw_a + kChunk * kDim;
    Tensor sA = make_tensor(
        make_smem_ptr(raw_a), SmemLayoutA{});
    Tensor sB = make_tensor(
        make_smem_ptr(raw_b), SmemLayoutB{});

    typename OperandA::GmemTiledCopy gmem_copy_a;
    typename OperandB::GmemTiledCopy gmem_copy_b;
    using TilerA = typename OperandA::GmemTiledCopy::Tiler_MN;
    using TilerB = typename OperandB::GmemTiledCopy::Tiler_MN;
    gmem_copy_a.desc_.template init<
        Element, false, get<0>(TilerA{}), get<1>(TilerA{})>(
        nullptr, kChunk, kDim, StrideK{params.q_heads * kDim, _1{}});
    gmem_copy_b.desc_.template init<
        Element, false, get<0>(TilerB{}), get<1>(TilerB{})>(
        nullptr, kChunk, kDim, StrideK{params.q_heads * kDim, _1{}});

    const int chunk = group % params.chunks;
    const int head_group = group / params.chunks;
    const int head = head_group % params.value_heads;
    const int batch = head_group / params.value_heads;
    const int q_head = head / (params.value_heads / params.q_heads);
    const int token = chunk * kChunk;
    const Element* group_k = params.k
        + ((batch * params.tokens + token) * params.q_heads + q_head) * kDim;
    StrideK stride{params.q_heads * kDim, _1{}};
    Tensor mK = make_tensor(
        make_gmem_ptr(group_k),
        Shape<Int<kChunk>, Int<kDim>>{}, stride);
    Tensor gA = local_tile(
        make_mix_tensor_like(mK), TileShape{}, make_coord(0, 0, _),
        Step<_1, X, _1>{});
    Tensor gB = local_tile(
        make_mix_tensor_like(mK), TileShape{}, make_coord(0, _, 0),
        Step<X, _1, _1>{});

    auto copy_a_thread = gmem_copy_a.get_slice(thread);
    auto copy_b_thread = gmem_copy_b.get_slice(thread);
    Tensor tAgA = copy_a_thread.partition_S(gA);
    Tensor tAsA = copy_a_thread.partition_D(sA);
    Tensor tBgB = copy_b_thread.partition_S(gB);
    Tensor tBsB = copy_b_thread.partition_D(sB);
    copy_aiu(gmem_copy_a, tAgA(_, _, _, 0), tAsA(_, _, _, 0), warp);
    copy_aiu(gmem_copy_b, tBgB(_, _, _, 0), tBsB(_, _, _, 0), warp);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread);
    Tensor accum = partition_fragment_C(
        tiled_mma, Shape<Int<kChunk>, Int<kChunk>>{});
    Tensor frag_a = thr_mma.partition_fragment_A(sA(_, _, 0));
    Tensor frag_b = thr_mma.partition_fragment_B(sB(_, _, 0));
    auto copy_a = make_tiled_copy_A(
        typename OperandA::SmemCopyAtom{}, tiled_mma);
    auto copy_b = make_tiled_copy_B(
        typename OperandB::SmemCopyAtom{}, tiled_mma);
    auto smem_copy_a_thread = copy_a.get_thread_slice(warp * 32);
    auto smem_copy_b_thread = copy_b.get_thread_slice(warp * 32);
    Tensor smem_a = smem_copy_a_thread.partition_S(make_mix_tensor_like(sA));
    Tensor smem_b = smem_copy_b_thread.partition_S(make_mix_tensor_like(sB));
    Tensor reg_a = smem_copy_a_thread.retile_D(frag_a);
    Tensor reg_b = smem_copy_b_thread.retile_D(frag_b);
    copy(copy_a, smem_a(_, _, _, 0), reg_a);
    copy(copy_b, smem_b(_, _, _, 0), reg_b);
    clear(accum);
    cute::gemm(tiled_mma, accum, frag_a, frag_b, accum);
    __syncthreads();

    float* gram = reinterpret_cast<float*>(storage.workspace.data());
    float* inverse = gram + kChunk * kChunk;
    Tensor coordinates = thr_mma.partition_C(
        make_identity_tensor(Shape<Int<kChunk>, Int<kChunk>>{}));
    for (int index = 0; index < size(accum); ++index) {
      auto coordinate = coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      gram[row * kChunk + column] = accum(index);
    }
    const int vector_base = group * kChunk;
    for (int linear = thread; linear < kChunk * kChunk;
         linear += blockDim.x) {
      const int row = linear / kChunk;
      const int column = linear - row * kChunk;
      if (row > column) {
        gram[linear] *= params.beta[vector_base + row];
      } else {
        gram[linear] = row == column ? 1.0f : 0.0f;
      }
      inverse[linear] = 0.0f;
    }
    __syncthreads();

    // Match the official hierarchical inverse: four independent 16x16
    // diagonal solves, two 16x16 off-diagonal products, then one 32x32
    // off-diagonal product. This reduces the serial synchronization depth
    // from 64 rows to 16 rows plus four product barriers.
    for (int local_row = 0; local_row < 16; ++local_row) {
      if (thread < 64) {
        const int block = thread / 16;
        const int column = thread - block * 16;
        const int base = block * 16;
        const int row = base + local_row;
        if (column == local_row) {
          inverse[row * kChunk + base + column] = 1.0f;
        } else if (column < local_row) {
          float sum = 0.0f;
          for (int inner = column; inner < local_row; ++inner) {
            sum += gram[row * kChunk + base + inner]
                * inverse[(base + inner) * kChunk + base + column];
          }
          inverse[row * kChunk + base + column] = -sum;
        }
      }
      __syncthreads();
    }

    // workspace[0:512] is now dead operand storage. Use it for
    // D_lower^-1 * (-L_offdiag) for block pairs (1,0) and (3,2).
    float* temporary = gram;
    for (int linear = thread; linear < 2 * 16 * 16;
         linear += blockDim.x) {
      const int pair = linear / (16 * 16);
      const int pair_element = linear - pair * 16 * 16;
      const int row = pair_element / 16;
      const int column = pair_element - row * 16;
      const int upper = pair * 32;
      const int lower = upper + 16;
      float sum = 0.0f;
      for (int inner = 0; inner < 16; ++inner) {
        sum -= inverse[(lower + row) * kChunk + lower + inner]
            * gram[(lower + inner) * kChunk + upper + column];
      }
      temporary[linear] = sum;
    }
    __syncthreads();
    for (int linear = thread; linear < 2 * 16 * 16;
         linear += blockDim.x) {
      const int pair = linear / (16 * 16);
      const int pair_element = linear - pair * 16 * 16;
      const int row = pair_element / 16;
      const int column = pair_element - row * 16;
      const int upper = pair * 32;
      const int lower = upper + 16;
      float sum = 0.0f;
      for (int inner = 0; inner < 16; ++inner) {
        sum += temporary[pair * 16 * 16 + row * 16 + inner]
            * inverse[(upper + inner) * kChunk + upper + column];
      }
      inverse[(lower + row) * kChunk + upper + column] = sum;
    }
    __syncthreads();

    // The two 32x32 diagonal inverse blocks are complete. Assemble the
    // bottom-left quadrant as -A1^-1 * L10 * A0^-1.
    for (int linear = thread; linear < 32 * 32;
         linear += blockDim.x) {
      const int row = linear / 32;
      const int column = linear - row * 32;
      float sum = 0.0f;
      for (int inner = 0; inner < 32; ++inner) {
        sum -= inverse[(32 + row) * kChunk + 32 + inner]
            * gram[(32 + inner) * kChunk + column];
      }
      temporary[linear] = sum;
    }
    __syncthreads();
    for (int linear = thread; linear < 32 * 32;
         linear += blockDim.x) {
      const int row = linear / 32;
      const int column = linear - row * 32;
      float sum = 0.0f;
      for (int inner = 0; inner < 32; ++inner) {
        sum += temporary[row * 32 + inner]
            * inverse[inner * kChunk + column];
      }
      inverse[(32 + row) * kChunk + column] = sum;
    }
    __syncthreads();

    for (int linear = thread; linear < kChunk * kChunk;
         linear += blockDim.x) {
      if (params.bf16_output) {
        const int row = linear / kChunk;
        const int column = linear - row * kChunk;
        float value = inverse[linear];
        if (params.gated_output) {
          // A is lower triangular. Do not evaluate the positive gate
          // difference above the diagonal: strong decay can overflow there,
          // and multiplying the mathematical zero by infinity produces NaN.
          value = column <= row
              ? value * stable_exp_difference(
                    params.g[vector_base + row]
                    - params.g[vector_base + column])
              : 0.0f;
        }
        params.output_bf16[group * kChunk * kChunk + linear] = Element(value);
      } else {
        params.output_fp32[group * kChunk * kChunk + linear] = inverse[linear];
      }
    }
  }
};

}  // namespace

extern "C" void launch_aiu_kkt_inverse_bf16_64x128(
    void* k,
    void* beta,
    void* output,
    int groups,
    void* raw_stream,
    int& result) {
  using Kernel = FlashQlaAiuKktInverse;
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  typename Kernel::Arguments arguments{
      reinterpret_cast<const Element*>(k),
      reinterpret_cast<const float*>(beta),
      nullptr,
      reinterpret_cast<float*>(output),
      nullptr,
      groups,
      64,
      1,
      1,
      1,
      false,
      false,
  };
  auto attribute_result = hggcFuncSetAttribute(
      cutlass::device_kernel<Kernel>,
      hggcFuncAttributeMaxDynamicSharedMemorySize,
      Kernel::SharedStorageSize);
  if (attribute_result != hggcSuccess) {
    result = int(attribute_result);
    return;
  }
  cutlass::device_kernel<Kernel><<<
      groups,
      Kernel::MaxThreadsPerBlock,
      Kernel::SharedStorageSize,
      stream>>>(arguments);
  result = int(hggcGetLastError());
}

extern "C" void launch_aiu_kkt_solve_bf16_128_strided(
    void* k,
    void* beta,
    void* output,
    int batch_size,
    int tokens,
    int q_heads,
    int value_heads,
    void* raw_stream,
    int& result) {
  using Kernel = FlashQlaAiuKktInverse;
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  const int chunks = tokens / kChunk;
  const int groups = batch_size * value_heads * chunks;
  typename Kernel::Arguments arguments{
      reinterpret_cast<const Element*>(k),
      reinterpret_cast<const float*>(beta),
      nullptr,
      nullptr,
      reinterpret_cast<Element*>(output),
      groups,
      tokens,
      q_heads,
      value_heads,
      chunks,
      true,
      false,
  };
  auto attribute_result = hggcFuncSetAttribute(
      cutlass::device_kernel<Kernel>,
      hggcFuncAttributeMaxDynamicSharedMemorySize,
      Kernel::SharedStorageSize);
  if (attribute_result != hggcSuccess) {
    result = int(attribute_result);
    return;
  }
  cutlass::device_kernel<Kernel><<<
      groups,
      Kernel::MaxThreadsPerBlock,
      Kernel::SharedStorageSize,
      stream>>>(arguments);
  result = int(hggcGetLastError());
}

extern "C" void launch_aiu_kkt_solve_bf16_128_strided_gated(
    void* k,
    void* beta,
    void* g,
    void* output,
    int batch_size,
    int tokens,
    int q_heads,
    int value_heads,
    void* raw_stream,
    int& result) {
  using Kernel = FlashQlaAiuKktInverse;
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  const int chunks = tokens / kChunk;
  const int groups = batch_size * value_heads * chunks;
  typename Kernel::Arguments arguments{
      reinterpret_cast<const Element*>(k),
      reinterpret_cast<const float*>(beta),
      reinterpret_cast<const float*>(g),
      nullptr,
      reinterpret_cast<Element*>(output),
      groups,
      tokens,
      q_heads,
      value_heads,
      chunks,
      true,
      true,
  };
  auto attribute_result = hggcFuncSetAttribute(
      cutlass::device_kernel<Kernel>,
      hggcFuncAttributeMaxDynamicSharedMemorySize,
      Kernel::SharedStorageSize);
  if (attribute_result != hggcSuccess) {
    result = int(attribute_result);
    return;
  }
  cutlass::device_kernel<Kernel><<<
      groups,
      Kernel::MaxThreadsPerBlock,
      Kernel::SharedStorageSize,
      stream>>>(arguments);
  result = int(hggcGetLastError());
}
