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

template <int M, int N, int K>
struct AiuTile {
  static constexpr int Rows = M;
  static constexpr int Columns = N;
  static constexpr int Inner = K;
  using OperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<
      ArchTag, Element, false, Int<M>, Int<K>, false>;
  using OperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<
      ArchTag, Element, false, Int<N>, Int<K>, true>;
  using SmemLayoutA = decltype(tile_to_shape(
      typename OperandA::SmemLayoutAtom{}, Shape<Int<M>, Int<K>, _1>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      typename OperandB::SmemLayoutAtom{}, Shape<Int<N>, Int<K>, _1>{}));
};

template <class Config>
CUTLASS_DEVICE void copy_global_a_to_tsm(
    const Element* source,
    int row_stride,
    Element* raw_a,
    int thread,
    int& warp) {
  constexpr int M = Config::Rows;
  constexpr int N = Config::Columns;
  constexpr int K = Config::Inner;
  using StrideA = Stride<int, _1>;
  using TileShape = Shape<Int<M>, Int<N>, Int<K>>;
  typename Config::OperandA::GmemTiledCopy gmem_copy;
  using Tiler = typename Config::OperandA::GmemTiledCopy::Tiler_MN;
  StrideA stride{row_stride, _1{}};
  gmem_copy.desc_.template init<
      Element, false, get<0>(Tiler{}), get<1>(Tiler{})>(
      nullptr, M, K, stride);
  Tensor source_tensor = make_tensor(
      make_gmem_ptr(source), Shape<Int<M>, Int<K>>{}, stride);
  Tensor source_tile = local_tile(
      make_mix_tensor_like(source_tensor), TileShape{}, make_coord(0, 0, _),
      Step<_1, X, _1>{});
  Tensor destination = make_tensor(
      make_smem_ptr(raw_a), typename Config::SmemLayoutA{});
  auto copy_thread = gmem_copy.get_slice(thread);
  Tensor source_partition = copy_thread.partition_S(source_tile);
  Tensor destination_partition = copy_thread.partition_D(destination);
  copy_aiu(
      gmem_copy,
      source_partition(_, _, _, 0),
      destination_partition(_, _, _, 0),
      warp);
  cp_async_fence();
  cp_async_wait<0>();
}

template <class Config>
CUTLASS_DEVICE void copy_global_b_to_tsm(
    const Element* source,
    int row_stride,
    Element* raw_b,
    int thread,
    int& warp) {
  constexpr int M = Config::Rows;
  constexpr int N = Config::Columns;
  constexpr int K = Config::Inner;
  using StrideB = Stride<int, _1>;
  using TileShape = Shape<Int<M>, Int<N>, Int<K>>;
  typename Config::OperandB::GmemTiledCopy gmem_copy;
  using Tiler = typename Config::OperandB::GmemTiledCopy::Tiler_MN;
  StrideB stride{row_stride, _1{}};
  gmem_copy.desc_.template init<
      Element, false, get<0>(Tiler{}), get<1>(Tiler{})>(
      nullptr, N, K, stride);
  Tensor source_tensor = make_tensor(
      make_gmem_ptr(source), Shape<Int<N>, Int<K>>{}, stride);
  Tensor source_tile = local_tile(
      make_mix_tensor_like(source_tensor), TileShape{}, make_coord(0, _, 0),
      Step<X, _1, _1>{});
  Tensor destination = make_tensor(
      make_smem_ptr(raw_b), typename Config::SmemLayoutB{});
  auto copy_thread = gmem_copy.get_slice(thread);
  Tensor source_partition = copy_thread.partition_S(source_tile);
  Tensor destination_partition = copy_thread.partition_D(destination);
  copy_aiu(
      gmem_copy,
      source_partition(_, _, _, 0),
      destination_partition(_, _, _, 0),
      warp);
  cp_async_fence();
  cp_async_wait<0>();
}

using Tile64128128 = AiuTile<64, 128, 128>;
using Tile6464128 = AiuTile<64, 64, 128>;
using Tile6412864 = AiuTile<64, 128, 64>;
using Tile646464 = AiuTile<64, 64, 64>;
using Tile12812864 = AiuTile<128, 128, 64>;
using Tile1286464 = AiuTile<128, 64, 64>;

CUTLASS_DEVICE int bf16_tsm_offset(int row, int column, int rows) {
  const int row_group = row >> 3;
  const int lane_row = row & 7;
  const int column_slab = column >> 6;
  const int slab_column = column & 63;
  const int vector = slab_column >> 3;
  const int lane_column = slab_column & 7;
  return column_slab * rows * 64 + row_group * 512
      + lane_row * 64 + (vector ^ lane_row) * 8 + lane_column;
}

template <class Config, class Accumulator>
CUTLASS_DEVICE void mma_from_tsm(
    Element* raw_a,
    Element* raw_b,
    int thread,
    int warp,
    Accumulator& accum,
    bool clear_accumulator) {
  Tensor sA = make_tensor(
      make_smem_ptr(raw_a), typename Config::SmemLayoutA{});
  Tensor sB = make_tensor(
      make_smem_ptr(raw_b), typename Config::SmemLayoutB{});
  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(thread);
  Tensor frag_a = thr_mma.partition_fragment_A(sA(_, _, 0));
  Tensor frag_b = thr_mma.partition_fragment_B(sB(_, _, 0));

  auto copy_a = make_tiled_copy_A(
      typename Config::OperandA::SmemCopyAtom{}, tiled_mma);
  auto copy_b = make_tiled_copy_B(
      typename Config::OperandB::SmemCopyAtom{}, tiled_mma);
  auto copy_a_thread = copy_a.get_thread_slice(warp * 32);
  auto copy_b_thread = copy_b.get_thread_slice(warp * 32);
  Tensor smem_a = copy_a_thread.partition_S(make_mix_tensor_like(sA));
  Tensor smem_b = copy_b_thread.partition_S(make_mix_tensor_like(sB));
  Tensor reg_a = copy_a_thread.retile_D(frag_a);
  Tensor reg_b = copy_b_thread.retile_D(frag_b);
  copy(copy_a, smem_a(_, _, _, 0), reg_a);
  copy(copy_b, smem_b(_, _, _, 0), reg_b);
  if (clear_accumulator) {
    clear(accum);
  }
  cute::gemm(tiled_mma, accum, frag_a, frag_b, accum);
}

struct FlashQlaFusedForward {
  struct Arguments {
    const Element* q;
    const Element* k;
    const Element* v;
    const Element* a;
    const float* g;
    const float* beta;
    const float* initial_state;
    Element* output;
    float* final_state;
    int batch_size;
    int tokens;
    int q_heads;
    int value_heads;
    float scale;
    bool use_initial_state;
  };
  using Params = Arguments;

  struct SharedStorage {
    cute::array_aligned<Element, 128 * 64> a_tsm;
    cute::array_aligned<Element, 128 * 64> b_tsm;
    cute::array_aligned<Element, 64 * 64> pg_tsm;
    cute::array_aligned<Element, 64 * 64> value_bf16;
    cute::array_aligned<float, 64> gamma;
    cute::array_aligned<float, 64> gate;
    cute::array_aligned<float, 64> decay_to_last;
  };

  static constexpr int MaxThreadsPerBlock = size(TiledMma{});
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    const int tile_group = blockIdx.x;
    const int tile_groups = params.batch_size * params.value_heads * 2;
    if (tile_group >= tile_groups) {
      return;
    }
    const int group = tile_group / 2;
    const int value_tile = tile_group - group * 2;
    const int value_start = value_tile * 64;
    const int thread = threadIdx.x;
    int warp = cutlass::canonical_warp_idx_sync();
    const int batch = group / params.value_heads;
    const int head = group - batch * params.value_heads;
    const int q_head = head / (params.value_heads / params.q_heads);
    const int chunks = params.tokens / kChunk;
    auto& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Element* a_tsm = storage.a_tsm.data();
    Element* b_tsm = storage.b_tsm.data();
    Element* pg_tsm = storage.pg_tsm.data();

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread);
    Tensor state = partition_fragment_C(
        tiled_mma, Shape<Int<128>, Int<64>>{});
    Tensor state_coordinates = thr_mma.partition_C(
        make_identity_tensor(Shape<Int<128>, Int<64>>{}));
    Tensor work = partition_fragment_C(
        tiled_mma, Shape<Int<64>, Int<64>>{});
    Tensor work_coordinates = thr_mma.partition_C(
        make_identity_tensor(Shape<Int<64>, Int<64>>{}));
    const int state_base = group * kDim * kDim;
    for (int index = 0; index < size(state); ++index) {
      auto coordinate = state_coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      state(index) = params.use_initial_state
          ? params.initial_state[
                state_base + row * kDim + value_start + column]
          : 0.0f;
    }

    for (int chunk = 0; chunk < chunks; ++chunk) {
      const int token_base = chunk * kChunk;
      const int gate_base = (group * chunks + chunk) * kChunk;
      if (thread < kChunk) {
        const float gate = params.g[gate_base + thread];
        storage.gamma[thread] = expf(gate);
        storage.gate[thread] = gate;
      }
      __syncthreads();
      if (thread < kChunk) {
        storage.decay_to_last[thread] = stable_exp_difference(
            storage.gate[kChunk - 1] - storage.gate[thread]);
      }

      // U = K @ S.
      const int k_source = (((batch * params.tokens + token_base)
          * params.q_heads + q_head) * kDim);
      copy_global_a_to_tsm<Tile6464128>(
          params.k + k_source, params.q_heads * kDim,
          a_tsm, thread, warp);
      for (int index = 0; index < size(state); ++index) {
        auto coordinate = state_coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        b_tsm[bf16_tsm_offset(column, row, kChunk)] = Element(state(index));
      }
      __syncthreads();
      mma_from_tsm<Tile6464128>(
          a_tsm, b_tsm, thread, warp, work, true);
      __syncthreads();

      // W = V - exp(g) * U, and Ag = G * A * diag(beta).
      for (int index = 0; index < size(work); ++index) {
        auto coordinate = work_coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        const int source = (((batch * params.tokens + token_base + row)
            * params.value_heads + head) * kDim) + value_start + column;
        const float value = float(params.v[source])
            - storage.gamma[row] * work(index);
        b_tsm[bf16_tsm_offset(column, row, kChunk)] = Element(value);
      }
      for (int linear = thread; linear < kChunk * kChunk;
           linear += blockDim.x) {
        const int row = linear / kChunk;
        const int column = linear - row * kChunk;
        float value = 0.0f;
          if (column <= row) {
            const int a_index = gate_base * kChunk + linear;
            value = float(params.a[a_index])
              * params.beta[gate_base + column];
        }
        a_tsm[bf16_tsm_offset(row, column, kChunk)] = Element(value);
      }
      __syncthreads();
      mma_from_tsm<Tile646464>(
          a_tsm, b_tsm, thread, warp, work, true);
      for (int index = 0; index < size(work); ++index) {
        auto coordinate = work_coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        storage.value_bf16[bf16_tsm_offset(column, row, kChunk)] =
            Element(work(index));
      }
      __syncthreads();

      // Compute P first and publish Pg in BF16 TSM. Q remains resident in
      // a_tsm until the following Q@S, while the P fragment dies here.
      const int q_source = (((batch * params.tokens + token_base)
          * params.q_heads + q_head) * kDim);
      copy_global_a_to_tsm<Tile6464128>(
          params.q + q_source, params.q_heads * kDim,
          a_tsm, thread, warp);
      copy_global_b_to_tsm<Tile6464128>(
          params.k + k_source, params.q_heads * kDim,
          b_tsm, thread, warp);
      __syncthreads();
      {
      Tensor p = partition_fragment_C(
          tiled_mma, Shape<Int<64>, Int<64>>{});
      Tensor p_coordinates = thr_mma.partition_C(
          make_identity_tensor(Shape<Int<64>, Int<64>>{}));
      mma_from_tsm<Tile6464128>(
          a_tsm, b_tsm, thread, warp, p, true);
      for (int index = 0; index < size(p); ++index) {
        auto coordinate = p_coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        float value = 0.0f;
        if (column <= row) {
          value = p(index) * params.scale
              * stable_exp_difference(
                  storage.gate[row] - storage.gate[column]);
        }
        pg_tsm[bf16_tsm_offset(row, column, kChunk)] = Element(value);
      }
      }
      __syncthreads();

      // O = scale * exp(g) * Q @ S. Keep O in its accumulator until Pg@Vd.
      for (int index = 0; index < size(state); ++index) {
        auto coordinate = state_coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        b_tsm[bf16_tsm_offset(column, row, kChunk)] = Element(state(index));
      }
      __syncthreads();
      mma_from_tsm<Tile6464128>(
          a_tsm, b_tsm, thread, warp, work, true);
      for (int index = 0; index < size(work); ++index) {
        const int row = get<0>(work_coordinates(index));
        work(index) *= params.scale * storage.gamma[row];
      }
      __syncthreads();

      __syncthreads();
      mma_from_tsm<Tile646464>(
          pg_tsm, storage.value_bf16.data(), thread, warp, work, false);
      for (int index = 0; index < size(work); ++index) {
        auto coordinate = work_coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        const int output = (((batch * params.tokens + token_base + row)
            * params.value_heads + head) * kDim) + value_start + column;
        params.output[output] = Element(work(index));
      }
      __syncthreads();

      // S = exp(g_last) S + K^T @ (exp(g_last-g) Vd).
      const float gamma_last = storage.gamma[kChunk - 1];
      for (int index = 0; index < size(state); ++index) {
        state(index) *= gamma_last;
      }
      for (int linear = thread; linear < kDim * kChunk;
           linear += blockDim.x) {
        const int row = linear / kChunk;
        const int column = linear - row * kChunk;
        const int source = (((batch * params.tokens + token_base + column)
            * params.q_heads + q_head) * kDim) + row;
        a_tsm[bf16_tsm_offset(row, column, kDim)] = params.k[source];
      }
      for (int linear = thread; linear < kChunk * kChunk;
           linear += blockDim.x) {
        const int token = linear / kChunk;
        const int value_column = linear - token * kChunk;
        const float value = float(storage.value_bf16[
            bf16_tsm_offset(value_column, token, kChunk)])
            * storage.decay_to_last[token];
        b_tsm[bf16_tsm_offset(value_column, token, kChunk)] = Element(value);
      }
      __syncthreads();
      mma_from_tsm<Tile1286464>(
          a_tsm, b_tsm, thread, warp, state, false);
      __syncthreads();
    }

    for (int index = 0; index < size(state); ++index) {
      auto coordinate = state_coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      params.final_state[
          state_base + row * kDim + value_start + column] = state(index);
    }
  }
};

// Official backward W/U preparation. One CTA owns one chunk/value-head and
// computes W=A diag(beta*exp(g)) K and U=A diag(beta) V directly from the
// token-major tensors. Both AIU GEMMs share the staged KKT/A tile, avoiding
// the framework head-major transposes without changing the equations.
struct FlashQlaWuForward {
  struct Arguments {
    const float* k;
    const float* v;
    const float* a;
    const float* g;
    const float* beta;
    float* w;
    float* u;
    int batch_size;
    int tokens;
    int q_heads;
    int value_heads;
  };
  using Params = Arguments;

  struct SharedStorage {
    cute::array_aligned<Element, 64 * 64> a_tsm;
    cute::array_aligned<Element, 128 * 64> vector_tsm;
    cute::array_aligned<float, 64> beta;
    cute::array_aligned<float, 64> beta_exp_g;
  };

  static constexpr int MaxThreadsPerBlock = size(TiledMma{});
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    const int chunks = params.tokens / kChunk;
    const int group = blockIdx.x;
    const int groups = params.batch_size * chunks * params.value_heads;
    if (group >= groups) {
      return;
    }
    const int head = group % params.value_heads;
    const int batch_chunk = group / params.value_heads;
    const int chunk = batch_chunk % chunks;
    const int batch = batch_chunk / chunks;
    const int q_head = head / (params.value_heads / params.q_heads);
    const int token_base = chunk * kChunk;
    const int thread = threadIdx.x;
    int warp = cutlass::canonical_warp_idx_sync();
    auto& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    if (thread < kChunk) {
      const int scalar = (batch * params.tokens + token_base + thread)
          * params.value_heads + head;
      const float beta = params.beta[scalar];
      storage.beta[thread] = beta;
      storage.beta_exp_g[thread] = beta * expf(params.g[scalar]);
    }
    __syncthreads();

    for (int linear = thread; linear < kChunk * kChunk;
         linear += blockDim.x) {
      const int row = linear / kChunk;
      const int column = linear - row * kChunk;
      const int source = (((batch * params.tokens + token_base + row)
          * params.value_heads + head) * kChunk) + column;
      storage.a_tsm[bf16_tsm_offset(row, column, kChunk)] =
          Element(params.a[source]);
    }
    for (int linear = thread; linear < kChunk * kDim;
         linear += blockDim.x) {
      const int row = linear / kDim;
      const int column = linear - row * kDim;
      const int token = token_base + row;
      const int source = ((batch * params.tokens + token)
          * params.q_heads + q_head) * kDim + column;
      storage.vector_tsm[bf16_tsm_offset(column, row, kChunk)] =
          Element(params.k[source] * storage.beta_exp_g[row]);
    }
    __syncthreads();

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread);
    Tensor output = partition_fragment_C(
        tiled_mma, Shape<Int<64>, Int<128>>{});
    Tensor coordinates = thr_mma.partition_C(
        make_identity_tensor(Shape<Int<64>, Int<128>>{}));
    mma_from_tsm<Tile6412864>(
        storage.a_tsm.data(), storage.vector_tsm.data(),
        thread, warp, output, true);
    for (int index = 0; index < size(output); ++index) {
      auto coordinate = coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      const int destination = ((batch * params.tokens + token_base + row)
          * params.value_heads + head) * kDim + column;
      params.w[destination] = output(index);
    }
    __syncthreads();

    for (int linear = thread; linear < kChunk * kDim;
         linear += blockDim.x) {
      const int row = linear / kDim;
      const int column = linear - row * kDim;
      const int token = token_base + row;
      const int source = ((batch * params.tokens + token)
          * params.value_heads + head) * kDim + column;
      storage.vector_tsm[bf16_tsm_offset(column, row, kChunk)] =
          Element(params.v[source] * storage.beta[row]);
    }
    __syncthreads();
    mma_from_tsm<Tile6412864>(
        storage.a_tsm.data(), storage.vector_tsm.data(),
        thread, warp, output, true);
    for (int index = 0; index < size(output); ++index) {
      auto coordinate = coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      const int destination = ((batch * params.tokens + token_base + row)
          * params.value_heads + head) * kDim + column;
      params.u[destination] = output(index);
    }
  }
};

// Native prepare-H stage used by the official backward dataflow. One CTA owns
// a 128x64 state tile, persists it across all chunks, and emits both the chunk
// boundary history H and the corrected values Vn. This removes the Python
// per-chunk launch loop while preserving the official BF16 MMA handoffs.
struct FlashQlaChunkStateForward {
  struct Arguments {
    const float* k;
    const float* w;
    const float* u;
    const float* g;
    const float* initial_state;
    float* history;
    float* vn;
    float* final_state;
    int batch_size;
    int tokens;
    int q_heads;
    int value_heads;
    bool use_initial_state;
  };
  using Params = Arguments;

  struct SharedStorage {
    cute::array_aligned<Element, 128 * 64> a_tsm;
    cute::array_aligned<Element, 128 * 64> b_tsm;
    cute::array_aligned<float, 64> gate;
    cute::array_aligned<float, 64> decay_to_last;
  };

  static constexpr int MaxThreadsPerBlock = size(TiledMma{});
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    const int tile_group = blockIdx.x;
    const int groups = params.batch_size * params.value_heads;
    if (tile_group >= groups * 2) {
      return;
    }
    const int group = tile_group / 2;
    const int value_tile = tile_group - group * 2;
    const int value_start = value_tile * 64;
    const int batch = group / params.value_heads;
    const int head = group - batch * params.value_heads;
    const int q_head = head / (params.value_heads / params.q_heads);
    const int chunks = params.tokens / kChunk;
    const int thread = threadIdx.x;
    int warp = cutlass::canonical_warp_idx_sync();
    auto& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread);
    Tensor state = partition_fragment_C(
        tiled_mma, Shape<Int<128>, Int<64>>{});
    Tensor state_coordinates = thr_mma.partition_C(
        make_identity_tensor(Shape<Int<128>, Int<64>>{}));
    Tensor value = partition_fragment_C(
        tiled_mma, Shape<Int<64>, Int<64>>{});
    Tensor value_coordinates = thr_mma.partition_C(
        make_identity_tensor(Shape<Int<64>, Int<64>>{}));

    const int state_base = group * kDim * kDim;
    for (int index = 0; index < size(state); ++index) {
      auto coordinate = state_coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      state(index) = params.use_initial_state
          ? params.initial_state[
                state_base + row * kDim + value_start + column]
          : 0.0f;
    }

    for (int chunk = 0; chunk < chunks; ++chunk) {
      const int token_base = chunk * kChunk;
      const int gate_base = (group * chunks + chunk) * kChunk;
      if (thread < kChunk) {
        storage.gate[thread] = params.g[gate_base + thread];
      }
      for (int index = 0; index < size(state); ++index) {
        auto coordinate = state_coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        const int destination = (((batch * chunks + chunk)
            * params.value_heads + head) * kDim + row) * kDim
            + value_start + column;
        params.history[destination] = state(index);
        storage.b_tsm[bf16_tsm_offset(column, row, kChunk)] =
            Element(state(index));
      }
      for (int linear = thread; linear < kChunk * kDim;
           linear += blockDim.x) {
        const int row = linear / kDim;
        const int column = linear - row * kDim;
        const int source = (((batch * params.tokens + token_base + row)
            * params.value_heads + head) * kDim) + column;
        storage.a_tsm[bf16_tsm_offset(row, column, kChunk)] =
            Element(params.w[source]);
      }
      __syncthreads();
      mma_from_tsm<Tile6464128>(
          storage.a_tsm.data(), storage.b_tsm.data(),
          thread, warp, value, true);
      __syncthreads();

      const float g_last = storage.gate[kChunk - 1];
      if (thread < kChunk) {
        storage.decay_to_last[thread] = stable_exp_difference(
            g_last - storage.gate[thread]);
      }
      __syncthreads();
      for (int index = 0; index < size(value); ++index) {
        auto coordinate = value_coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        const int destination = (((batch * params.tokens + token_base + row)
            * params.value_heads + head) * kDim) + value_start + column;
        value(index) = params.u[destination] - value(index);
        params.vn[destination] = value(index);
        storage.b_tsm[bf16_tsm_offset(column, row, kChunk)] = Element(
            value(index) * storage.decay_to_last[row]);
      }
      const float state_decay = expf(g_last);
      for (int index = 0; index < size(state); ++index) {
        state(index) *= state_decay;
      }
      for (int linear = thread; linear < kDim * kChunk;
           linear += blockDim.x) {
        const int row = linear / kChunk;
        const int column = linear - row * kChunk;
        const int source = (((batch * params.tokens + token_base + column)
            * params.q_heads + q_head) * kDim) + row;
        storage.a_tsm[bf16_tsm_offset(row, column, kDim)] =
            Element(params.k[source]);
      }
      __syncthreads();
      mma_from_tsm<Tile1286464>(
          storage.a_tsm.data(), storage.b_tsm.data(),
          thread, warp, state, false);
      __syncthreads();
    }

    for (int index = 0; index < size(state); ++index) {
      auto coordinate = state_coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      params.final_state[
          state_base + row * kDim + value_start + column] = state(index);
    }
  }
};

// Official fused_gdr_h topology for the backward recomputation path.  One CTA
// owns one batch/value-head, keeps the complete 128x128 state in FP32
// accumulators, and walks chunks forward.  W is published because the reverse
// recurrence consumes it later; U never leaves the CTA.  This folds the
// separate W/U launch and its U global-memory round trip into prepare-H while
// preserving the official BF16 MMA handoffs.
struct FlashQlaFusedPrepareH {
  struct Arguments {
    const float* k;
    const float* v;
    const float* a;
    const float* g;
    const float* beta;
    const float* initial_state;
    float* w;
    float* history;
    float* vn;
    float* final_state;
    int batch_size;
    int tokens;
    int q_heads;
    int value_heads;
    bool use_initial_state;
  };
  using Params = Arguments;

  struct SharedStorage {
    cute::array_aligned<Element, 64 * 64> a_tsm;
    cute::array_aligned<Element, 128 * 64> vector_tsm;
    cute::array_aligned<Element, 64 * 128> w_or_value_tsm;
    cute::array_aligned<Element, 128 * 128> state_tsm;
    cute::array_aligned<float, 64 * 128> ws;
    cute::array_aligned<float, 64> gate;
    cute::array_aligned<float, 64> beta;
    cute::array_aligned<float, 64> beta_exp_g;
    cute::array_aligned<float, 64> decay_to_last;
  };

  static constexpr int MaxThreadsPerBlock = size(TiledMma{});
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    const int group = blockIdx.x;
    if (group >= params.batch_size * params.value_heads) {
      return;
    }
    const int batch = group / params.value_heads;
    const int head = group - batch * params.value_heads;
    const int q_head = head / (params.value_heads / params.q_heads);
    const int chunks = params.tokens / kChunk;
    const int thread = threadIdx.x;
    int warp = cutlass::canonical_warp_idx_sync();
    auto& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread);
    Tensor state = partition_fragment_C(
        tiled_mma, Shape<Int<128>, Int<128>>{});
    Tensor state_coordinates = thr_mma.partition_C(
        make_identity_tensor(Shape<Int<128>, Int<128>>{}));
    Tensor work = partition_fragment_C(
        tiled_mma, Shape<Int<64>, Int<128>>{});
    Tensor work_coordinates = thr_mma.partition_C(
        make_identity_tensor(Shape<Int<64>, Int<128>>{}));

    const int state_base = group * kDim * kDim;
    for (int index = 0; index < size(state); ++index) {
      auto coordinate = state_coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      state(index) = params.use_initial_state
          ? params.initial_state[state_base + row * kDim + column]
          : 0.0f;
    }

    for (int chunk = 0; chunk < chunks; ++chunk) {
      const int token_base = chunk * kChunk;
      const int gate_base = (group * chunks + chunk) * kChunk;
      for (int index = 0; index < size(state); ++index) {
        auto coordinate = state_coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        const int destination = (((batch * chunks + chunk)
            * params.value_heads + head) * kDim + row) * kDim + column;
        params.history[destination] = state(index);
        storage.state_tsm[bf16_tsm_offset(column, row, kDim)] =
            Element(state(index));
      }
      if (thread < kChunk) {
        const int scalar = (batch * params.tokens + token_base + thread)
            * params.value_heads + head;
        const float gate = params.g[gate_base + thread];
        const float beta = params.beta[scalar];
        storage.gate[thread] = gate;
        storage.beta[thread] = beta;
        storage.beta_exp_g[thread] = beta * expf(gate);
      }
      for (int linear = thread; linear < kChunk * kChunk;
           linear += blockDim.x) {
        const int row = linear / kChunk;
        const int column = linear - row * kChunk;
        const int source = (((batch * params.tokens + token_base + row)
            * params.value_heads + head) * kChunk) + column;
        storage.a_tsm[bf16_tsm_offset(row, column, kChunk)] =
            Element(params.a[source]);
      }
      __syncthreads();

      // W = A @ (beta * exp(g) * K).
      for (int linear = thread; linear < kChunk * kDim;
           linear += blockDim.x) {
        const int row = linear / kDim;
        const int column = linear - row * kDim;
        const int source = (((batch * params.tokens + token_base + row)
            * params.q_heads + q_head) * kDim) + column;
        storage.vector_tsm[bf16_tsm_offset(column, row, kChunk)] = Element(
            params.k[source] * storage.beta_exp_g[row]);
      }
      __syncthreads();
      mma_from_tsm<Tile6412864>(
          storage.a_tsm.data(), storage.vector_tsm.data(),
          thread, warp, work, true);
      for (int index = 0; index < size(work); ++index) {
        auto coordinate = work_coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        const int destination = (((batch * params.tokens + token_base + row)
            * params.value_heads + head) * kDim) + column;
        params.w[destination] = work(index);
        storage.w_or_value_tsm[bf16_tsm_offset(row, column, kChunk)] =
            Element(work(index));
      }
      __syncthreads();

      // W @ S0.
      mma_from_tsm<Tile64128128>(
          storage.w_or_value_tsm.data(), storage.state_tsm.data(),
          thread, warp, work, true);
      for (int index = 0; index < size(work); ++index) {
        auto coordinate = work_coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        storage.ws[row * kDim + column] = work(index);
      }
      __syncthreads();

      // U = A @ (beta * V), then V' = U - W @ S0.
      for (int linear = thread; linear < kChunk * kDim;
           linear += blockDim.x) {
        const int row = linear / kDim;
        const int column = linear - row * kDim;
        const int source = (((batch * params.tokens + token_base + row)
            * params.value_heads + head) * kDim) + column;
        storage.vector_tsm[bf16_tsm_offset(column, row, kChunk)] = Element(
            params.v[source] * storage.beta[row]);
      }
      __syncthreads();
      mma_from_tsm<Tile6412864>(
          storage.a_tsm.data(), storage.vector_tsm.data(),
          thread, warp, work, true);
      const float g_last = storage.gate[kChunk - 1];
      if (thread < kChunk) {
        storage.decay_to_last[thread] = stable_exp_difference(
            g_last - storage.gate[thread]);
      }
      __syncthreads();
      for (int index = 0; index < size(work); ++index) {
        auto coordinate = work_coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        work(index) -= storage.ws[row * kDim + column];
        const int destination = (((batch * params.tokens + token_base + row)
            * params.value_heads + head) * kDim) + column;
        params.vn[destination] = work(index);
        storage.w_or_value_tsm[
            bf16_tsm_offset(column, row, kDim)] = Element(
                work(index) * storage.decay_to_last[row]);
      }
      const float state_decay = expf(g_last);
      for (int index = 0; index < size(state); ++index) {
        state(index) *= state_decay;
      }
      for (int linear = thread; linear < kDim * kChunk;
           linear += blockDim.x) {
        const int row = linear / kChunk;
        const int column = linear - row * kChunk;
        const int source = (((batch * params.tokens + token_base + column)
            * params.q_heads + q_head) * kDim) + row;
        storage.vector_tsm[bf16_tsm_offset(row, column, kDim)] =
            Element(params.k[source]);
      }
      __syncthreads();
      mma_from_tsm<Tile12812864>(
          storage.vector_tsm.data(), storage.w_or_value_tsm.data(),
          thread, warp, state, false);
      __syncthreads();
    }

    for (int index = 0; index < size(state); ++index) {
      auto coordinate = state_coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      params.final_state[state_base + row * kDim + column] = state(index);
    }
  }
};

// Reverse prepare-H stage from the official fused backward. The state-gradient
// tile remains in FP32 registers while one CTA walks chunks in reverse and
// produces dH plus the accumulated dV contribution.
struct FlashQlaChunkStateBackward {
  struct Arguments {
    const float* q;
    const float* k;
    const float* w;
    const float* g;
    const float* do_value;
    const float* dv_input;
    const float* terminal_state_grad;
    float* dh;
    float* dv_output;
    float* initial_state_grad;
    int batch_size;
    int tokens;
    int q_heads;
    int value_heads;
    float scale;
    bool use_terminal_state_grad;
  };
  using Params = Arguments;

  struct SharedStorage {
    cute::array_aligned<Element, 128 * 64> a_tsm;
    cute::array_aligned<Element, 128 * 64> b_tsm;
    cute::array_aligned<float, 64> gate;
  };

  static constexpr int MaxThreadsPerBlock = size(TiledMma{});
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    const int tile_group = blockIdx.x;
    const int groups = params.batch_size * params.value_heads;
    if (tile_group >= groups * 2) {
      return;
    }
    const int group = tile_group / 2;
    const int value_tile = tile_group - group * 2;
    const int value_start = value_tile * 64;
    const int batch = group / params.value_heads;
    const int head = group - batch * params.value_heads;
    const int q_head = head / (params.value_heads / params.q_heads);
    const int chunks = params.tokens / kChunk;
    const int thread = threadIdx.x;
    int warp = cutlass::canonical_warp_idx_sync();
    auto& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread);
    Tensor dstate = partition_fragment_C(
        tiled_mma, Shape<Int<128>, Int<64>>{});
    Tensor state_coordinates = thr_mma.partition_C(
        make_identity_tensor(Shape<Int<128>, Int<64>>{}));
    Tensor value = partition_fragment_C(
        tiled_mma, Shape<Int<64>, Int<64>>{});
    Tensor value_coordinates = thr_mma.partition_C(
        make_identity_tensor(Shape<Int<64>, Int<64>>{}));

    const int state_base = group * kDim * kDim;
    for (int index = 0; index < size(dstate); ++index) {
      auto coordinate = state_coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      dstate(index) = params.use_terminal_state_grad
          ? params.terminal_state_grad[
                state_base + row * kDim + value_start + column]
          : 0.0f;
    }

    for (int chunk = chunks - 1; chunk >= 0; --chunk) {
      const int token_base = chunk * kChunk;
      const int gate_base = (group * chunks + chunk) * kChunk;
      if (thread < kChunk) {
        storage.gate[thread] = params.g[gate_base + thread];
      }
      for (int index = 0; index < size(dstate); ++index) {
        auto coordinate = state_coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        const int destination = (((batch * chunks + chunk)
            * params.value_heads + head) * kDim + row) * kDim
            + value_start + column;
        params.dh[destination] = dstate(index);
        storage.b_tsm[bf16_tsm_offset(column, row, kChunk)] =
            Element(dstate(index));
      }
      __syncthreads();

      // dV += diag(exp(g_last-g)) K dState.
      const float g_last = storage.gate[kChunk - 1];
      for (int linear = thread; linear < kChunk * kDim;
           linear += blockDim.x) {
        const int row = linear / kDim;
        const int column = linear - row * kDim;
        const int source = (((batch * params.tokens + token_base + row)
            * params.q_heads + q_head) * kDim) + column;
        storage.a_tsm[bf16_tsm_offset(row, column, kChunk)] = Element(
            params.k[source] * stable_exp_difference(
                g_last - storage.gate[row]));
      }
      __syncthreads();
      mma_from_tsm<Tile6464128>(
          storage.a_tsm.data(), storage.b_tsm.data(),
          thread, warp, value, true);
      for (int index = 0; index < size(value); ++index) {
        auto coordinate = value_coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        const int destination = (((batch * params.tokens + token_base + row)
            * params.value_heads + head) * kDim) + value_start + column;
        value(index) += params.dv_input[destination];
        params.dv_output[destination] = value(index);
        storage.b_tsm[bf16_tsm_offset(column, row, kChunk)] =
            Element(value(index));
      }
      for (int index = 0; index < size(dstate); ++index) {
        dstate(index) *= expf(g_last);
      }
      __syncthreads();

      // dState += scale * (Q * exp(g))^T @ dO.
      for (int linear = thread; linear < kDim * kChunk;
           linear += blockDim.x) {
        const int row = linear / kChunk;
        const int column = linear - row * kChunk;
        const int source = (((batch * params.tokens + token_base + column)
            * params.q_heads + q_head) * kDim) + row;
        storage.a_tsm[bf16_tsm_offset(row, column, kDim)] = Element(
            params.q[source] * params.scale * expf(storage.gate[column]));
      }
      for (int linear = thread; linear < kChunk * 64;
           linear += blockDim.x) {
        const int row = linear / 64;
        const int column = linear - row * 64;
        const int source = (((batch * params.tokens + token_base + row)
            * params.value_heads + head) * kDim) + value_start + column;
        storage.b_tsm[bf16_tsm_offset(column, row, kChunk)] =
            Element(params.do_value[source]);
      }
      __syncthreads();
      mma_from_tsm<Tile1286464>(
          storage.a_tsm.data(), storage.b_tsm.data(),
          thread, warp, dstate, false);
      __syncthreads();

      // dState -= W^T @ dV.
      for (int linear = thread; linear < kDim * kChunk;
           linear += blockDim.x) {
        const int row = linear / kChunk;
        const int column = linear - row * kChunk;
        const int source = (((batch * params.tokens + token_base + column)
            * params.value_heads + head) * kDim) + row;
        storage.a_tsm[bf16_tsm_offset(row, column, kDim)] =
            Element(-params.w[source]);
      }
      for (int linear = thread; linear < kChunk * 64;
           linear += blockDim.x) {
        const int row = linear / 64;
        const int column = linear - row * 64;
        const int source = (((batch * params.tokens + token_base + row)
            * params.value_heads + head) * kDim) + value_start + column;
        storage.b_tsm[bf16_tsm_offset(column, row, kChunk)] =
            Element(params.dv_output[source]);
      }
      __syncthreads();
      mma_from_tsm<Tile1286464>(
          storage.a_tsm.data(), storage.b_tsm.data(),
          thread, warp, dstate, false);
      __syncthreads();
    }

    for (int index = 0; index < size(dstate); ++index) {
      auto coordinate = state_coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      params.initial_state_grad[
          state_base + row * kDim + value_start + column] = dstate(index);
    }
  }
};

// One reverse chunk step with the same official recurrence as the persistent
// kernel above. AutoCP supplies many independent segments, so launching this
// grid once per chunk exposes all segment/head/value tiles concurrently while
// keeping dV, dH publication and both state MMAs in one kernel.
struct FlashQlaChunkStateBackwardStep {
  struct Arguments {
    const float* q;
    const float* k;
    const float* w;
    const float* g;
    const float* do_value;
    float* dv;
    float* dstate;
    float* dh;
    int batch_size;
    int tokens;
    int q_heads;
    int value_heads;
    int chunk;
    float scale;
  };
  using Params = Arguments;

  struct SharedStorage {
    cute::array_aligned<Element, 128 * 64> a_tsm;
    cute::array_aligned<Element, 128 * 64> b_tsm;
    cute::array_aligned<float, 64> gate;
  };

  static constexpr int MaxThreadsPerBlock = size(TiledMma{});
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    const int tile_group = blockIdx.x;
    const int groups = params.batch_size * params.value_heads;
    if (tile_group >= groups * 2) {
      return;
    }
    const int group = tile_group / 2;
    const int value_tile = tile_group - group * 2;
    const int value_start = value_tile * 64;
    const int batch = group / params.value_heads;
    const int head = group - batch * params.value_heads;
    const int q_head = head / (params.value_heads / params.q_heads);
    const int chunks = params.tokens / kChunk;
    const int token_base = params.chunk * kChunk;
    const int gate_base = (group * chunks + params.chunk) * kChunk;
    const int thread = threadIdx.x;
    int warp = cutlass::canonical_warp_idx_sync();
    auto& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread);
    Tensor state = partition_fragment_C(
        tiled_mma, Shape<Int<128>, Int<64>>{});
    Tensor state_coordinates = thr_mma.partition_C(
        make_identity_tensor(Shape<Int<128>, Int<64>>{}));
    Tensor value = partition_fragment_C(
        tiled_mma, Shape<Int<64>, Int<64>>{});
    Tensor value_coordinates = thr_mma.partition_C(
        make_identity_tensor(Shape<Int<64>, Int<64>>{}));

    const int state_base = group * kDim * kDim;
    for (int index = 0; index < size(state); ++index) {
      auto coordinate = state_coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      const int state_index = state_base + row * kDim
          + value_start + column;
      state(index) = params.dstate[state_index];
      const int history = (((batch * chunks + params.chunk)
          * params.value_heads + head) * kDim + row) * kDim
          + value_start + column;
      params.dh[history] = state(index);
      storage.b_tsm[bf16_tsm_offset(column, row, kChunk)] =
          Element(state(index));
    }
    if (thread < kChunk) {
      storage.gate[thread] = params.g[gate_base + thread];
    }
    __syncthreads();

    const float g_last = storage.gate[kChunk - 1];
    for (int linear = thread; linear < kChunk * kDim;
         linear += blockDim.x) {
      const int row = linear / kDim;
      const int column = linear - row * kDim;
      const int source = (((batch * params.tokens + token_base + row)
          * params.q_heads + q_head) * kDim) + column;
      storage.a_tsm[bf16_tsm_offset(row, column, kChunk)] = Element(
          params.k[source] * stable_exp_difference(
              g_last - storage.gate[row]));
    }
    __syncthreads();
    mma_from_tsm<Tile6464128>(
        storage.a_tsm.data(), storage.b_tsm.data(),
        thread, warp, value, true);
    for (int index = 0; index < size(value); ++index) {
      auto coordinate = value_coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      const int destination = (((batch * params.tokens + token_base + row)
          * params.value_heads + head) * kDim) + value_start + column;
      value(index) += params.dv[destination];
      params.dv[destination] = value(index);
      storage.b_tsm[bf16_tsm_offset(column, row, kChunk)] =
          Element(value(index));
    }
    for (int index = 0; index < size(state); ++index) {
      state(index) *= expf(g_last);
    }
    __syncthreads();

    for (int linear = thread; linear < kDim * kChunk;
         linear += blockDim.x) {
      const int row = linear / kChunk;
      const int column = linear - row * kChunk;
      const int source = (((batch * params.tokens + token_base + column)
          * params.q_heads + q_head) * kDim) + row;
      storage.a_tsm[bf16_tsm_offset(row, column, kDim)] = Element(
          params.q[source] * params.scale * expf(storage.gate[column]));
    }
    for (int linear = thread; linear < kChunk * 64;
         linear += blockDim.x) {
      const int row = linear / 64;
      const int column = linear - row * 64;
      const int source = (((batch * params.tokens + token_base + row)
          * params.value_heads + head) * kDim) + value_start + column;
      storage.b_tsm[bf16_tsm_offset(column, row, kChunk)] =
          Element(params.do_value[source]);
    }
    __syncthreads();
    mma_from_tsm<Tile1286464>(
        storage.a_tsm.data(), storage.b_tsm.data(),
        thread, warp, state, false);
    __syncthreads();

    for (int linear = thread; linear < kDim * kChunk;
         linear += blockDim.x) {
      const int row = linear / kChunk;
      const int column = linear - row * kChunk;
      const int source = (((batch * params.tokens + token_base + column)
          * params.value_heads + head) * kDim) + row;
      storage.a_tsm[bf16_tsm_offset(row, column, kDim)] =
          Element(-params.w[source]);
    }
    for (int linear = thread; linear < kChunk * 64;
         linear += blockDim.x) {
      const int row = linear / 64;
      const int column = linear - row * 64;
      const int source = (((batch * params.tokens + token_base + row)
          * params.value_heads + head) * kDim) + value_start + column;
      storage.b_tsm[bf16_tsm_offset(column, row, kChunk)] =
          Element(params.dv[source]);
    }
    __syncthreads();
    mma_from_tsm<Tile1286464>(
        storage.a_tsm.data(), storage.b_tsm.data(),
        thread, warp, state, false);

    for (int index = 0; index < size(state); ++index) {
      auto coordinate = state_coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      params.dstate[state_base + row * kDim + value_start + column] =
          state(index);
    }
  }
};

// Persistent reverse K/S schedule. A single CTA owns the complete 128x128
// state gradient for one batch/value-head. The K consumer computes the full
// dV tile while the S consumer concurrently accumulates Q^T@dO, then consumes
// the published dV for the W^T@dV update. This is the reverse-state core of
// the official fused backward without splitting the value dimension.
struct FlashQlaChunkStateBackwardWarpSpecialized {
  using Arguments = FlashQlaChunkStateBackward::Arguments;
  using Params = Arguments;

  struct SharedStorage {
    cute::array_aligned<Element, 128 * 128> dstate_b;
    cute::array_aligned<Element, 64 * 128> k_a;
    cute::array_aligned<Element, 128 * 64> state_a;
    cute::array_aligned<Element, 128 * 64> state_b;
  };

  static constexpr int MaxThreadsPerBlock = 256;
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    const int group = blockIdx.x;
    const int groups = params.batch_size * params.value_heads;
    if (group >= groups) {
      return;
    }
    const int consumer = threadIdx.x / 128;
    const int thread = threadIdx.x - consumer * 128;
    const int warp = thread / 32;
    const int batch = group / params.value_heads;
    const int head = group - batch * params.value_heads;
    const int q_head = head / (params.value_heads / params.q_heads);
    const int chunks = params.tokens / kChunk;
    const int state_base = group * kDim * kDim;
    auto& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread);
    if (consumer == 0) {
      Tensor dstate = partition_fragment_C(
          tiled_mma, Shape<Int<128>, Int<128>>{});
      Tensor coordinates = thr_mma.partition_C(
          make_identity_tensor(Shape<Int<128>, Int<128>>{}));
      for (int index = 0; index < size(dstate); ++index) {
        auto coordinate = coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        dstate(index) = params.use_terminal_state_grad
            ? params.terminal_state_grad[
                  state_base + row * kDim + column]
            : 0.0f;
      }

      for (int chunk = chunks - 1; chunk >= 0; --chunk) {
        const int token_base = chunk * kChunk;
        const int gate_base = (group * chunks + chunk) * kChunk;
        const float g_last = params.g[gate_base + kChunk - 1];

        // Publish dH for this chunk and its official BF16 K-consumer handoff.
        for (int index = 0; index < size(dstate); ++index) {
          auto coordinate = coordinates(index);
          const int row = get<0>(coordinate);
          const int column = get<1>(coordinate);
          const int destination = (((batch * chunks + chunk)
              * params.value_heads + head) * kDim + row) * kDim + column;
          params.dh[destination] = dstate(index);
          storage.dstate_b[bf16_tsm_offset(column, row, kDim)] =
              Element(dstate(index));
        }

        // Prepare Q^T and dO^T while K prepares its decayed key tile.
        for (int linear = thread; linear < kDim * kChunk; linear += 128) {
          const int row = linear / kChunk;
          const int column = linear - row * kChunk;
          const int source = (((batch * params.tokens + token_base + column)
              * params.q_heads + q_head) * kDim) + row;
          storage.state_a[bf16_tsm_offset(row, column, kDim)] = Element(
              params.q[source] * params.scale
              * expf(params.g[gate_base + column]));
        }
        for (int linear = thread; linear < kChunk * kDim; linear += 128) {
          const int row = linear / kDim;
          const int column = linear - row * kDim;
          const int source = (((batch * params.tokens + token_base + row)
              * params.value_heads + head) * kDim) + column;
          storage.state_b[bf16_tsm_offset(column, row, kDim)] =
              Element(params.do_value[source]);
        }
        __syncthreads();

        // S and K execute their first MMA concurrently.
        for (int index = 0; index < size(dstate); ++index) {
          dstate(index) *= expf(g_last);
        }
        mma_from_tsm<Tile12812864>(
            storage.state_a.data(), storage.state_b.data(),
            thread, warp, dstate, false);
        __syncthreads();

        // Wait for K to publish the complete dV tile into state_b.
        __syncthreads();
        for (int linear = thread; linear < kDim * kChunk; linear += 128) {
          const int row = linear / kChunk;
          const int column = linear - row * kChunk;
          const int source = (((batch * params.tokens + token_base + column)
              * params.value_heads + head) * kDim) + row;
          storage.state_a[bf16_tsm_offset(row, column, kDim)] =
              Element(-params.w[source]);
        }
        __syncthreads();
        mma_from_tsm<Tile12812864>(
            storage.state_a.data(), storage.state_b.data(),
            thread, warp, dstate, false);
        __syncthreads();
      }

      for (int index = 0; index < size(dstate); ++index) {
        auto coordinate = coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        params.initial_state_grad[state_base + row * kDim + column] =
            dstate(index);
      }
    } else {
      Tensor value = partition_fragment_C(
          tiled_mma, Shape<Int<64>, Int<128>>{});
      Tensor coordinates = thr_mma.partition_C(
          make_identity_tensor(Shape<Int<64>, Int<128>>{}));

      for (int chunk = chunks - 1; chunk >= 0; --chunk) {
        const int token_base = chunk * kChunk;
        const int gate_base = (group * chunks + chunk) * kChunk;
        const float g_last = params.g[gate_base + kChunk - 1];
        for (int linear = thread; linear < kChunk * kDim; linear += 128) {
          const int row = linear / kDim;
          const int column = linear - row * kDim;
          const int source = (((batch * params.tokens + token_base + row)
              * params.q_heads + q_head) * kDim) + column;
          storage.k_a[bf16_tsm_offset(row, column, kChunk)] = Element(
              params.k[source] * stable_exp_difference(
                  g_last - params.g[gate_base + row]));
        }
        __syncthreads();
        mma_from_tsm<Tile64128128>(
            storage.k_a.data(), storage.dstate_b.data(),
            thread, warp, value, true);
        __syncthreads();

        for (int index = 0; index < size(value); ++index) {
          auto coordinate = coordinates(index);
          const int row = get<0>(coordinate);
          const int column = get<1>(coordinate);
          const int destination = (((batch * params.tokens + token_base + row)
              * params.value_heads + head) * kDim) + column;
          value(index) += params.dv_input[destination];
          params.dv_output[destination] = value(index);
          storage.state_b[bf16_tsm_offset(column, row, kDim)] =
              Element(value(index));
        }
        __syncthreads();
        __syncthreads();
        __syncthreads();
      }
    }
  }
};

// Official AutoCP backward warmup. One CTA owns a CP segment/value-head:
// PR computes P/R, XY computes X/Y, and DH keeps the 128x128 state gradient
// resident. The three 128-thread consumers mirror the official Hopper
// cp_bwd producer/consumer algebra while using PPU TSM handoffs.
struct FlashQlaCpDhBackwardWarpSpecialized {
  struct Arguments {
    const float* q;
    const float* k;
    const float* a;
    const float* g;
    const float* beta;
    const float* do_value;
    float* dh0;
    int batch_size;
    int tokens;
    int q_heads;
    int value_heads;
    float scale;
  };
  using Params = Arguments;

  struct SharedStorage {
    cute::array_aligned<Element, 64 * 128> q_tsm;
    cute::array_aligned<Element, 64 * 128> k_p_tsm;
    cute::array_aligned<Element, 128 * 64> k_xy_tsm;
    cute::array_aligned<Element, 128 * 64> x_tsm;
    cute::array_aligned<Element, 64 * 128> dh_tsm;
    cute::array_aligned<Element, 64 * 64> ap_tsm;
    cute::array_aligned<Element, 64 * 64> do_tsm;
    cute::array_aligned<float, 64> gate;
  };

  static constexpr int MaxThreadsPerBlock = 384;
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    const int tile_group = blockIdx.x;
    const int groups = params.batch_size * params.value_heads;
    if (tile_group >= groups * 2) {
      return;
    }
    const int group = tile_group / 2;
    const int value_start = (tile_group - group * 2) * 64;
    const int consumer = threadIdx.x / 128;
    const int thread = threadIdx.x - consumer * 128;
    const int warp = thread / 32;
    const int batch = group / params.value_heads;
    const int head = group - batch * params.value_heads;
    const int q_head = head / (params.value_heads / params.q_heads);
    const int chunks = params.tokens / kChunk;
    auto& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread);

    if (consumer == 0) {
      Tensor dh = partition_fragment_C(
          tiled_mma, Shape<Int<128>, Int<64>>{});
      Tensor coordinates = thr_mma.partition_C(
          make_identity_tensor(Shape<Int<128>, Int<64>>{}));
      clear(dh);

      for (int chunk = chunks - 1; chunk >= 0; --chunk) {
        const int token_base = chunk * kChunk;
        const int gate_base = (group * chunks + chunk) * kChunk;
        for (int index = 0; index < size(dh); ++index) {
          auto coordinate = coordinates(index);
          const int row = get<0>(coordinate);
          const int column = get<1>(coordinate);
          storage.dh_tsm[bf16_tsm_offset(column, row, kChunk)] =
              Element(dh(index));
        }
        for (int linear = thread; linear < kChunk * 64; linear += 128) {
          const int row = linear / 64;
          const int column = linear - row * 64;
          const int source = (((batch * params.tokens + token_base + row)
              * params.value_heads + head) * kDim) + value_start + column;
          storage.do_tsm[bf16_tsm_offset(column, row, kChunk)] =
              Element(params.do_value[source]);
        }
        if (thread < kChunk) {
          storage.gate[thread] = params.g[gate_base + thread];
        }
        __syncthreads();

        for (int index = 0; index < size(dh); ++index) {
          dh(index) *= expf(storage.gate[kChunk - 1]);
        }
        __syncthreads();
        __syncthreads();
        __syncthreads();

        mma_from_tsm<Tile1286464>(
            storage.q_tsm.data(), storage.do_tsm.data(),
            thread, warp, dh, false);
        mma_from_tsm<Tile1286464>(
            storage.x_tsm.data(), storage.k_xy_tsm.data(),
            thread, warp, dh, false);
        __syncthreads();
      }

      const int state_base = group * kDim * kDim;
      for (int index = 0; index < size(dh); ++index) {
        auto coordinate = coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        params.dh0[state_base + row * kDim + value_start + column] = dh(index);
      }
    } else if (consumer == 1) {
      for (int chunk = chunks - 1; chunk >= 0; --chunk) {
        const int token_base = chunk * kChunk;
        const int gate_base = (group * chunks + chunk) * kChunk;
        for (int linear = thread; linear < kChunk * kDim; linear += 128) {
          const int row = linear / kDim;
          const int column = linear - row * kDim;
          const int q_source = (((batch * params.tokens + token_base + row)
              * params.q_heads + q_head) * kDim) + column;
          storage.q_tsm[bf16_tsm_offset(row, column, kChunk)] =
              Element(params.q[q_source]);
          const int k_source = (((batch * params.tokens + token_base + row)
              * params.q_heads + q_head) * kDim) + column;
          storage.k_p_tsm[bf16_tsm_offset(row, column, kChunk)] =
              Element(params.k[k_source]);
        }
        __syncthreads();

        {
          Tensor p = partition_fragment_C(
              tiled_mma, Shape<Int<64>, Int<64>>{});
          Tensor coordinates = thr_mma.partition_C(
              make_identity_tensor(Shape<Int<64>, Int<64>>{}));
          mma_from_tsm<Tile6464128>(
              storage.q_tsm.data(), storage.k_p_tsm.data(),
              thread, warp, p, true);
          __syncthreads();
          for (int index = 0; index < size(p); ++index) {
            auto coordinate = coordinates(index);
            const int row = get<0>(coordinate);
            const int column = get<1>(coordinate);
            storage.ap_tsm[bf16_tsm_offset(row, column, kChunk)] =
                Element(column <= row ? -p(index) : 0.0f);
          }
        }
        __syncthreads();

        {
          Tensor r = partition_fragment_C(
              tiled_mma, Shape<Int<64>, Int<128>>{});
          Tensor coordinates = thr_mma.partition_C(
              make_identity_tensor(Shape<Int<64>, Int<128>>{}));
          for (int index = 0; index < size(r); ++index) {
            auto coordinate = coordinates(index);
            const int row = get<0>(coordinate);
            const int column = get<1>(coordinate);
            const int source = (((batch * params.tokens + token_base + row)
                * params.q_heads + q_head) * kDim) + column;
            r(index) = params.q[source];
          }
          mma_from_tsm<Tile6412864>(
              storage.ap_tsm.data(), storage.x_tsm.data(),
              thread, warp, r, false);
          for (int index = 0; index < size(r); ++index) {
            auto coordinate = coordinates(index);
            const int row = get<0>(coordinate);
            const int column = get<1>(coordinate);
            storage.q_tsm[bf16_tsm_offset(column, row, kDim)] = Element(
                r(index) * params.scale * expf(storage.gate[row]));
          }
        }
        __syncthreads();
        __syncthreads();
      }
    } else {
      for (int chunk = chunks - 1; chunk >= 0; --chunk) {
        const int token_base = chunk * kChunk;
        const int gate_base = (group * chunks + chunk) * kChunk;
        for (int linear = thread; linear < kChunk * kChunk; linear += 128) {
          const int row = linear / kChunk;
          const int column = linear - row * kChunk;
          const int source = ((((batch * params.tokens + token_base + row)
              * params.value_heads + head) * kChunk) + column);
          storage.ap_tsm[bf16_tsm_offset(row, column, kChunk)] = Element(
              params.a[source] * params.beta[
                  (batch * params.tokens + token_base + column)
                  * params.value_heads + head]);
        }
        for (int linear = thread; linear < kChunk * kDim; linear += 128) {
          const int row = linear / kDim;
          const int column = linear - row * kDim;
          const int source = (((batch * params.tokens + token_base + row)
              * params.q_heads + q_head) * kDim) + column;
          storage.k_xy_tsm[bf16_tsm_offset(column, row, kDim)] =
              Element(params.k[source]);
        }
        __syncthreads();

        Tensor x = partition_fragment_C(
            tiled_mma, Shape<Int<64>, Int<128>>{});
        Tensor x_coordinates = thr_mma.partition_C(
            make_identity_tensor(Shape<Int<64>, Int<128>>{}));
        mma_from_tsm<Tile6412864>(
            storage.ap_tsm.data(), storage.k_xy_tsm.data(),
            thread, warp, x, true);
        __syncthreads();
        for (int index = 0; index < size(x); ++index) {
          auto coordinate = x_coordinates(index);
          const int row = get<0>(coordinate);
          const int column = get<1>(coordinate);
          storage.x_tsm[bf16_tsm_offset(column, row, kDim)] =
              Element(x(index));
        }
        __syncthreads();

        Tensor y = partition_fragment_C(
            tiled_mma, Shape<Int<64>, Int<64>>{});
        Tensor y_coordinates = thr_mma.partition_C(
            make_identity_tensor(Shape<Int<64>, Int<64>>{}));
        mma_from_tsm<Tile6464128>(
            storage.k_p_tsm.data(), storage.dh_tsm.data(),
            thread, warp, y, true);
        const float decay = -expf(storage.gate[kChunk - 1]);
        for (int index = 0; index < size(y); ++index) {
          auto coordinate = y_coordinates(index);
          const int row = get<0>(coordinate);
          const int column = get<1>(coordinate);
          storage.k_xy_tsm[bf16_tsm_offset(column, row, kChunk)] =
              Element(decay * y(index));
        }
        __syncthreads();
        __syncthreads();
      }
    }
  }
};

// First K-consumer stage of the official fused backward:
// dV = (scale * Lower(exp(g_i-g_j)) * QK^T)^T @ dO.
// Each CTA owns one chunk/value-head, exposing chunk-level parallelism while
// fusing both AIU MMAs and the causal gate epilogue.
struct FlashQlaChunkDvBackward {
  struct Arguments {
    const float* q;
    const float* k;
    const float* g;
    const float* do_value;
    float* dv;
    int batch_size;
    int tokens;
    int q_heads;
    int value_heads;
    int chunks;
    float scale;
  };
  using Params = Arguments;

  struct SharedStorage {
    cute::array_aligned<Element, 64 * 128> a_tsm;
    cute::array_aligned<Element, 128 * 64> b_tsm;
    cute::array_aligned<float, 64> gate;
  };

  static constexpr int MaxThreadsPerBlock = size(TiledMma{});
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    const int tile = blockIdx.x;
    const int total = params.batch_size * params.value_heads * params.chunks;
    if (tile >= total) {
      return;
    }
    const int chunk = tile % params.chunks;
    const int group = tile / params.chunks;
    const int head = group % params.value_heads;
    const int batch = group / params.value_heads;
    const int q_head = head / (params.value_heads / params.q_heads);
    const int token_base = chunk * kChunk;
    const int gate_base = (group * params.chunks + chunk) * kChunk;
    const int thread = threadIdx.x;
    int warp = cutlass::canonical_warp_idx_sync();
    auto& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread);
    Tensor p = partition_fragment_C(
        tiled_mma, Shape<Int<64>, Int<64>>{});
    Tensor p_coordinates = thr_mma.partition_C(
        make_identity_tensor(Shape<Int<64>, Int<64>>{}));
    Tensor output = partition_fragment_C(
        tiled_mma, Shape<Int<64>, Int<128>>{});
    Tensor output_coordinates = thr_mma.partition_C(
        make_identity_tensor(Shape<Int<64>, Int<128>>{}));

    if (thread < kChunk) {
      storage.gate[thread] = params.g[gate_base + thread];
    }
    for (int linear = thread; linear < kChunk * kDim;
         linear += blockDim.x) {
      const int row = linear / kDim;
      const int column = linear - row * kDim;
      const int q_source = (((batch * params.tokens + token_base + row)
          * params.q_heads + q_head) * kDim) + column;
      storage.a_tsm[bf16_tsm_offset(row, column, kChunk)] =
          Element(params.q[q_source]);
      const int k_source = (((batch * params.tokens + token_base + row)
          * params.q_heads + q_head) * kDim) + column;
      storage.b_tsm[bf16_tsm_offset(row, column, kChunk)] =
          Element(params.k[k_source]);
    }
    __syncthreads();
    mma_from_tsm<Tile6464128>(
        storage.a_tsm.data(), storage.b_tsm.data(),
        thread, warp, p, true);
    __syncthreads();

    // Publish the transpose directly so the second MMA computes Pg^T @ dO.
    for (int index = 0; index < size(p); ++index) {
      auto coordinate = p_coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      const float entry = column <= row
          ? p(index) * params.scale * stable_exp_difference(
                storage.gate[row] - storage.gate[column])
          : 0.0f;
      storage.a_tsm[bf16_tsm_offset(column, row, kChunk)] = Element(entry);
    }
    for (int linear = thread; linear < kChunk * kDim;
         linear += blockDim.x) {
      const int row = linear / kDim;
      const int column = linear - row * kDim;
      const int source = (((batch * params.tokens + token_base + row)
          * params.value_heads + head) * kDim) + column;
      storage.b_tsm[bf16_tsm_offset(column, row, kDim)] =
          Element(params.do_value[source]);
    }
    __syncthreads();
    mma_from_tsm<Tile6412864>(
        storage.a_tsm.data(), storage.b_tsm.data(),
        thread, warp, output, true);
    for (int index = 0; index < size(output); ++index) {
      auto coordinate = output_coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      const int destination = (((batch * params.tokens + token_base + row)
          * params.value_heads + head) * kDim) + column;
      params.dv[destination] = output(index);
    }
  }
};

// Shared H/dH portion of the official A/K consumers. A chunk/head CTA reuses
// one 64x128 input tile and one 128x128 state tile for dQ0, dW and dK0.
struct FlashQlaChunkDqkWBackward {
  struct Arguments {
    const float* do_value;
    const float* v_corrected;
    const float* dv;
    const float* history;
    const float* dh;
    const float* g;
    float* dq;
    float* dk;
    float* dw;
    int batch_size;
    int tokens;
    int value_heads;
    int chunks;
    float scale;
  };
  using Params = Arguments;

  struct SharedStorage {
    cute::array_aligned<Element, 64 * 128> a_tsm;
    cute::array_aligned<Element, 128 * 128> b_tsm;
    cute::array_aligned<float, 64> gate;
  };

  static constexpr int MaxThreadsPerBlock = size(TiledMma{});
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  template <class Source>
  CUTLASS_DEVICE void load_a(
      Source source,
      int source_base,
      int row_stride,
      SharedStorage& storage,
      int thread) {
    for (int linear = thread; linear < kChunk * kDim;
         linear += blockDim.x) {
      const int row = linear / kDim;
      const int column = linear - row * kDim;
      storage.a_tsm[bf16_tsm_offset(row, column, kChunk)] =
          Element(source[source_base + row * row_stride + column]);
    }
  }

  CUTLASS_DEVICE void load_state_b(
      const float* source,
      int source_base,
      SharedStorage& storage,
      int thread) {
    for (int linear = thread; linear < kDim * kDim;
         linear += blockDim.x) {
      const int row = linear / kDim;
      const int column = linear - row * kDim;
      storage.b_tsm[bf16_tsm_offset(row, column, kDim)] =
          Element(source[source_base + row * kDim + column]);
    }
  }

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    const int tile = blockIdx.x;
    const int total = params.batch_size * params.value_heads * params.chunks;
    if (tile >= total) {
      return;
    }
    const int chunk = tile % params.chunks;
    const int group = tile / params.chunks;
    const int head = group % params.value_heads;
    const int batch = group / params.value_heads;
    const int token_base = chunk * kChunk;
    const int vector_base = (((batch * params.tokens + token_base)
        * params.value_heads + head) * kDim);
    const int state_base = (((batch * params.chunks + chunk)
        * params.value_heads + head) * kDim * kDim);
    const int gate_base = (group * params.chunks + chunk) * kChunk;
    const int row_stride = params.value_heads * kDim;
    const int thread = threadIdx.x;
    int warp = cutlass::canonical_warp_idx_sync();
    auto& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread);
    Tensor output = partition_fragment_C(
        tiled_mma, Shape<Int<64>, Int<128>>{});
    Tensor coordinates = thr_mma.partition_C(
        make_identity_tensor(Shape<Int<64>, Int<128>>{}));
    if (thread < kChunk) {
      storage.gate[thread] = params.g[gate_base + thread];
    }

    // dQ0 = scale * exp(g) * (dO @ H^T).
    load_a(params.do_value, vector_base, row_stride, storage, thread);
    load_state_b(params.history, state_base, storage, thread);
    __syncthreads();
    mma_from_tsm<Tile64128128>(
        storage.a_tsm.data(), storage.b_tsm.data(),
        thread, warp, output, true);
    for (int index = 0; index < size(output); ++index) {
      auto coordinate = coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      params.dq[vector_base + row * row_stride + column] = output(index)
          * params.scale * expf(storage.gate[row]);
    }
    __syncthreads();

    // dW = -(dV @ H^T).
    load_a(params.dv, vector_base, row_stride, storage, thread);
    __syncthreads();
    mma_from_tsm<Tile64128128>(
        storage.a_tsm.data(), storage.b_tsm.data(),
        thread, warp, output, true);
    for (int index = 0; index < size(output); ++index) {
      auto coordinate = coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      params.dw[vector_base + row * row_stride + column] = -output(index);
    }
    __syncthreads();

    // dK0 = exp(g_last-g) * (V' @ dH^T).
    load_a(params.v_corrected, vector_base, row_stride, storage, thread);
    load_state_b(params.dh, state_base, storage, thread);
    __syncthreads();
    mma_from_tsm<Tile64128128>(
        storage.a_tsm.data(), storage.b_tsm.data(),
        thread, warp, output, true);
    const float g_last = storage.gate[kChunk - 1];
    for (int index = 0; index < size(output); ++index) {
      auto coordinate = coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      params.dk[vector_base + row * row_stride + column] = output(index)
          * stable_exp_difference(g_last - storage.gate[row]);
    }
  }
};

// Complete per-chunk A/K consumer stage from the official fused backward.
// Besides the direct H/dH terms above, this keeps dPg/dP in BF16 shared
// memory, applies the causal gate, accumulates dQ/dK, and produces dg before
// the reverse chunk cumsum. One CTA owns one chunk/value-head pair.
struct FlashQlaChunkDqkwgBackward {
  struct Arguments {
    const float* q;
    const float* k;
    const float* do_value;
    const float* v_corrected;
    const float* dv;
    const float* history;
    const float* dh;
    const float* g;
    float* dq;
    float* dk;
    float* dw;
    float* dg;
    int batch_size;
    int tokens;
    int q_heads;
    int value_heads;
    int chunks;
    float scale;
  };
  using Params = Arguments;

  struct SharedStorage {
    cute::array_aligned<Element, 64 * 128> a_tsm;
    cute::array_aligned<Element, 128 * 128> b_tsm;
    cute::array_aligned<Element, 64 * 64> dp_tsm;
    cute::array_aligned<float, 64 * 64> temporary_tsm;
    cute::array_aligned<float, 64> gate;
    cute::array_aligned<float, 64> dg;
    cute::array_aligned<float, 64> state_reduction;
    cute::array_aligned<float, 64> k_reduction;
    float dg_last;
  };

  static constexpr int MaxThreadsPerBlock = size(TiledMma{});
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  CUTLASS_DEVICE void load_vector_a(
      const float* source,
      int source_base,
      int row_stride,
      SharedStorage& storage,
      int thread) {
    for (int linear = thread; linear < kChunk * kDim;
         linear += blockDim.x) {
      const int row = linear / kDim;
      const int column = linear - row * kDim;
      storage.a_tsm[bf16_tsm_offset(row, column, kChunk)] =
          Element(source[source_base + row * row_stride + column]);
    }
  }

  CUTLASS_DEVICE void load_state_b(
      const float* source,
      int source_base,
      SharedStorage& storage,
      int thread) {
    for (int linear = thread; linear < kDim * kDim;
         linear += blockDim.x) {
      const int row = linear / kDim;
      const int column = linear - row * kDim;
      storage.b_tsm[bf16_tsm_offset(row, column, kDim)] =
          Element(source[source_base + row * kDim + column]);
    }
  }

  CUTLASS_DEVICE void load_token_matrix_b(
      const float* source,
      int source_base,
      int row_stride,
      SharedStorage& storage,
      int thread) {
    for (int linear = thread; linear < kChunk * kDim;
         linear += blockDim.x) {
      const int row = linear / kDim;
      const int column = linear - row * kDim;
      storage.b_tsm[bf16_tsm_offset(row, column, kChunk)] =
          Element(source[source_base + row * row_stride + column]);
    }
  }

  CUTLASS_DEVICE void load_token_transpose_b(
      const float* source,
      int source_base,
      int row_stride,
      SharedStorage& storage,
      int thread) {
    for (int linear = thread; linear < kChunk * kDim;
         linear += blockDim.x) {
      const int row = linear / kDim;
      const int column = linear - row * kDim;
      storage.b_tsm[bf16_tsm_offset(column, row, kDim)] =
          Element(source[source_base + row * row_stride + column]);
    }
  }

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    const int tile = blockIdx.x;
    const int total = params.batch_size * params.value_heads * params.chunks;
    if (tile >= total) {
      return;
    }
    const int chunk = tile % params.chunks;
    const int group = tile / params.chunks;
    const int head = group % params.value_heads;
    const int batch = group / params.value_heads;
    const int q_head = head / (params.value_heads / params.q_heads);
    const int token_base = chunk * kChunk;
    const int value_base = (((batch * params.tokens + token_base)
        * params.value_heads + head) * kDim);
    const int q_base = (((batch * params.tokens + token_base)
        * params.q_heads + q_head) * kDim);
    const int value_stride = params.value_heads * kDim;
    const int q_stride = params.q_heads * kDim;
    const int state_base = (((batch * params.chunks + chunk)
        * params.value_heads + head) * kDim * kDim);
    const int gate_base = (group * params.chunks + chunk) * kChunk;
    const int thread = threadIdx.x;
    int warp = cutlass::canonical_warp_idx_sync();
    auto& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread);
    Tensor output128 = partition_fragment_C(
        tiled_mma, Shape<Int<64>, Int<128>>{});
    Tensor coordinates128 = thr_mma.partition_C(
        make_identity_tensor(Shape<Int<64>, Int<128>>{}));
    Tensor output64 = partition_fragment_C(
        tiled_mma, Shape<Int<64>, Int<64>>{});
    Tensor coordinates64 = thr_mma.partition_C(
        make_identity_tensor(Shape<Int<64>, Int<64>>{}));

    if (thread < kChunk) {
      storage.gate[thread] = params.g[gate_base + thread];
      storage.dg[thread] = 0.0f;
      storage.state_reduction[thread] = 0.0f;
      storage.k_reduction[thread] = 0.0f;
    }
    if (thread == 0) {
      storage.dg_last = 0.0f;
    }
    __syncthreads();

    // dQ0 = scale * exp(g) * (dO @ H^T).
    load_vector_a(params.do_value, value_base, value_stride, storage, thread);
    load_state_b(params.history, state_base, storage, thread);
    __syncthreads();
    mma_from_tsm<Tile64128128>(
        storage.a_tsm.data(), storage.b_tsm.data(),
        thread, warp, output128, true);
    for (int index = 0; index < size(output128); ++index) {
      auto coordinate = coordinates128(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      params.dq[value_base + row * value_stride + column] = output128(index)
          * params.scale * expf(storage.gate[row]);
    }
    __syncthreads();

    // dW = -(dV @ H^T), reusing H resident in shared memory.
    load_vector_a(params.dv, value_base, value_stride, storage, thread);
    __syncthreads();
    mma_from_tsm<Tile64128128>(
        storage.a_tsm.data(), storage.b_tsm.data(),
        thread, warp, output128, true);
    for (int index = 0; index < size(output128); ++index) {
      auto coordinate = coordinates128(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      params.dw[value_base + row * value_stride + column] = -output128(index);
    }
    __syncthreads();

    // dK0 = exp(g_last-g) * (V' @ dH^T).
    load_vector_a(
        params.v_corrected, value_base, value_stride, storage, thread);
    load_state_b(params.dh, state_base, storage, thread);
    __syncthreads();
    mma_from_tsm<Tile64128128>(
        storage.a_tsm.data(), storage.b_tsm.data(),
        thread, warp, output128, true);
    const float g_last = storage.gate[kChunk - 1];
    for (int index = 0; index < size(output128); ++index) {
      auto coordinate = coordinates128(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      params.dk[value_base + row * value_stride + column] = output128(index)
          * stable_exp_difference(g_last - storage.gate[row]);
    }
    __syncthreads();

    // Direct gate derivatives and the state-boundary contribution. These are
    // evaluated before dP contributions are added to dQ/dK, matching official
    // CONSUMER_K/CONSUMER_S ordering.
    if (thread < kChunk) {
      float q_dq = 0.0f;
      float k_dk = 0.0f;
      for (int column = 0; column < kDim; ++column) {
        q_dq += params.q[q_base + thread * q_stride + column]
            * params.dq[value_base + thread * value_stride + column];
        k_dk += params.k[q_base + thread * q_stride + column]
            * params.dk[value_base + thread * value_stride + column];
      }
      storage.dg[thread] = q_dq - k_dk;
      storage.k_reduction[thread] = k_dk;
      float state_sum = 0.0f;
      for (int linear = thread; linear < kDim * kDim; linear += kChunk) {
        state_sum += params.history[state_base + linear]
            * params.dh[state_base + linear];
      }
      storage.state_reduction[thread] = state_sum;
    }
    __syncthreads();
    if (thread == 0) {
      float total = 0.0f;
      for (int row = 0; row < kChunk; ++row) {
        total += storage.k_reduction[row];
        total += expf(g_last) * storage.state_reduction[row];
      }
      storage.dg_last = total;
    }
    __syncthreads();

    // dPg = dO @ V'^T. Preserve the official FP32-accumulate/BF16-shared
    // handoff, and materialize gated dP for its two following MMAs.
    load_vector_a(params.do_value, value_base, value_stride, storage, thread);
    load_token_matrix_b(
        params.v_corrected, value_base, value_stride, storage, thread);
    __syncthreads();
    mma_from_tsm<Tile6464128>(
        storage.a_tsm.data(), storage.b_tsm.data(),
        thread, warp, output64, true);
    for (int index = 0; index < size(output64); ++index) {
      auto coordinate = coordinates64(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      const int offset = bf16_tsm_offset(row, column, kChunk);
      storage.temporary_tsm[offset] = output64(index);
      const float decay = column <= row
          ? stable_exp_difference(storage.gate[row] - storage.gate[column])
          : 0.0f;
      storage.dp_tsm[offset] = Element(output64(index) * params.scale * decay);
    }
    __syncthreads();

    // P = Q @ K^T, then dg += row_sum(dPg*Pg)-col_sum(dPg*Pg).
    load_vector_a(params.q, q_base, q_stride, storage, thread);
    load_token_matrix_b(params.k, q_base, q_stride, storage, thread);
    __syncthreads();
    mma_from_tsm<Tile6464128>(
        storage.a_tsm.data(), storage.b_tsm.data(),
        thread, warp, output64, true);
    for (int index = 0; index < size(output64); ++index) {
      auto coordinate = coordinates64(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      const int offset = bf16_tsm_offset(row, column, kChunk);
      const float decay = column <= row
          ? stable_exp_difference(storage.gate[row] - storage.gate[column])
          : 0.0f;
      const Element pg = Element(output64(index) * params.scale * decay);
      storage.temporary_tsm[offset] *= float(pg);
    }
    __syncthreads();
    if (thread < kChunk) {
      float row_sum = 0.0f;
      float column_sum = 0.0f;
      for (int column = 0; column < kChunk; ++column) {
        row_sum += storage.temporary_tsm[
            bf16_tsm_offset(thread, column, kChunk)];
        column_sum += storage.temporary_tsm[
            bf16_tsm_offset(column, thread, kChunk)];
      }
      storage.dg[thread] += row_sum - column_sum;
    }
    __syncthreads();

    // dQ += dP @ K.
    for (int linear = thread; linear < kChunk * kChunk;
         linear += blockDim.x) {
      const int row = linear / kChunk;
      const int column = linear - row * kChunk;
      storage.a_tsm[bf16_tsm_offset(row, column, kChunk)] =
          storage.dp_tsm[bf16_tsm_offset(row, column, kChunk)];
    }
    load_token_transpose_b(params.k, q_base, q_stride, storage, thread);
    __syncthreads();
    mma_from_tsm<Tile6412864>(
        storage.a_tsm.data(), storage.b_tsm.data(),
        thread, warp, output128, true);
    for (int index = 0; index < size(output128); ++index) {
      auto coordinate = coordinates128(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      params.dq[value_base + row * value_stride + column] += output128(index);
    }
    __syncthreads();

    // dK += dP^T @ Q.
    for (int linear = thread; linear < kChunk * kChunk;
         linear += blockDim.x) {
      const int row = linear / kChunk;
      const int column = linear - row * kChunk;
      storage.a_tsm[bf16_tsm_offset(row, column, kChunk)] =
          storage.dp_tsm[bf16_tsm_offset(column, row, kChunk)];
    }
    load_token_transpose_b(params.q, q_base, q_stride, storage, thread);
    __syncthreads();
    mma_from_tsm<Tile6412864>(
        storage.a_tsm.data(), storage.b_tsm.data(),
        thread, warp, output128, true);
    for (int index = 0; index < size(output128); ++index) {
      auto coordinate = coordinates128(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      params.dk[value_base + row * value_stride + column] += output128(index);
    }
    __syncthreads();

    if (thread < kChunk) {
      const int destination = ((batch * params.tokens + token_base + thread)
          * params.value_heads + head);
      params.dg[destination] = storage.dg[thread]
          + (thread == kChunk - 1 ? storage.dg_last : 0.0f);
    }
  }
};

// Two-consumer form of the complete chunk stage. Consumer K owns the three
// H/dH MMAs while consumer A concurrently owns dPg/P and the two dP MMAs.
// This mirrors the central concurrency invariant in the official 512-thread
// kernel while leaving the cross-chunk S recurrence in its validated stage.
struct FlashQlaChunkDqkwgBackwardWarpSpecialized {
  using Arguments = FlashQlaChunkDqkwgBackward::Arguments;
  using Params = Arguments;

  struct SharedStorage {
    cute::array_aligned<Element, 64 * 128> direct_a;
    cute::array_aligned<Element, 128 * 128> direct_b;
    cute::array_aligned<Element, 64 * 128> dp_a;
    cute::array_aligned<Element, 128 * 128> dp_b;
    cute::array_aligned<Element, 64 * 64> dp;
    cute::array_aligned<float, 64 * 64> gate_product;
    cute::array_aligned<float, 64> gate;
    cute::array_aligned<float, 64> dg;
    cute::array_aligned<float, 64> state_reduction;
    cute::array_aligned<float, 64> k_reduction;
    float dg_last;
  };

  static constexpr int MaxThreadsPerBlock = 256;
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    const int tile = blockIdx.x;
    const int total = params.batch_size * params.value_heads * params.chunks;
    if (tile >= total) {
      return;
    }
    const int chunk = tile % params.chunks;
    const int group = tile / params.chunks;
    const int head = group % params.value_heads;
    const int batch = group / params.value_heads;
    const int q_head = head / (params.value_heads / params.q_heads);
    const int token_base = chunk * kChunk;
    const int value_base = (((batch * params.tokens + token_base)
        * params.value_heads + head) * kDim);
    const int q_base = (((batch * params.tokens + token_base)
        * params.q_heads + q_head) * kDim);
    const int value_stride = params.value_heads * kDim;
    const int q_stride = params.q_heads * kDim;
    const int state_base = (((batch * params.chunks + chunk)
        * params.value_heads + head) * kDim * kDim);
    const int gate_base = (group * params.chunks + chunk) * kChunk;
    const int consumer = threadIdx.x / 128;
    const int thread = threadIdx.x - consumer * 128;
    const int warp = thread / 32;
    auto& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    if (threadIdx.x < kChunk) {
      storage.gate[threadIdx.x] = params.g[gate_base + threadIdx.x];
      storage.dg[threadIdx.x] = 0.0f;
      storage.state_reduction[threadIdx.x] = 0.0f;
      storage.k_reduction[threadIdx.x] = 0.0f;
    }
    if (threadIdx.x == 0) {
      storage.dg_last = 0.0f;
    }
    __syncthreads();

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread);
    if (consumer == 0) {
      Tensor output = partition_fragment_C(
          tiled_mma, Shape<Int<64>, Int<128>>{});
      Tensor coordinates = thr_mma.partition_C(
          make_identity_tensor(Shape<Int<64>, Int<128>>{}));

      // Stage 1: dQ0, concurrent with dPg.
      for (int linear = thread; linear < kChunk * kDim; linear += 128) {
        const int row = linear / kDim;
        const int column = linear - row * kDim;
        storage.direct_a[bf16_tsm_offset(row, column, kChunk)] = Element(
            params.do_value[value_base + row * value_stride + column]);
      }
      for (int linear = thread; linear < kDim * kDim; linear += 128) {
        const int row = linear / kDim;
        const int column = linear - row * kDim;
        storage.direct_b[bf16_tsm_offset(row, column, kDim)] =
            Element(params.history[state_base + row * kDim + column]);
      }
      __syncthreads();
      mma_from_tsm<Tile64128128>(
          storage.direct_a.data(), storage.direct_b.data(),
          thread, warp, output, true);
      for (int index = 0; index < size(output); ++index) {
        auto coordinate = coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        params.dq[value_base + row * value_stride + column] = output(index)
            * params.scale * expf(storage.gate[row]);
      }
      __syncthreads();

      // Stage 2: dW, concurrent with P.
      for (int linear = thread; linear < kChunk * kDim; linear += 128) {
        const int row = linear / kDim;
        const int column = linear - row * kDim;
        storage.direct_a[bf16_tsm_offset(row, column, kChunk)] = Element(
            params.dv[value_base + row * value_stride + column]);
      }
      __syncthreads();
      mma_from_tsm<Tile64128128>(
          storage.direct_a.data(), storage.direct_b.data(),
          thread, warp, output, true);
      for (int index = 0; index < size(output); ++index) {
        auto coordinate = coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        params.dw[value_base + row * value_stride + column] = -output(index);
      }
      __syncthreads();

      // Stage 3: dK0 while A finishes publishing dP/gate products.
      for (int linear = thread; linear < kChunk * kDim; linear += 128) {
        const int row = linear / kDim;
        const int column = linear - row * kDim;
        storage.direct_a[bf16_tsm_offset(row, column, kChunk)] = Element(
            params.v_corrected[value_base + row * value_stride + column]);
      }
      for (int linear = thread; linear < kDim * kDim; linear += 128) {
        const int row = linear / kDim;
        const int column = linear - row * kDim;
        storage.direct_b[bf16_tsm_offset(row, column, kDim)] =
            Element(params.dh[state_base + row * kDim + column]);
      }
      __syncthreads();
      mma_from_tsm<Tile64128128>(
          storage.direct_a.data(), storage.direct_b.data(),
          thread, warp, output, true);
      const float g_last = storage.gate[kChunk - 1];
      for (int index = 0; index < size(output); ++index) {
        auto coordinate = coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        params.dk[value_base + row * value_stride + column] = output(index)
            * stable_exp_difference(g_last - storage.gate[row]);
      }
      __syncthreads();

      // The A consumer has now published dPg*Pg. Finish all gate reductions
      // before it is allowed to add dP contributions into dQ/dK.
      if (thread < kChunk) {
        float q_dq = 0.0f;
        float k_dk = 0.0f;
        for (int column = 0; column < kDim; ++column) {
          q_dq += params.q[q_base + thread * q_stride + column]
              * params.dq[value_base + thread * value_stride + column];
          k_dk += params.k[q_base + thread * q_stride + column]
              * params.dk[value_base + thread * value_stride + column];
        }
        float row_sum = 0.0f;
        float column_sum = 0.0f;
        for (int column = 0; column < kChunk; ++column) {
          row_sum += storage.gate_product[
              bf16_tsm_offset(thread, column, kChunk)];
          column_sum += storage.gate_product[
              bf16_tsm_offset(column, thread, kChunk)];
        }
        storage.dg[thread] = q_dq - k_dk + row_sum - column_sum;
        storage.k_reduction[thread] = k_dk;
        float state_sum = 0.0f;
        for (int linear = thread; linear < kDim * kDim; linear += kChunk) {
          state_sum += params.history[state_base + linear]
              * params.dh[state_base + linear];
        }
        storage.state_reduction[thread] = state_sum;
      }
      __syncthreads();
      if (thread == 0) {
        float total_dg_last = 0.0f;
        for (int row = 0; row < kChunk; ++row) {
          total_dg_last += storage.k_reduction[row]
              + expf(g_last) * storage.state_reduction[row];
        }
        storage.dg_last = total_dg_last;
      }
      __syncthreads();

      // Match the two dP MMA publication barriers.
      __syncthreads();
      __syncthreads();
      __syncthreads();
      if (thread < kChunk) {
        const int destination = ((batch * params.tokens + token_base + thread)
            * params.value_heads + head);
        params.dg[destination] = storage.dg[thread]
            + (thread == kChunk - 1 ? storage.dg_last : 0.0f);
      }
    } else {
      // Scope the 64x64 accumulator to the two matrix stages, allowing the
      // compiler to reuse its registers for the later 64x128 accumulator.
      {
        Tensor output = partition_fragment_C(
            tiled_mma, Shape<Int<64>, Int<64>>{});
        Tensor coordinates = thr_mma.partition_C(
            make_identity_tensor(Shape<Int<64>, Int<64>>{}));

        // Stage 1: dPg = dO @ V'^T.
        for (int linear = thread; linear < kChunk * kDim; linear += 128) {
          const int row = linear / kDim;
          const int column = linear - row * kDim;
          storage.dp_a[bf16_tsm_offset(row, column, kChunk)] = Element(
              params.do_value[value_base + row * value_stride + column]);
          storage.dp_b[bf16_tsm_offset(row, column, kChunk)] = Element(
              params.v_corrected[value_base + row * value_stride + column]);
        }
        __syncthreads();
        mma_from_tsm<Tile6464128>(
            storage.dp_a.data(), storage.dp_b.data(),
            thread, warp, output, true);
        for (int index = 0; index < size(output); ++index) {
          auto coordinate = coordinates(index);
          const int row = get<0>(coordinate);
          const int column = get<1>(coordinate);
          const int offset = bf16_tsm_offset(row, column, kChunk);
          storage.gate_product[offset] = output(index);
          const float decay = column <= row
              ? stable_exp_difference(
                    storage.gate[row] - storage.gate[column])
              : 0.0f;
          storage.dp[offset] = Element(output(index) * params.scale * decay);
        }
        __syncthreads();

        // Stage 2: P = Q @ K^T and publish dPg*Pg for dg.
        for (int linear = thread; linear < kChunk * kDim; linear += 128) {
          const int row = linear / kDim;
          const int column = linear - row * kDim;
          storage.dp_a[bf16_tsm_offset(row, column, kChunk)] = Element(
              params.q[q_base + row * q_stride + column]);
          storage.dp_b[bf16_tsm_offset(row, column, kChunk)] = Element(
              params.k[q_base + row * q_stride + column]);
        }
        __syncthreads();
        mma_from_tsm<Tile6464128>(
            storage.dp_a.data(), storage.dp_b.data(),
            thread, warp, output, true);
        for (int index = 0; index < size(output); ++index) {
          auto coordinate = coordinates(index);
          const int row = get<0>(coordinate);
          const int column = get<1>(coordinate);
          const int offset = bf16_tsm_offset(row, column, kChunk);
          const float decay = column <= row
              ? stable_exp_difference(
                    storage.gate[row] - storage.gate[column])
              : 0.0f;
          const Element pg = Element(output(index) * params.scale * decay);
          storage.gate_product[offset] *= float(pg);
        }
        __syncthreads();
        // Match K's dK0 load/MMA publication stage.
        __syncthreads();
        __syncthreads();
      }

      // Wait until K has consumed the direct dQ/dK values for dg.
      __syncthreads();
      for (int linear = thread; linear < kChunk * kChunk; linear += 128) {
        const int row = linear / kChunk;
        const int column = linear - row * kChunk;
        storage.dp_a[bf16_tsm_offset(row, column, kChunk)] =
            storage.dp[bf16_tsm_offset(row, column, kChunk)];
      }
      for (int linear = thread; linear < kChunk * kDim; linear += 128) {
        const int row = linear / kDim;
        const int column = linear - row * kDim;
        storage.dp_b[bf16_tsm_offset(column, row, kDim)] = Element(
            params.k[q_base + row * q_stride + column]);
      }
      __syncthreads();

      Tensor output = partition_fragment_C(
          tiled_mma, Shape<Int<64>, Int<128>>{});
      Tensor coordinates = thr_mma.partition_C(
          make_identity_tensor(Shape<Int<64>, Int<128>>{}));
      mma_from_tsm<Tile6412864>(
          storage.dp_a.data(), storage.dp_b.data(),
          thread, warp, output, true);
      for (int index = 0; index < size(output); ++index) {
        auto coordinate = coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        params.dq[value_base + row * value_stride + column] += output(index);
      }
      __syncthreads();

      for (int linear = thread; linear < kChunk * kChunk; linear += 128) {
        const int row = linear / kChunk;
        const int column = linear - row * kChunk;
        storage.dp_a[bf16_tsm_offset(row, column, kChunk)] =
            storage.dp[bf16_tsm_offset(column, row, kChunk)];
      }
      for (int linear = thread; linear < kChunk * kDim; linear += 128) {
        const int row = linear / kDim;
        const int column = linear - row * kDim;
        storage.dp_b[bf16_tsm_offset(column, row, kDim)] = Element(
            params.q[q_base + row * q_stride + column]);
      }
      __syncthreads();
      mma_from_tsm<Tile6412864>(
          storage.dp_a.data(), storage.dp_b.data(),
          thread, warp, output, true);
      for (int index = 0; index < size(output); ++index) {
        auto coordinate = coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        params.dk[value_base + row * value_stride + column] += output(index);
      }
      __syncthreads();
    }
  }
};

// Keep the K and A consumers in separate device call frames. HGCC otherwise
// merges both divergent paths into one 256-register frame and spills roughly
// 1 KiB per thread even though each 128-thread consumer only needs one live
// accumulator class.
struct FlashQlaChunkDqkwgBackwardNoInline {
  using Base = FlashQlaChunkDqkwgBackwardWarpSpecialized;
  using Arguments = Base::Arguments;
  using Params = Arguments;
  using SharedStorage = Base::SharedStorage;

  static constexpr int MaxThreadsPerBlock = 256;
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  static __attribute__((__noinline__)) __device__ void run_k(
      Params const& params, SharedStorage& storage) {
    const int tile = blockIdx.x;
    const int chunk = tile % params.chunks;
    const int group = tile / params.chunks;
    const int head = group % params.value_heads;
    const int batch = group / params.value_heads;
    const int q_head = head / (params.value_heads / params.q_heads);
    const int token_base = chunk * kChunk;
    const int value_base = (((batch * params.tokens + token_base)
        * params.value_heads + head) * kDim);
    const int q_base = (((batch * params.tokens + token_base)
        * params.q_heads + q_head) * kDim);
    const int value_stride = params.value_heads * kDim;
    const int q_stride = params.q_heads * kDim;
    const int state_base = (((batch * params.chunks + chunk)
        * params.value_heads + head) * kDim * kDim);
    const int thread = threadIdx.x;
    const int warp = thread / 32;
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread);
    Tensor output = partition_fragment_C(
        tiled_mma, Shape<Int<64>, Int<128>>{});
    Tensor coordinates = thr_mma.partition_C(
        make_identity_tensor(Shape<Int<64>, Int<128>>{}));

    for (int linear = thread; linear < kChunk * kDim; linear += 128) {
      const int row = linear / kDim;
      const int column = linear - row * kDim;
      storage.direct_a[bf16_tsm_offset(row, column, kChunk)] = Element(
          params.do_value[value_base + row * value_stride + column]);
    }
    for (int linear = thread; linear < kDim * kDim; linear += 128) {
      const int row = linear / kDim;
      const int column = linear - row * kDim;
      storage.direct_b[bf16_tsm_offset(row, column, kDim)] =
          Element(params.history[state_base + row * kDim + column]);
    }
    __syncthreads();
    mma_from_tsm<Tile64128128>(
        storage.direct_a.data(), storage.direct_b.data(),
        thread, warp, output, true);
    for (int index = 0; index < size(output); ++index) {
      auto coordinate = coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      params.dq[value_base + row * value_stride + column] = output(index)
          * params.scale * expf(storage.gate[row]);
    }
    __syncthreads();

    for (int linear = thread; linear < kChunk * kDim; linear += 128) {
      const int row = linear / kDim;
      const int column = linear - row * kDim;
      storage.direct_a[bf16_tsm_offset(row, column, kChunk)] = Element(
          params.dv[value_base + row * value_stride + column]);
    }
    __syncthreads();
    mma_from_tsm<Tile64128128>(
        storage.direct_a.data(), storage.direct_b.data(),
        thread, warp, output, true);
    for (int index = 0; index < size(output); ++index) {
      auto coordinate = coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      params.dw[value_base + row * value_stride + column] = -output(index);
    }
    __syncthreads();

    for (int linear = thread; linear < kChunk * kDim; linear += 128) {
      const int row = linear / kDim;
      const int column = linear - row * kDim;
      storage.direct_a[bf16_tsm_offset(row, column, kChunk)] = Element(
          params.v_corrected[value_base + row * value_stride + column]);
    }
    for (int linear = thread; linear < kDim * kDim; linear += 128) {
      const int row = linear / kDim;
      const int column = linear - row * kDim;
      storage.direct_b[bf16_tsm_offset(row, column, kDim)] =
          Element(params.dh[state_base + row * kDim + column]);
    }
    __syncthreads();
    mma_from_tsm<Tile64128128>(
        storage.direct_a.data(), storage.direct_b.data(),
        thread, warp, output, true);
    const float g_last = storage.gate[kChunk - 1];
    for (int index = 0; index < size(output); ++index) {
      auto coordinate = coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      params.dk[value_base + row * value_stride + column] = output(index)
          * stable_exp_difference(g_last - storage.gate[row]);
    }
    __syncthreads();

    if (thread < kChunk) {
      float q_dq = 0.0f;
      float k_dk = 0.0f;
      for (int column = 0; column < kDim; ++column) {
        q_dq += params.q[q_base + thread * q_stride + column]
            * params.dq[value_base + thread * value_stride + column];
        k_dk += params.k[q_base + thread * q_stride + column]
            * params.dk[value_base + thread * value_stride + column];
      }
      float row_sum = 0.0f;
      float column_sum = 0.0f;
      for (int column = 0; column < kChunk; ++column) {
        row_sum += storage.gate_product[
            bf16_tsm_offset(thread, column, kChunk)];
        column_sum += storage.gate_product[
            bf16_tsm_offset(column, thread, kChunk)];
      }
      storage.dg[thread] = q_dq - k_dk + row_sum - column_sum;
      storage.k_reduction[thread] = k_dk;
      float state_sum = 0.0f;
      for (int linear = thread; linear < kDim * kDim; linear += kChunk) {
        state_sum += params.history[state_base + linear]
            * params.dh[state_base + linear];
      }
      storage.state_reduction[thread] = state_sum;
    }
    __syncthreads();
    if (thread == 0) {
      float total_dg_last = 0.0f;
      for (int row = 0; row < kChunk; ++row) {
        total_dg_last += storage.k_reduction[row]
            + expf(g_last) * storage.state_reduction[row];
      }
      storage.dg_last = total_dg_last;
    }
    __syncthreads();
    __syncthreads();
    __syncthreads();
    __syncthreads();
    if (thread < kChunk) {
      const int destination = ((batch * params.tokens + token_base + thread)
          * params.value_heads + head);
      params.dg[destination] = storage.dg[thread]
          + (thread == kChunk - 1 ? storage.dg_last : 0.0f);
    }
  }

  static __attribute__((__noinline__)) __device__ void run_a(
      Params const& params, SharedStorage& storage) {
    const int tile = blockIdx.x;
    const int chunk = tile % params.chunks;
    const int group = tile / params.chunks;
    const int head = group % params.value_heads;
    const int batch = group / params.value_heads;
    const int q_head = head / (params.value_heads / params.q_heads);
    const int token_base = chunk * kChunk;
    const int value_base = (((batch * params.tokens + token_base)
        * params.value_heads + head) * kDim);
    const int q_base = (((batch * params.tokens + token_base)
        * params.q_heads + q_head) * kDim);
    const int value_stride = params.value_heads * kDim;
    const int q_stride = params.q_heads * kDim;
    const int thread = threadIdx.x - 128;
    const int warp = thread / 32;
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread);

    {
      Tensor output = partition_fragment_C(
          tiled_mma, Shape<Int<64>, Int<64>>{});
      Tensor coordinates = thr_mma.partition_C(
          make_identity_tensor(Shape<Int<64>, Int<64>>{}));
      for (int linear = thread; linear < kChunk * kDim; linear += 128) {
        const int row = linear / kDim;
        const int column = linear - row * kDim;
        storage.dp_a[bf16_tsm_offset(row, column, kChunk)] = Element(
            params.do_value[value_base + row * value_stride + column]);
        storage.dp_b[bf16_tsm_offset(row, column, kChunk)] = Element(
            params.v_corrected[value_base + row * value_stride + column]);
      }
      __syncthreads();
      mma_from_tsm<Tile6464128>(
          storage.dp_a.data(), storage.dp_b.data(),
          thread, warp, output, true);
      for (int index = 0; index < size(output); ++index) {
        auto coordinate = coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        const int offset = bf16_tsm_offset(row, column, kChunk);
        storage.gate_product[offset] = output(index);
        const float decay = column <= row
            ? stable_exp_difference(storage.gate[row] - storage.gate[column])
            : 0.0f;
        storage.dp[offset] = Element(output(index) * params.scale * decay);
      }
      __syncthreads();

      for (int linear = thread; linear < kChunk * kDim; linear += 128) {
        const int row = linear / kDim;
        const int column = linear - row * kDim;
        storage.dp_a[bf16_tsm_offset(row, column, kChunk)] = Element(
            params.q[q_base + row * q_stride + column]);
        storage.dp_b[bf16_tsm_offset(row, column, kChunk)] = Element(
            params.k[q_base + row * q_stride + column]);
      }
      __syncthreads();
      mma_from_tsm<Tile6464128>(
          storage.dp_a.data(), storage.dp_b.data(),
          thread, warp, output, true);
      for (int index = 0; index < size(output); ++index) {
        auto coordinate = coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        const int offset = bf16_tsm_offset(row, column, kChunk);
        const float decay = column <= row
            ? stable_exp_difference(storage.gate[row] - storage.gate[column])
            : 0.0f;
        const Element pg = Element(output(index) * params.scale * decay);
        storage.gate_product[offset] *= float(pg);
      }
      __syncthreads();
      __syncthreads();
      __syncthreads();
    }

    __syncthreads();
    for (int linear = thread; linear < kChunk * kChunk; linear += 128) {
      const int row = linear / kChunk;
      const int column = linear - row * kChunk;
      storage.dp_a[bf16_tsm_offset(row, column, kChunk)] =
          storage.dp[bf16_tsm_offset(row, column, kChunk)];
    }
    for (int linear = thread; linear < kChunk * kDim; linear += 128) {
      const int row = linear / kDim;
      const int column = linear - row * kDim;
      storage.dp_b[bf16_tsm_offset(column, row, kDim)] = Element(
          params.k[q_base + row * q_stride + column]);
    }
    __syncthreads();

    Tensor output = partition_fragment_C(
        tiled_mma, Shape<Int<64>, Int<128>>{});
    Tensor coordinates = thr_mma.partition_C(
        make_identity_tensor(Shape<Int<64>, Int<128>>{}));
    mma_from_tsm<Tile6412864>(
        storage.dp_a.data(), storage.dp_b.data(),
        thread, warp, output, true);
    for (int index = 0; index < size(output); ++index) {
      auto coordinate = coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      params.dq[value_base + row * value_stride + column] += output(index);
    }
    __syncthreads();

    for (int linear = thread; linear < kChunk * kChunk; linear += 128) {
      const int row = linear / kChunk;
      const int column = linear - row * kChunk;
      storage.dp_a[bf16_tsm_offset(row, column, kChunk)] =
          storage.dp[bf16_tsm_offset(column, row, kChunk)];
    }
    for (int linear = thread; linear < kChunk * kDim; linear += 128) {
      const int row = linear / kDim;
      const int column = linear - row * kDim;
      storage.dp_b[bf16_tsm_offset(column, row, kDim)] = Element(
          params.q[q_base + row * q_stride + column]);
    }
    __syncthreads();
    mma_from_tsm<Tile6412864>(
        storage.dp_a.data(), storage.dp_b.data(),
        thread, warp, output, true);
    for (int index = 0; index < size(output); ++index) {
      auto coordinate = coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      params.dk[value_base + row * value_stride + column] += output(index);
    }
    __syncthreads();
  }

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    const int tile = blockIdx.x;
    const int total = params.batch_size * params.value_heads * params.chunks;
    if (tile >= total) {
      return;
    }
    const int chunk = tile % params.chunks;
    const int group = tile / params.chunks;
    const int gate_base = (group * params.chunks + chunk) * kChunk;
    auto& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    if (threadIdx.x < kChunk) {
      storage.gate[threadIdx.x] = params.g[gate_base + threadIdx.x];
      storage.dg[threadIdx.x] = 0.0f;
      storage.state_reduction[threadIdx.x] = 0.0f;
      storage.k_reduction[threadIdx.x] = 0.0f;
    }
    if (threadIdx.x == 0) {
      storage.dg_last = 0.0f;
    }
    __syncthreads();
    if (threadIdx.x < 128) {
      run_k(params, storage);
    } else {
      run_a(params, storage);
    }
  }
};

#include "ppu_flash_qla_fused_bwd.inc"

// Official AutoCP affine recurrence. Each CTA owns a 128x64 state tile and
// advances all active warmup chunks without materializing U/Y/state deltas in
// global memory. matrix_mode uses an identity initial state and omits V,
// yielding the segment propagation matrix M with the same recurrence.
struct FlashQlaAffineState {
  struct Arguments {
    const Element* k;
    const Element* v;
    const Element* x;
    const float* gamma_last;
    const float* reverse_decay;
    const int* warmup_chunks;
    const uint8_t* fallback;
    float* output;
    int groups;
    int chunks;
    bool matrix_mode;
  };
  using Params = Arguments;

  struct SharedStorage {
    cute::array_aligned<Element, 128 * 64> a_tsm;
    cute::array_aligned<Element, 128 * 64> b_tsm;
  };

  static constexpr int MaxThreadsPerBlock = size(TiledMma{});
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    const int tile_group = blockIdx.x;
    if (tile_group >= params.groups * 2) {
      return;
    }
    const int group = tile_group / 2;
    const int value_tile = tile_group - group * 2;
    const int value_start = value_tile * 64;
    const int thread = threadIdx.x;
    int warp = cutlass::canonical_warp_idx_sync();
    auto& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Element* a_tsm = storage.a_tsm.data();
    Element* b_tsm = storage.b_tsm.data();

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread);
    Tensor state = partition_fragment_C(
        tiled_mma, Shape<Int<128>, Int<64>>{});
    Tensor state_coordinates = thr_mma.partition_C(
        make_identity_tensor(Shape<Int<128>, Int<64>>{}));
    Tensor work = partition_fragment_C(
        tiled_mma, Shape<Int<64>, Int<64>>{});
    Tensor work_coordinates = thr_mma.partition_C(
        make_identity_tensor(Shape<Int<64>, Int<64>>{}));

    const bool active_group =
        !params.matrix_mode || params.fallback[group] != 0;
    for (int index = 0; index < size(state); ++index) {
      auto coordinate = state_coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      state(index) = params.matrix_mode && active_group
          && row == value_start + column ? 1.0f : 0.0f;
    }

    const int active_chunks = params.matrix_mode
        ? (active_group ? params.chunks : 0)
        : params.warmup_chunks[group];
    const int first_chunk = params.chunks - active_chunks;
    for (int chunk = first_chunk; chunk < params.chunks; ++chunk) {
      const int matrix_base =
          (group * params.chunks + chunk) * kChunk * kDim;
      copy_global_a_to_tsm<Tile6464128>(
          params.k + matrix_base, kDim, a_tsm, thread, warp);
      for (int index = 0; index < size(state); ++index) {
        auto coordinate = state_coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        b_tsm[bf16_tsm_offset(column, row, kChunk)] =
            Element(state(index));
      }
      __syncthreads();
      mma_from_tsm<Tile6464128>(
          a_tsm, b_tsm, thread, warp, work, true);
      __syncthreads();

      const float gamma =
          params.gamma_last[group * params.chunks + chunk];
      const int reverse_base =
          (group * params.chunks + chunk) * kChunk;
      for (int index = 0; index < size(work); ++index) {
        auto coordinate = work_coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        float value = gamma * work(index);
        if (!params.matrix_mode) {
          value -= params.reverse_decay[reverse_base + row]
              * float(params.v[
                  matrix_base + row * kDim + value_start + column]);
        }
        b_tsm[bf16_tsm_offset(column, row, kChunk)] = Element(value);
      }
      for (int index = 0; index < size(state); ++index) {
        state(index) *= gamma;
      }
      for (int linear = thread; linear < kDim * kChunk;
           linear += blockDim.x) {
        const int row = linear / kChunk;
        const int column = linear - row * kChunk;
        a_tsm[bf16_tsm_offset(row, column, kDim)] =
            params.x[matrix_base + column * kDim + row];
      }
      __syncthreads();
      mma_from_tsm<Tile1286464>(
          a_tsm, b_tsm, thread, warp, state, false);
      __syncthreads();
    }

    const int output_base = group * kDim * kDim;
    for (int index = 0; index < size(state); ++index) {
      auto coordinate = state_coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      params.output[
          output_base + row * kDim + value_start + column] = state(index);
    }
  }
};

// PPU specialization of the official producer + S/V/O consumer schedule.
// Each 128-thread consumer owns only one live accumulator class, avoiding the
// 256-register cap and 800-byte/thread spill of the single-consumer prototype.
struct FlashQlaFusedForwardWarpSpecialized {
  using Arguments = FlashQlaFusedForward::Arguments;
  using Params = Arguments;

  struct SharedStorage {
    cute::array_aligned<Element, 64 * 128> q_a;
    cute::array_aligned<Element, 64 * 128> k_a;
    cute::array_aligned<Element, 128 * 128> k_b;
    cute::array_aligned<Element, 128 * 128> state_b;
    cute::array_aligned<Element, 64 * 64> ag_a;
    cute::array_aligned<Element, 64 * 64> pg_a;
    cute::array_aligned<Element, 128 * 64> value_b;
    cute::array_aligned<Element, 128 * 64> vd_b;
  };

  static constexpr int MaxThreadsPerBlock = 384;
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    const int group = blockIdx.x;
    const int groups = params.batch_size * params.value_heads;
    if (group >= groups) {
      return;
    }
    const int consumer = threadIdx.x / 128;
    const int thread = threadIdx.x - consumer * 128;
    const int warp = thread / 32;
    const int batch = group / params.value_heads;
    const int head = group - batch * params.value_heads;
    const int q_head = head / (params.value_heads / params.q_heads);
    const int chunks = params.tokens / kChunk;
    const int state_base = group * kDim * kDim;
    auto& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread);

    if (consumer == 0) {
      Tensor state = partition_fragment_C(
          tiled_mma, Shape<Int<128>, Int<128>>{});
      Tensor coordinates = thr_mma.partition_C(
          make_identity_tensor(Shape<Int<128>, Int<128>>{}));
      for (int index = 0; index < size(state); ++index) {
        auto coordinate = coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        state(index) = params.use_initial_state
            ? params.initial_state[state_base + row * kDim + column]
            : 0.0f;
      }

      for (int chunk = 0; chunk < chunks; ++chunk) {
        const int token_base = chunk * kChunk;
        const int gate_base = (group * chunks + chunk) * kChunk;

        // Stage A: publish the current FP32 state at official BF16 handoff.
        for (int index = 0; index < size(state); ++index) {
          auto coordinate = coordinates(index);
          const int row = get<0>(coordinate);
          const int column = get<1>(coordinate);
          storage.state_b[bf16_tsm_offset(column, row, kDim)] =
              Element(state(index));
        }
        __syncthreads();

        // Stage B: V computes K@S while O computes Q@K^T.
        __syncthreads();

        // Stage C: decay state and prepare K^T for the final state update.
        const float g_last = params.g[gate_base + kChunk - 1];
        const float gamma_last = expf(g_last);
        for (int index = 0; index < size(state); ++index) {
          state(index) *= gamma_last;
        }
        for (int linear = thread; linear < kDim * kChunk; linear += 128) {
          const int row = linear / kChunk;
          const int column = linear - row * kChunk;
          const int source = (((batch * params.tokens + token_base + column)
              * params.q_heads + q_head) * kDim) + row;
          storage.k_a[bf16_tsm_offset(row, column, kDim)] = params.k[source];
        }
        __syncthreads();

        // Stage D: V computes Ag@W while O computes Q@S.
        __syncthreads();

        // Stage E: wait for Vd/Vn publication.
        __syncthreads();

        // Stage F: S += K^T @ Vn.
        mma_from_tsm<Tile12812864>(
            storage.k_a.data(), storage.value_b.data(),
            thread, warp, state, false);
        __syncthreads();
      }

      for (int index = 0; index < size(state); ++index) {
        auto coordinate = coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        params.final_state[state_base + row * kDim + column] = state(index);
      }
    } else if (consumer == 1) {
      Tensor value = partition_fragment_C(
          tiled_mma, Shape<Int<64>, Int<128>>{});
      Tensor coordinates = thr_mma.partition_C(
          make_identity_tensor(Shape<Int<64>, Int<128>>{}));

      for (int chunk = 0; chunk < chunks; ++chunk) {
        const int token_base = chunk * kChunk;
        const int gate_base = (group * chunks + chunk) * kChunk;

        // Stage A: K and Ag producer.
        for (int linear = thread; linear < kChunk * kDim; linear += 128) {
          const int row = linear / kDim;
          const int column = linear - row * kDim;
          const int source = (((batch * params.tokens + token_base + row)
              * params.q_heads + q_head) * kDim) + column;
          storage.k_a[bf16_tsm_offset(row, column, kChunk)] = params.k[source];
        }
        for (int linear = thread; linear < kChunk * kChunk; linear += 128) {
          const int row = linear / kChunk;
          const int column = linear - row * kChunk;
          float entry = 0.0f;
          if (column <= row) {
            entry = float(params.a[gate_base * kChunk + linear])
                * stable_exp_difference(
                    params.g[gate_base + row]
                    - params.g[gate_base + column])
                * params.beta[gate_base + column];
          }
          storage.ag_a[bf16_tsm_offset(row, column, kChunk)] = Element(entry);
        }
        __syncthreads();

        // Stage B: U = K @ S.
        mma_from_tsm<Tile64128128>(
            storage.k_a.data(), storage.state_b.data(),
            thread, warp, value, true);
        __syncthreads();

        // Stage C: W = V - exp(g) U.
        for (int index = 0; index < size(value); ++index) {
          auto coordinate = coordinates(index);
          const int row = get<0>(coordinate);
          const int column = get<1>(coordinate);
          const int source = (((batch * params.tokens + token_base + row)
              * params.value_heads + head) * kDim) + column;
          const float entry = float(params.v[source])
              - expf(params.g[gate_base + row]) * value(index);
          storage.value_b[bf16_tsm_offset(column, row, kDim)] = Element(entry);
        }
        __syncthreads();

        // Stage D: Vd = Ag @ W.
        mma_from_tsm<Tile6412864>(
            storage.ag_a.data(), storage.value_b.data(),
            thread, warp, value, true);
        __syncthreads();

        // Stage E: publish both official BF16 consumers of the FP32 Vd fragment.
        const float g_last = params.g[gate_base + kChunk - 1];
        for (int index = 0; index < size(value); ++index) {
          auto coordinate = coordinates(index);
          const int row = get<0>(coordinate);
          const int column = get<1>(coordinate);
          const int offset = bf16_tsm_offset(column, row, kDim);
          storage.vd_b[offset] = Element(value(index));
          storage.value_b[offset] = Element(
              value(index) * stable_exp_difference(
                  g_last - params.g[gate_base + row]));
        }
        __syncthreads();

        // Stage F: O and S consume Vd/Vn concurrently.
        __syncthreads();
      }
    } else {
      Tensor output = partition_fragment_C(
          tiled_mma, Shape<Int<64>, Int<128>>{});
      Tensor coordinates = thr_mma.partition_C(
          make_identity_tensor(Shape<Int<64>, Int<128>>{}));

      for (int chunk = 0; chunk < chunks; ++chunk) {
        const int token_base = chunk * kChunk;
        const int gate_base = (group * chunks + chunk) * kChunk;

        // Stage A: Q and the padded K^T operand for P = QK^T.
        for (int linear = thread; linear < kChunk * kDim; linear += 128) {
          const int row = linear / kDim;
          const int column = linear - row * kDim;
          const int source = (((batch * params.tokens + token_base + row)
              * params.q_heads + q_head) * kDim) + column;
          storage.q_a[bf16_tsm_offset(row, column, kChunk)] = params.q[source];
        }
        for (int linear = thread; linear < kDim * kDim; linear += 128) {
          const int row = linear / kDim;
          const int column = linear - row * kDim;
          Element entry = Element(0.0f);
          if (row < kChunk) {
            const int source = (((batch * params.tokens + token_base + row)
                * params.q_heads + q_head) * kDim) + column;
            entry = params.k[source];
          }
          storage.k_b[bf16_tsm_offset(row, column, kDim)] = entry;
        }
        __syncthreads();

        // Stage B: P = Q @ K^T.
        mma_from_tsm<Tile64128128>(
            storage.q_a.data(), storage.k_b.data(),
            thread, warp, output, true);
        __syncthreads();

        // Stage C: Pg = scale * Lower(G) * P.
        for (int index = 0; index < size(output); ++index) {
          auto coordinate = coordinates(index);
          const int row = get<0>(coordinate);
          const int column = get<1>(coordinate);
          if (column < kChunk) {
            float entry = 0.0f;
            if (column <= row) {
              entry = output(index) * params.scale
                  * stable_exp_difference(
                      params.g[gate_base + row]
                      - params.g[gate_base + column]);
            }
            storage.pg_a[bf16_tsm_offset(row, column, kChunk)] = Element(entry);
          }
        }
        __syncthreads();

        // Stage D: O = Q @ S.
        mma_from_tsm<Tile64128128>(
            storage.q_a.data(), storage.state_b.data(),
            thread, warp, output, true);
        __syncthreads();

        // Stage E: O = scale * exp(g) * O.
        for (int index = 0; index < size(output); ++index) {
          auto coordinate = coordinates(index);
          output(index) *= params.scale
              * expf(params.g[gate_base + get<0>(coordinate)]);
        }
        __syncthreads();

        // Stage F: O += Pg @ Vd, then commit this chunk.
        mma_from_tsm<Tile6412864>(
            storage.pg_a.data(), storage.vd_b.data(),
            thread, warp, output, false);
        for (int index = 0; index < size(output); ++index) {
          auto coordinate = coordinates(index);
          const int row = get<0>(coordinate);
          const int column = get<1>(coordinate);
          const int destination = (((batch * params.tokens + token_base + row)
              * params.value_heads + head) * kDim) + column;
          params.output[destination] = Element(output(index));
        }
        __syncthreads();
      }
    }
  }
};

// Fuse the vector scaling and per-token reductions surrounding the official
// WY-backward matrix products. Four independent token/head rows share one CTA;
// each 128-thread consumer owns one 128-wide vector and two reductions.
struct FlashQlaWyBackwardPreprocess {
  struct Arguments {
    const float* dk_beta_g;
    const float* dv_beta;
    const float* k;
    const float* v;
    const float* beta;
    const float* g;
    float* dk;
    float* dv;
    float* db;
    float* dg;
    int batch_size;
    int tokens;
    int value_heads;
  };
  using Params = Arguments;

  struct SharedStorage {
    cute::array_aligned<float, 4 * kDim> db;
    cute::array_aligned<float, 4 * kDim> dg;
  };

  static constexpr int MaxThreadsPerBlock = 4 * kDim;
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    const int consumer = threadIdx.x / kDim;
    const int lane = threadIdx.x - consumer * kDim;
    const int output_row = blockIdx.x * 4 + consumer;
    const int rows = params.batch_size * params.tokens * params.value_heads;
    const bool active = output_row < rows;
    int input_row = 0;
    if (active) {
      const int head = output_row % params.value_heads;
      const int token_linear = output_row / params.value_heads;
      const int batch = token_linear / params.tokens;
      const int token = token_linear - batch * params.tokens;
      const int chunks = params.tokens / kChunk;
      const int chunk = token / kChunk;
      const int chunk_row = token - chunk * kChunk;
      input_row = (((batch * chunks + chunk) * params.value_heads + head)
          * kChunk) + chunk_row;
    }
    const int input_index = input_row * kDim + lane;
    const int output_index = output_row * kDim + lane;
    auto& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    const int shared_index = consumer * kDim + lane;

    float db_value = 0.0f;
    float dg_value = 0.0f;
    if (active) {
      const float beta = params.beta[input_row];
      const float exp_g = expf(params.g[input_row]);
      const float dk_beta_g = params.dk_beta_g[input_index];
      const float dv_beta = params.dv_beta[input_index];
      const float key = params.k[input_index];
      params.dk[output_index] = dk_beta_g * beta * exp_g;
      params.dv[output_index] = dv_beta * beta;
      const float key_product = dk_beta_g * key;
      db_value = key_product * exp_g
          + dv_beta * params.v[input_index];
      dg_value = key_product * beta * exp_g;
    }
    storage.db[shared_index] = db_value;
    storage.dg[shared_index] = dg_value;
    __syncthreads();
    for (int stride = kDim / 2; stride > 0; stride >>= 1) {
      if (lane < stride) {
        storage.db[shared_index] += storage.db[shared_index + stride];
        storage.dg[shared_index] += storage.dg[shared_index + stride];
      }
      __syncthreads();
    }
    if (active && lane == 0) {
      params.db[output_row] = storage.db[consumer * kDim];
      params.dg[output_row] = storage.dg[consumer * kDim];
    }
  }
};

struct FlashQlaWyBackwardPostprocess {
  struct Arguments {
    const float* dk_da;
    const float* dk_beta;
    const float* dk1;
    const float* k;
    const float* beta;
    const float* dA;
    const float* A;
    const float* dg1;
    float* dk;
    float* db;
    float* dg;
    int batch_size;
    int tokens;
    int value_heads;
  };
  using Params = Arguments;

  struct SharedStorage {
    cute::array_aligned<float, 4 * kDim> db;
    cute::array_aligned<float, 4 * kDim> dg;
  };

  static constexpr int MaxThreadsPerBlock = 4 * kDim;
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    const int consumer = threadIdx.x / kDim;
    const int lane = threadIdx.x - consumer * kDim;
    const int output_row = blockIdx.x * 4 + consumer;
    const int rows = params.batch_size * params.tokens * params.value_heads;
    const bool active = output_row < rows;
    int input_row = 0;
    int batch = 0;
    int chunk = 0;
    int chunk_row = 0;
    int head = 0;
    if (active) {
      head = output_row % params.value_heads;
      const int token_linear = output_row / params.value_heads;
      batch = token_linear / params.tokens;
      const int token = token_linear - batch * params.tokens;
      chunk = token / kChunk;
      chunk_row = token - chunk * kChunk;
      const int chunks = params.tokens / kChunk;
      input_row = (((batch * chunks + chunk) * params.value_heads + head)
          * kChunk) + chunk_row;
    }
    const int input_index = input_row * kDim + lane;
    const int output_index = output_row * kDim + lane;
    auto& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    const int shared_index = consumer * kDim + lane;

    float db_value = 0.0f;
    float dg_value = 0.0f;
    if (active) {
      const float dk_beta = params.dk_beta[input_index];
      db_value = dk_beta * params.k[input_index];
      params.dk[output_index] += params.dk_da[input_index]
          + dk_beta * params.beta[input_row] + params.dk1[input_index];

      if (lane < kChunk) {
        const int chunks = params.tokens / kChunk;
        const int matrix_base = (((batch * chunks + chunk)
            * params.value_heads + head) * kChunk * kChunk);
        const int row_offset = matrix_base + chunk_row * kChunk + lane;
        const int column_offset = matrix_base + lane * kChunk + chunk_row;
        dg_value = params.dA[row_offset] * params.A[row_offset]
            - params.dA[column_offset] * params.A[column_offset];
      }
    }
    storage.db[shared_index] = db_value;
    storage.dg[shared_index] = dg_value;
    __syncthreads();
    for (int stride = kDim / 2; stride > 0; stride >>= 1) {
      if (lane < stride) {
        storage.db[shared_index] += storage.db[shared_index + stride];
        storage.dg[shared_index] += storage.dg[shared_index + stride];
      }
      __syncthreads();
    }
    if (active && lane == 0) {
      params.db[output_row] += storage.db[consumer * kDim];
      params.dg[output_row] += storage.dg[consumer * kDim]
          + params.dg1[input_row];
    }
  }
};

// One-CTA implementation of the official WY/KKT backward chain.  A CTA owns
// one chunk/value-head and performs every 64x64/64x128 product with BF16 TSM
// handoffs and FP32 accumulation, replacing the framework bmm sequence plus
// the separate preprocess/postprocess launches.
struct FlashQlaFusedWyBackward {
  struct Arguments {
    const float* k;
    const float* v;
    const float* beta;
    const float* a;
    const float* g;
    const float* dw;
    const float* du;
    const float* dk1;
    const float* dg1;
    float* dk;
    float* dv;
    float* db;
    float* dg;
    int batch_size;
    int tokens;
    int q_heads;
    int value_heads;
  };
  using Params = Arguments;

  struct SharedStorage {
    cute::array_aligned<Element, 64 * 128> vec_a;
    cute::array_aligned<Element, 64 * 128> vec_b;
    cute::array_aligned<Element, 64 * 64> mat_a;
    cute::array_aligned<Element, 64 * 64> mat_b;
    cute::array_aligned<float, 64 * 64> da;
    cute::array_aligned<float, 64 * 64> gate_product;
    cute::array_aligned<float, 64 * 128> vector;
    cute::array_aligned<float, 64> db;
    cute::array_aligned<float, 64> dg;
  };

  static constexpr int MaxThreadsPerBlock = size(TiledMma{});
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    const int group = blockIdx.x;
    const int chunks = params.tokens / kChunk;
    const int groups = params.batch_size * chunks * params.value_heads;
    if (group >= groups) {
      return;
    }
    const int head = group % params.value_heads;
    const int batch_chunk = group / params.value_heads;
    const int chunk = batch_chunk % chunks;
    const int batch = batch_chunk / chunks;
    const int q_head = head / (params.value_heads / params.q_heads);
    const int token_base = chunk * kChunk;
    const int thread = threadIdx.x;
    int warp = cutlass::canonical_warp_idx_sync();
    auto& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread);
    Tensor matrix = partition_fragment_C(
        tiled_mma, Shape<Int<64>, Int<64>>{});
    Tensor matrix_coordinates = thr_mma.partition_C(
        make_identity_tensor(Shape<Int<64>, Int<64>>{}));

    // dA0 = dW @ (K * beta * exp(g))^T
    //     + dU @ (V * beta)^T.
    for (int linear = thread; linear < kChunk * kDim;
         linear += blockDim.x) {
      const int row = linear / kDim;
      const int column = linear - row * kDim;
      const int token = token_base + row;
      const int value_index = (((batch * params.tokens + token)
          * params.value_heads + head) * kDim) + column;
      const int key_index = (((batch * params.tokens + token)
          * params.q_heads + q_head) * kDim) + column;
      const int scalar = (batch * params.tokens + token)
          * params.value_heads + head;
      storage.vec_a[bf16_tsm_offset(row, column, kChunk)] =
          Element(params.dw[value_index]);
      storage.vec_b[bf16_tsm_offset(row, column, kChunk)] = Element(
          params.k[key_index] * params.beta[scalar] * expf(params.g[scalar]));
    }
    __syncthreads();
    mma_from_tsm<Tile6464128>(
        storage.vec_a.data(), storage.vec_b.data(),
        thread, warp, matrix, true);
    for (int linear = thread; linear < kChunk * kDim;
         linear += blockDim.x) {
      const int row = linear / kDim;
      const int column = linear - row * kDim;
      const int token = token_base + row;
      const int value_index = (((batch * params.tokens + token)
          * params.value_heads + head) * kDim) + column;
      const int scalar = (batch * params.tokens + token)
          * params.value_heads + head;
      storage.vec_a[bf16_tsm_offset(row, column, kChunk)] =
          Element(params.du[value_index]);
      storage.vec_b[bf16_tsm_offset(row, column, kChunk)] = Element(
          params.v[value_index] * params.beta[scalar]);
    }
    __syncthreads();
    mma_from_tsm<Tile6464128>(
        storage.vec_a.data(), storage.vec_b.data(),
        thread, warp, matrix, false);
    for (int index = 0; index < size(matrix); ++index) {
      auto coordinate = matrix_coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      storage.da[row * kChunk + column] =
          row > column ? matrix(index) : 0.0f;
    }
    __syncthreads();

    // dA = A^T @ tril(dA0, -1) @ A^T.
    for (int linear = thread; linear < kChunk * kChunk;
         linear += blockDim.x) {
      const int row = linear / kChunk;
      const int column = linear - row * kChunk;
      const int a_transposed = (((batch * params.tokens + token_base + column)
          * params.value_heads + head) * kChunk) + row;
      storage.mat_a[bf16_tsm_offset(row, column, kChunk)] =
          Element(params.a[a_transposed]);
      storage.mat_b[bf16_tsm_offset(column, row, kChunk)] =
          Element(storage.da[row * kChunk + column]);
    }
    __syncthreads();
    mma_from_tsm<Tile646464>(
        storage.mat_a.data(), storage.mat_b.data(),
        thread, warp, matrix, true);
    for (int index = 0; index < size(matrix); ++index) {
      auto coordinate = matrix_coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      storage.mat_a[bf16_tsm_offset(row, column, kChunk)] =
          Element(matrix(index));
    }
    for (int linear = thread; linear < kChunk * kChunk;
         linear += blockDim.x) {
      const int row = linear / kChunk;
      const int column = linear - row * kChunk;
      const int a_index = (((batch * params.tokens + token_base + row)
          * params.value_heads + head) * kChunk) + column;
      storage.mat_b[bf16_tsm_offset(row, column, kChunk)] =
          Element(params.a[a_index]);
    }
    __syncthreads();
    mma_from_tsm<Tile646464>(
        storage.mat_a.data(), storage.mat_b.data(),
        thread, warp, matrix, true);
    for (int index = 0; index < size(matrix); ++index) {
      auto coordinate = matrix_coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      float entry = 0.0f;
      if (row > column) {
        const int row_scalar = (batch * params.tokens + token_base + row)
            * params.value_heads + head;
        const int column_scalar =
            (batch * params.tokens + token_base + column)
            * params.value_heads + head;
        entry = -matrix(index) * stable_exp_difference(
            params.g[row_scalar] - params.g[column_scalar]);
      }
      storage.da[row * kChunk + column] = entry;
      storage.mat_a[bf16_tsm_offset(row, column, kChunk)] = Element(entry);
    }
    __syncthreads();

    // Agram = (K * beta) @ K^T, retained only as dA*Agram for dg.
    for (int linear = thread; linear < kChunk * kDim;
         linear += blockDim.x) {
      const int row = linear / kDim;
      const int column = linear - row * kDim;
      const int token = token_base + row;
      const int key_index = (((batch * params.tokens + token)
          * params.q_heads + q_head) * kDim) + column;
      const int scalar = (batch * params.tokens + token)
          * params.value_heads + head;
      storage.vec_a[bf16_tsm_offset(row, column, kChunk)] = Element(
          params.k[key_index] * params.beta[scalar]);
      storage.vec_b[bf16_tsm_offset(row, column, kChunk)] =
          Element(params.k[key_index]);
    }
    __syncthreads();
    mma_from_tsm<Tile6464128>(
        storage.vec_a.data(), storage.vec_b.data(),
        thread, warp, matrix, true);
    for (int index = 0; index < size(matrix); ++index) {
      auto coordinate = matrix_coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      const int linear = row * kChunk + column;
      storage.gate_product[linear] = storage.da[linear] * matrix(index);
    }
    __syncthreads();

    Tensor vector = partition_fragment_C(
        tiled_mma, Shape<Int<64>, Int<128>>{});
    Tensor vector_coordinates = thr_mma.partition_C(
        make_identity_tensor(Shape<Int<64>, Int<128>>{}));

    // dk_beta_g = A^T @ dW and its direct dk/db/dg contribution.
    for (int linear = thread; linear < kChunk * kChunk;
         linear += blockDim.x) {
      const int row = linear / kChunk;
      const int column = linear - row * kChunk;
      const int a_transposed = (((batch * params.tokens + token_base + column)
          * params.value_heads + head) * kChunk) + row;
      storage.mat_a[bf16_tsm_offset(row, column, kChunk)] =
          Element(params.a[a_transposed]);
    }
    for (int linear = thread; linear < kChunk * kDim;
         linear += blockDim.x) {
      const int row = linear / kDim;
      const int column = linear - row * kDim;
      const int value_index = (((batch * params.tokens + token_base + row)
          * params.value_heads + head) * kDim) + column;
      storage.vec_b[bf16_tsm_offset(column, row, kChunk)] =
          Element(params.dw[value_index]);
    }
    __syncthreads();
    mma_from_tsm<Tile6412864>(
        storage.mat_a.data(), storage.vec_b.data(),
        thread, warp, vector, true);
    for (int index = 0; index < size(vector); ++index) {
      auto coordinate = vector_coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      const int token = token_base + row;
      const int scalar = (batch * params.tokens + token)
          * params.value_heads + head;
      const int destination = scalar * kDim + column;
      storage.vector[row * kDim + column] = vector(index);
      params.dk[destination] = vector(index) * params.beta[scalar]
          * expf(params.g[scalar]);
    }
    __syncthreads();
    if (thread < kChunk) {
      const int token = token_base + thread;
      const int scalar = (batch * params.tokens + token)
          * params.value_heads + head;
      float dot = 0.0f;
      for (int column = 0; column < kDim; ++column) {
        const int key_index = (((batch * params.tokens + token)
            * params.q_heads + q_head) * kDim) + column;
        dot += storage.vector[thread * kDim + column]
            * params.k[key_index];
      }
      storage.db[thread] = dot * expf(params.g[scalar]);
      storage.dg[thread] = dot * params.beta[scalar]
          * expf(params.g[scalar]);
    }
    __syncthreads();

    // dv_beta = A^T @ dU.
    for (int linear = thread; linear < kChunk * kDim;
         linear += blockDim.x) {
      const int row = linear / kDim;
      const int column = linear - row * kDim;
      const int value_index = (((batch * params.tokens + token_base + row)
          * params.value_heads + head) * kDim) + column;
      storage.vec_b[bf16_tsm_offset(column, row, kChunk)] =
          Element(params.du[value_index]);
    }
    __syncthreads();
    mma_from_tsm<Tile6412864>(
        storage.mat_a.data(), storage.vec_b.data(),
        thread, warp, vector, true);
    for (int index = 0; index < size(vector); ++index) {
      auto coordinate = vector_coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      const int token = token_base + row;
      const int scalar = (batch * params.tokens + token)
          * params.value_heads + head;
      storage.vector[row * kDim + column] = vector(index);
      params.dv[scalar * kDim + column] =
          vector(index) * params.beta[scalar];
    }
    __syncthreads();
    if (thread < kChunk) {
      const int token = token_base + thread;
      const int scalar = (batch * params.tokens + token)
          * params.value_heads + head;
      float dot = 0.0f;
      for (int column = 0; column < kDim; ++column) {
        dot += storage.vector[thread * kDim + column]
            * params.v[scalar * kDim + column];
      }
      storage.db[thread] += dot;
    }
    __syncthreads();

    // dk_beta = dA @ K.
    for (int linear = thread; linear < kChunk * kChunk;
         linear += blockDim.x) {
      const int row = linear / kChunk;
      const int column = linear - row * kChunk;
      storage.mat_a[bf16_tsm_offset(row, column, kChunk)] =
          Element(storage.da[linear]);
    }
    for (int linear = thread; linear < kChunk * kDim;
         linear += blockDim.x) {
      const int row = linear / kDim;
      const int column = linear - row * kDim;
      const int key_index = (((batch * params.tokens + token_base + row)
          * params.q_heads + q_head) * kDim) + column;
      storage.vec_b[bf16_tsm_offset(column, row, kChunk)] =
          Element(params.k[key_index]);
    }
    __syncthreads();
    mma_from_tsm<Tile6412864>(
        storage.mat_a.data(), storage.vec_b.data(),
        thread, warp, vector, true);
    for (int index = 0; index < size(vector); ++index) {
      auto coordinate = vector_coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      storage.vector[row * kDim + column] = vector(index);
    }
    __syncthreads();
    if (thread < kChunk) {
      const int token = token_base + thread;
      float dot = 0.0f;
      for (int column = 0; column < kDim; ++column) {
        const int key_index = (((batch * params.tokens + token)
            * params.q_heads + q_head) * kDim) + column;
        dot += storage.vector[thread * kDim + column]
            * params.k[key_index];
      }
      storage.db[thread] += dot;
    }
    __syncthreads();

    // dk_da = dA^T @ (K * beta), followed by final vector/scalar epilogues.
    for (int linear = thread; linear < kChunk * kChunk;
         linear += blockDim.x) {
      const int row = linear / kChunk;
      const int column = linear - row * kChunk;
      storage.mat_a[bf16_tsm_offset(row, column, kChunk)] =
          Element(storage.da[column * kChunk + row]);
    }
    for (int linear = thread; linear < kChunk * kDim;
         linear += blockDim.x) {
      const int row = linear / kDim;
      const int column = linear - row * kDim;
      const int token = token_base + row;
      const int scalar = (batch * params.tokens + token)
          * params.value_heads + head;
      const int key_index = (((batch * params.tokens + token)
          * params.q_heads + q_head) * kDim) + column;
      storage.vec_b[bf16_tsm_offset(column, row, kChunk)] = Element(
          params.k[key_index] * params.beta[scalar]);
    }
    __syncthreads();
    mma_from_tsm<Tile6412864>(
        storage.mat_a.data(), storage.vec_b.data(),
        thread, warp, vector, true);
    for (int index = 0; index < size(vector); ++index) {
      auto coordinate = vector_coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      const int token = token_base + row;
      const int scalar = (batch * params.tokens + token)
          * params.value_heads + head;
      const int destination = scalar * kDim + column;
      params.dk[destination] += vector(index)
          + storage.vector[row * kDim + column] * params.beta[scalar]
          + params.dk1[destination];
    }
    __syncthreads();
    if (thread < kChunk) {
      float row_sum = 0.0f;
      float column_sum = 0.0f;
      for (int column = 0; column < kChunk; ++column) {
        row_sum += storage.gate_product[thread * kChunk + column];
        column_sum += storage.gate_product[column * kChunk + thread];
      }
      const int scalar = (batch * params.tokens + token_base + thread)
          * params.value_heads + head;
      params.db[scalar] = storage.db[thread];
      params.dg[scalar] = storage.dg[thread] + row_sum - column_sum
          + params.dg1[scalar];
    }
  }
};

// Convert the five BF16 tensors retained by the public autograd context into
// the FP32 working tensors used by the official decomposed backward. Keeping
// this in one launch preserves exact cast semantics while removing four
// launch round trips and traversing all independent inputs concurrently.
__global__ void flash_qla_prepare_backward_inputs_bf16_128_kernel(
    const Element* q,
    const Element* k,
    const Element* v,
    const Element* a,
    const Element* do_value,
    float* q_float,
    float* k_float,
    float* v_float,
    float* a_float,
    float* do_float,
    int q_elements,
    int value_elements,
    int a_elements) {
  const int stride = blockDim.x * gridDim.x;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < value_elements;
       index += stride) {
    v_float[index] = float(v[index]);
    do_float[index] = float(do_value[index]);
    if (index < q_elements) {
      q_float[index] = float(q[index]);
      k_float[index] = float(k[index]);
    }
    if (index < a_elements) {
      a_float[index] = float(a[index]);
    }
  }
}

}  // namespace

extern "C" void launch_flash_qla_prepare_backward_inputs_bf16_128_v1(
    void* q,
    void* k,
    void* v,
    void* a,
    void* do_value,
    void* q_float,
    void* k_float,
    void* v_float,
    void* a_float,
    void* do_float,
    int batch_size,
    int tokens,
    int q_heads,
    int value_heads,
    void* raw_stream,
    int& result) {
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  const int q_elements = batch_size * tokens * q_heads * kDim;
  const int value_elements = batch_size * tokens * value_heads * kDim;
  const int a_elements = batch_size * tokens * value_heads * kChunk;
  int blocks = (value_elements + 255) / 256;
  blocks = blocks < 4096 ? blocks : 4096;
  flash_qla_prepare_backward_inputs_bf16_128_kernel<<<
      blocks, 256, 0, stream>>>(
      reinterpret_cast<const Element*>(q),
      reinterpret_cast<const Element*>(k),
      reinterpret_cast<const Element*>(v),
      reinterpret_cast<const Element*>(a),
      reinterpret_cast<const Element*>(do_value),
      reinterpret_cast<float*>(q_float),
      reinterpret_cast<float*>(k_float),
      reinterpret_cast<float*>(v_float),
      reinterpret_cast<float*>(a_float),
      reinterpret_cast<float*>(do_float),
      q_elements,
      value_elements,
      a_elements);
  result = int(hggcGetLastError());
}

extern "C" void launch_flash_qla_fused_forward_bf16_128(
    void* q,
    void* k,
    void* v,
    void* a,
    void* g,
    void* beta,
    void* initial_state,
    void* output,
    void* final_state,
    int batch_size,
    int tokens,
    int q_heads,
    int value_heads,
    float scale,
    bool use_initial_state,
    void* raw_stream,
    int& result) {
  using Kernel = FlashQlaFusedForward;
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  typename Kernel::Arguments arguments{
      reinterpret_cast<const Element*>(q),
      reinterpret_cast<const Element*>(k),
      reinterpret_cast<const Element*>(v),
      reinterpret_cast<const Element*>(a),
      reinterpret_cast<const float*>(g),
      reinterpret_cast<const float*>(beta),
      reinterpret_cast<const float*>(initial_state),
      reinterpret_cast<Element*>(output),
      reinterpret_cast<float*>(final_state),
      batch_size,
      tokens,
      q_heads,
      value_heads,
      scale,
      use_initial_state,
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
      batch_size * value_heads * 2,
      Kernel::MaxThreadsPerBlock,
      Kernel::SharedStorageSize,
      stream>>>(arguments);
  result = int(hggcGetLastError());
}

extern "C" void launch_flash_qla_cp_dh_backward_bf16_128(
    void* q,
    void* k,
    void* a,
    void* g,
    void* beta,
    void* do_value,
    void* dh0,
    int batch_size,
    int tokens,
    int q_heads,
    int value_heads,
    float scale,
    void* raw_stream,
    int& result) {
  using Kernel = FlashQlaCpDhBackwardWarpSpecialized;
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  typename Kernel::Arguments arguments{
      reinterpret_cast<const float*>(q),
      reinterpret_cast<const float*>(k),
      reinterpret_cast<const float*>(a),
      reinterpret_cast<const float*>(g),
      reinterpret_cast<const float*>(beta),
      reinterpret_cast<const float*>(do_value),
      reinterpret_cast<float*>(dh0),
      batch_size,
      tokens,
      q_heads,
      value_heads,
      scale,
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
      batch_size * value_heads * 2,
      Kernel::MaxThreadsPerBlock,
      Kernel::SharedStorageSize,
      stream>>>(arguments);
  result = int(hggcGetLastError());
}

extern "C" void launch_flash_qla_wu_forward_bf16_128(
    void* k,
    void* v,
    void* a,
    void* g,
    void* beta,
    void* w,
    void* u,
    int batch_size,
    int tokens,
    int q_heads,
    int value_heads,
    void* raw_stream,
    int& result) {
  using Kernel = FlashQlaWuForward;
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  typename Kernel::Arguments arguments{
      reinterpret_cast<const float*>(k),
      reinterpret_cast<const float*>(v),
      reinterpret_cast<const float*>(a),
      reinterpret_cast<const float*>(g),
      reinterpret_cast<const float*>(beta),
      reinterpret_cast<float*>(w),
      reinterpret_cast<float*>(u),
      batch_size,
      tokens,
      q_heads,
      value_heads,
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
      batch_size * (tokens / kChunk) * value_heads,
      Kernel::MaxThreadsPerBlock,
      Kernel::SharedStorageSize,
      stream>>>(arguments);
  result = int(hggcGetLastError());
}

extern "C" void launch_flash_qla_chunk_state_forward_bf16_128(
    void* k,
    void* w,
    void* u,
    void* g,
    void* initial_state,
    void* history,
    void* vn,
    void* final_state,
    int batch_size,
    int tokens,
    int q_heads,
    int value_heads,
    bool use_initial_state,
    void* raw_stream,
    int& result) {
  using Kernel = FlashQlaChunkStateForward;
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  typename Kernel::Arguments arguments{
      reinterpret_cast<const float*>(k),
      reinterpret_cast<const float*>(w),
      reinterpret_cast<const float*>(u),
      reinterpret_cast<const float*>(g),
      reinterpret_cast<const float*>(initial_state),
      reinterpret_cast<float*>(history),
      reinterpret_cast<float*>(vn),
      reinterpret_cast<float*>(final_state),
      batch_size,
      tokens,
      q_heads,
      value_heads,
      use_initial_state,
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
      batch_size * value_heads * 2,
      Kernel::MaxThreadsPerBlock,
      Kernel::SharedStorageSize,
      stream>>>(arguments);
  result = int(hggcGetLastError());
}

extern "C" void launch_flash_qla_fused_prepare_h_bf16_128(
    void* k,
    void* v,
    void* a,
    void* g,
    void* beta,
    void* initial_state,
    void* w,
    void* history,
    void* vn,
    void* final_state,
    int batch_size,
    int tokens,
    int q_heads,
    int value_heads,
    bool use_initial_state,
    void* raw_stream,
    int& result) {
  using Kernel = FlashQlaFusedPrepareH;
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  typename Kernel::Arguments arguments{
      reinterpret_cast<const float*>(k),
      reinterpret_cast<const float*>(v),
      reinterpret_cast<const float*>(a),
      reinterpret_cast<const float*>(g),
      reinterpret_cast<const float*>(beta),
      reinterpret_cast<const float*>(initial_state),
      reinterpret_cast<float*>(w),
      reinterpret_cast<float*>(history),
      reinterpret_cast<float*>(vn),
      reinterpret_cast<float*>(final_state),
      batch_size,
      tokens,
      q_heads,
      value_heads,
      use_initial_state,
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
      batch_size * value_heads,
      Kernel::MaxThreadsPerBlock,
      Kernel::SharedStorageSize,
      stream>>>(arguments);
  result = int(hggcGetLastError());
}

extern "C" void launch_flash_qla_chunk_state_backward_bf16_128(
    void* q,
    void* k,
    void* w,
    void* g,
    void* do_value,
    void* dv_input,
    void* terminal_state_grad,
    void* dh,
    void* dv_output,
    void* initial_state_grad,
    int batch_size,
    int tokens,
    int q_heads,
    int value_heads,
    float scale,
    bool use_terminal_state_grad,
    void* raw_stream,
    int& result) {
  using Kernel = FlashQlaChunkStateBackwardWarpSpecialized;
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  typename Kernel::Arguments arguments{
      reinterpret_cast<const float*>(q),
      reinterpret_cast<const float*>(k),
      reinterpret_cast<const float*>(w),
      reinterpret_cast<const float*>(g),
      reinterpret_cast<const float*>(do_value),
      reinterpret_cast<const float*>(dv_input),
      reinterpret_cast<const float*>(terminal_state_grad),
      reinterpret_cast<float*>(dh),
      reinterpret_cast<float*>(dv_output),
      reinterpret_cast<float*>(initial_state_grad),
      batch_size,
      tokens,
      q_heads,
      value_heads,
      scale,
      use_terminal_state_grad,
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
      batch_size * value_heads,
      Kernel::MaxThreadsPerBlock,
      Kernel::SharedStorageSize,
      stream>>>(arguments);
  result = int(hggcGetLastError());
}

extern "C" void launch_flash_qla_chunk_state_backward_step_bf16_128(
    void* q,
    void* k,
    void* w,
    void* g,
    void* do_value,
    void* dv,
    void* dstate,
    void* dh,
    int batch_size,
    int tokens,
    int q_heads,
    int value_heads,
    int chunk,
    float scale,
    void* raw_stream,
    int& result) {
  using Kernel = FlashQlaChunkStateBackwardStep;
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  typename Kernel::Arguments arguments{
      reinterpret_cast<const float*>(q),
      reinterpret_cast<const float*>(k),
      reinterpret_cast<const float*>(w),
      reinterpret_cast<const float*>(g),
      reinterpret_cast<const float*>(do_value),
      reinterpret_cast<float*>(dv),
      reinterpret_cast<float*>(dstate),
      reinterpret_cast<float*>(dh),
      batch_size,
      tokens,
      q_heads,
      value_heads,
      chunk,
      scale,
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
      batch_size * value_heads * 2,
      Kernel::MaxThreadsPerBlock,
      Kernel::SharedStorageSize,
      stream>>>(arguments);
  result = int(hggcGetLastError());
}

extern "C" void launch_flash_qla_fused_state_dqkwg_backward_bf16_128_v2(
    void* q,
    void* k,
    void* w,
    void* g,
    void* do_value,
    void* terminal_state_grad,
    void* v_corrected,
    void* history,
    void* dq,
    void* dk,
    void* dw,
    void* dg,
    void* dv_output,
    void* initial_state_grad,
    int batch_size,
    int tokens,
    int q_heads,
    int value_heads,
    float scale,
    bool use_terminal_state_grad,
    void* raw_stream,
    int& result) {
  using Kernel = FlashQlaFusedStateDqkwgBackward;
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  typename Kernel::Arguments arguments{
      reinterpret_cast<const float*>(q),
      reinterpret_cast<const float*>(k),
      reinterpret_cast<const float*>(w),
      reinterpret_cast<const float*>(g),
      reinterpret_cast<const float*>(do_value),
      reinterpret_cast<const float*>(terminal_state_grad),
      reinterpret_cast<const float*>(v_corrected),
      reinterpret_cast<const float*>(history),
      reinterpret_cast<float*>(dq),
      reinterpret_cast<float*>(dk),
      reinterpret_cast<float*>(dw),
      reinterpret_cast<float*>(dg),
      reinterpret_cast<float*>(dv_output),
      reinterpret_cast<float*>(initial_state_grad),
      batch_size,
      tokens,
      q_heads,
      value_heads,
      scale,
      use_terminal_state_grad,
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
      batch_size * value_heads,
      Kernel::MaxThreadsPerBlock,
      Kernel::SharedStorageSize,
      stream>>>(arguments);
  result = int(hggcGetLastError());
}

extern "C" void launch_flash_qla_chunk_dv_backward_bf16_128(
    void* q,
    void* k,
    void* g,
    void* do_value,
    void* dv,
    int batch_size,
    int tokens,
    int q_heads,
    int value_heads,
    float scale,
    void* raw_stream,
    int& result) {
  using Kernel = FlashQlaChunkDvBackward;
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  const int chunks = tokens / kChunk;
  typename Kernel::Arguments arguments{
      reinterpret_cast<const float*>(q),
      reinterpret_cast<const float*>(k),
      reinterpret_cast<const float*>(g),
      reinterpret_cast<const float*>(do_value),
      reinterpret_cast<float*>(dv),
      batch_size,
      tokens,
      q_heads,
      value_heads,
      chunks,
      scale,
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
      batch_size * value_heads * chunks,
      Kernel::MaxThreadsPerBlock,
      Kernel::SharedStorageSize,
      stream>>>(arguments);
  result = int(hggcGetLastError());
}

extern "C" void launch_flash_qla_chunk_dqkw_backward_bf16_128(
    void* do_value,
    void* v_corrected,
    void* dv,
    void* history,
    void* dh,
    void* g,
    void* dq,
    void* dk,
    void* dw,
    int batch_size,
    int tokens,
    int value_heads,
    float scale,
    void* raw_stream,
    int& result) {
  using Kernel = FlashQlaChunkDqkWBackward;
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  const int chunks = tokens / kChunk;
  typename Kernel::Arguments arguments{
      reinterpret_cast<const float*>(do_value),
      reinterpret_cast<const float*>(v_corrected),
      reinterpret_cast<const float*>(dv),
      reinterpret_cast<const float*>(history),
      reinterpret_cast<const float*>(dh),
      reinterpret_cast<const float*>(g),
      reinterpret_cast<float*>(dq),
      reinterpret_cast<float*>(dk),
      reinterpret_cast<float*>(dw),
      batch_size,
      tokens,
      value_heads,
      chunks,
      scale,
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
      batch_size * value_heads * chunks,
      Kernel::MaxThreadsPerBlock,
      Kernel::SharedStorageSize,
      stream>>>(arguments);
  result = int(hggcGetLastError());
}

extern "C" void launch_flash_qla_chunk_dqkwg_backward_bf16_128(
    void* q,
    void* k,
    void* do_value,
    void* v_corrected,
    void* dv,
    void* history,
    void* dh,
    void* g,
    void* dq,
    void* dk,
    void* dw,
    void* dg,
    int batch_size,
    int tokens,
    int q_heads,
    int value_heads,
    float scale,
    void* raw_stream,
    int& result) {
  using Kernel = FlashQlaChunkDqkwgBackwardWarpSpecialized;
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  const int chunks = tokens / kChunk;
  typename Kernel::Arguments arguments{
      reinterpret_cast<const float*>(q),
      reinterpret_cast<const float*>(k),
      reinterpret_cast<const float*>(do_value),
      reinterpret_cast<const float*>(v_corrected),
      reinterpret_cast<const float*>(dv),
      reinterpret_cast<const float*>(history),
      reinterpret_cast<const float*>(dh),
      reinterpret_cast<const float*>(g),
      reinterpret_cast<float*>(dq),
      reinterpret_cast<float*>(dk),
      reinterpret_cast<float*>(dw),
      reinterpret_cast<float*>(dg),
      batch_size,
      tokens,
      q_heads,
      value_heads,
      chunks,
      scale,
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
      batch_size * value_heads * chunks,
      Kernel::MaxThreadsPerBlock,
      Kernel::SharedStorageSize,
      stream>>>(arguments);
  result = int(hggcGetLastError());
}

extern "C" void launch_flash_qla_wy_backward_preprocess_128(
    void* dk_beta_g,
    void* dv_beta,
    void* k,
    void* v,
    void* beta,
    void* g,
    void* dk,
    void* dv,
    void* db,
    void* dg,
    int batch_size,
    int tokens,
    int value_heads,
    void* raw_stream,
    int& result) {
  using Kernel = FlashQlaWyBackwardPreprocess;
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  typename Kernel::Arguments arguments{
      reinterpret_cast<const float*>(dk_beta_g),
      reinterpret_cast<const float*>(dv_beta),
      reinterpret_cast<const float*>(k),
      reinterpret_cast<const float*>(v),
      reinterpret_cast<const float*>(beta),
      reinterpret_cast<const float*>(g),
      reinterpret_cast<float*>(dk),
      reinterpret_cast<float*>(dv),
      reinterpret_cast<float*>(db),
      reinterpret_cast<float*>(dg),
      batch_size,
      tokens,
      value_heads,
  };
  const int rows = batch_size * tokens * value_heads;
  cutlass::device_kernel<Kernel><<<
      (rows + 3) / 4,
      Kernel::MaxThreadsPerBlock,
      Kernel::SharedStorageSize,
      stream>>>(arguments);
  result = int(hggcGetLastError());
}

extern "C" void launch_flash_qla_wy_backward_postprocess_128(
    void* dk_da,
    void* dk_beta,
    void* dk1,
    void* k,
    void* beta,
    void* dA,
    void* A,
    void* dg1,
    void* dk,
    void* db,
    void* dg,
    int batch_size,
    int tokens,
    int value_heads,
    void* raw_stream,
    int& result) {
  using Kernel = FlashQlaWyBackwardPostprocess;
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  typename Kernel::Arguments arguments{
      reinterpret_cast<const float*>(dk_da),
      reinterpret_cast<const float*>(dk_beta),
      reinterpret_cast<const float*>(dk1),
      reinterpret_cast<const float*>(k),
      reinterpret_cast<const float*>(beta),
      reinterpret_cast<const float*>(dA),
      reinterpret_cast<const float*>(A),
      reinterpret_cast<const float*>(dg1),
      reinterpret_cast<float*>(dk),
      reinterpret_cast<float*>(db),
      reinterpret_cast<float*>(dg),
      batch_size,
      tokens,
      value_heads,
  };
  const int rows = batch_size * tokens * value_heads;
  cutlass::device_kernel<Kernel><<<
      (rows + 3) / 4,
      Kernel::MaxThreadsPerBlock,
      Kernel::SharedStorageSize,
      stream>>>(arguments);
  result = int(hggcGetLastError());
}

extern "C" void launch_flash_qla_fused_wy_backward_128_v1(
    void* k,
    void* v,
    void* beta,
    void* a,
    void* g,
    void* dw,
    void* du,
    void* dk1,
    void* dg1,
    void* dk,
    void* dv,
    void* db,
    void* dg,
    int batch_size,
    int tokens,
    int q_heads,
    int value_heads,
    void* raw_stream,
    int& result) {
  using Kernel = FlashQlaFusedWyBackward;
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  typename Kernel::Arguments arguments{
      reinterpret_cast<const float*>(k),
      reinterpret_cast<const float*>(v),
      reinterpret_cast<const float*>(beta),
      reinterpret_cast<const float*>(a),
      reinterpret_cast<const float*>(g),
      reinterpret_cast<const float*>(dw),
      reinterpret_cast<const float*>(du),
      reinterpret_cast<const float*>(dk1),
      reinterpret_cast<const float*>(dg1),
      reinterpret_cast<float*>(dk),
      reinterpret_cast<float*>(dv),
      reinterpret_cast<float*>(db),
      reinterpret_cast<float*>(dg),
      batch_size,
      tokens,
      q_heads,
      value_heads,
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
      batch_size * (tokens / kChunk) * value_heads,
      Kernel::MaxThreadsPerBlock,
      Kernel::SharedStorageSize,
      stream>>>(arguments);
  result = int(hggcGetLastError());
}

extern "C" void launch_flash_qla_affine_state_bf16_128(
    void* k,
    void* v,
    void* x,
    void* gamma_last,
    void* reverse_decay,
    void* warmup_chunks,
    void* fallback,
    void* output,
    int groups,
    int chunks,
    bool matrix_mode,
    void* raw_stream,
    int& result) {
  using Kernel = FlashQlaAffineState;
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  typename Kernel::Arguments arguments{
      reinterpret_cast<const Element*>(k),
      reinterpret_cast<const Element*>(v),
      reinterpret_cast<const Element*>(x),
      reinterpret_cast<const float*>(gamma_last),
      reinterpret_cast<const float*>(reverse_decay),
      reinterpret_cast<const int*>(warmup_chunks),
      reinterpret_cast<const uint8_t*>(fallback),
      reinterpret_cast<float*>(output),
      groups,
      chunks,
      matrix_mode,
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
      groups * 2,
      Kernel::MaxThreadsPerBlock,
      Kernel::SharedStorageSize,
      stream>>>(arguments);
  result = int(hggcGetLastError());
}
