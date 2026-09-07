// Official persistent reverse S/K/A topology for PPU. One CTA owns one
// batch/value-head and walks all chunks in reverse. S keeps dState resident,
// K consumes its BF16 handoff for dV and the direct H/dH terms, while A owns
// dPg/P/dP. This removes the global dH stage boundary entirely.
struct FlashQlaFusedStateDqkwgBackward {
  struct Arguments {
    const float* q;
    const float* k;
    const float* w;
    const float* g;
    const float* do_value;
    const float* terminal_state_grad;
    const float* v_corrected;
    const float* history;
    float* dq;
    float* dk;
    float* dw;
    float* dg;
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
    cute::array_aligned<Element, 128 * 128> dstate_b;
    cute::array_aligned<Element, 64 * 128> state_k_a;
    cute::array_aligned<Element, 128 * 64> state_a;
    cute::array_aligned<Element, 128 * 64> state_b;
    cute::array_aligned<Element, 64 * 128> direct_a;
    cute::array_aligned<Element, 128 * 128> direct_b;
    cute::array_aligned<Element, 64 * 128> dp_a;
    cute::array_aligned<Element, 128 * 128> dp_b;
    cute::array_aligned<Element, 64 * 64> dp;
    cute::array_aligned<float, 64 * 64> gate_product;
    cute::array_aligned<float, 64> gate;
    cute::array_aligned<float, 64> decay_to_last;
    cute::array_aligned<float, 64> dg;
    cute::array_aligned<float, 64> state_reduction;
    cute::array_aligned<float, 64> k_reduction;
    float dg_last;
  };

  static constexpr int MaxThreadsPerBlock = 384;
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  static __attribute__((__noinline__)) __device__ void run_s(
      Params const& params, SharedStorage& storage) {
    const int group = blockIdx.x;
    const int batch = group / params.value_heads;
    const int head = group - batch * params.value_heads;
    const int q_head = head / (params.value_heads / params.q_heads);
    const int chunks = params.tokens / kChunk;
    const int state_base = group * kDim * kDim;
    const int thread = threadIdx.x;
    const int warp = thread / 32;
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread);
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
      if (thread < kChunk) {
        const float gate = params.g[gate_base + thread];
        storage.gate[thread] = gate;
        storage.decay_to_last[thread] = stable_exp_difference(
            params.g[gate_base + kChunk - 1] - gate);
      }
      for (int index = 0; index < size(dstate); ++index) {
        auto coordinate = coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        storage.dstate_b[bf16_tsm_offset(column, row, kDim)] =
            Element(dstate(index));
      }
      // With a 128-thread consumer and a 64-column tile, every thread keeps
      // the same token column across the strided loop.  Evaluate exp(g) once
      // per thread instead of once per Q element.
      const int gate_column = thread & (kChunk - 1);
      const float scaled_gate = params.scale
          * expf(params.g[gate_base + gate_column]);
      for (int linear = thread; linear < kDim * kChunk; linear += 128) {
        const int row = linear / kChunk;
        const int column = linear - row * kChunk;
        const int source = (((batch * params.tokens + token_base + column)
            * params.q_heads + q_head) * kDim) + row;
        storage.state_a[bf16_tsm_offset(row, column, kDim)] = Element(
            params.q[source] * scaled_gate);
      }
      for (int linear = thread; linear < kChunk * kDim; linear += 128) {
        const int row = linear / kDim;
        const int column = linear - row * kDim;
        const int source = (((batch * params.tokens + token_base + row)
            * params.value_heads + head) * kDim) + column;
        storage.state_b[bf16_tsm_offset(column, row, kDim)] =
            Element(params.do_value[source]);
      }
      __syncthreads();  // 00

      const float g_last = params.g[gate_base + kChunk - 1];
      const float state_decay = expf(g_last);
      for (int index = 0; index < size(dstate); ++index) {
        dstate(index) *= state_decay;
      }
      mma_from_tsm<Tile12812864>(
          storage.state_a.data(), storage.state_b.data(),
          thread, warp, dstate, false);
      __syncthreads();  // 01
      __syncthreads();  // 02: K publishes dV into state_b.

      for (int linear = thread; linear < kDim * kChunk; linear += 128) {
        const int row = linear / kChunk;
        const int column = linear - row * kChunk;
        const int source = (((batch * params.tokens + token_base + column)
            * params.value_heads + head) * kDim) + row;
        storage.state_a[bf16_tsm_offset(row, column, kDim)] =
            Element(-params.w[source]);
      }
      __syncthreads();  // 03
      mma_from_tsm<Tile12812864>(
          storage.state_a.data(), storage.state_b.data(),
          thread, warp, dstate, false);
      __syncthreads();  // 04

      __syncthreads();  // 05
      __syncthreads();  // 06
      __syncthreads();  // 07
      __syncthreads();  // 08
      __syncthreads();  // 09
      __syncthreads();  // 10
      __syncthreads();  // 11
      __syncthreads();  // 12
      __syncthreads();  // 13
      __syncthreads();  // 14
      __syncthreads();  // 15
      __syncthreads();  // 16
    }

    for (int index = 0; index < size(dstate); ++index) {
      auto coordinate = coordinates(index);
      const int row = get<0>(coordinate);
      const int column = get<1>(coordinate);
      params.initial_state_grad[state_base + row * kDim + column] =
          dstate(index);
    }
  }

  static __attribute__((__noinline__)) __device__ void run_k(
      Params const& params, SharedStorage& storage) {
    const int group = blockIdx.x;
    const int batch = group / params.value_heads;
    const int head = group - batch * params.value_heads;
    const int q_head = head / (params.value_heads / params.q_heads);
    const int chunks = params.tokens / kChunk;
    const int thread = threadIdx.x - 128;
    const int warp = thread / 32;
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread);
    Tensor output = partition_fragment_C(
        tiled_mma, Shape<Int<64>, Int<128>>{});
    Tensor coordinates = thr_mma.partition_C(
        make_identity_tensor(Shape<Int<64>, Int<128>>{}));

    for (int chunk = chunks - 1; chunk >= 0; --chunk) {
      const int token_base = chunk * kChunk;
      const int gate_base = (group * chunks + chunk) * kChunk;
      const int value_base = (((batch * params.tokens + token_base)
          * params.value_heads + head) * kDim);
      const int q_base = (((batch * params.tokens + token_base)
          * params.q_heads + q_head) * kDim);
      const int value_stride = params.value_heads * kDim;
      const int q_stride = params.q_heads * kDim;
      const int history_base = (((batch * chunks + chunk)
          * params.value_heads + head) * kDim * kDim);

      for (int linear = thread; linear < kChunk * kDim; linear += 128) {
        const int row = linear / kDim;
        const int column = linear - row * kDim;
        storage.state_k_a[bf16_tsm_offset(row, column, kChunk)] = Element(
            params.k[q_base + row * q_stride + column]);
      }
      __syncthreads();  // 00
      mma_from_tsm<Tile64128128>(
          storage.state_k_a.data(), storage.dstate_b.data(),
          thread, warp, output, true);
      __syncthreads();  // 01
      for (int index = 0; index < size(output); ++index) {
        const int row = get<0>(coordinates(index));
        output(index) *= storage.decay_to_last[row];
      }

      // dV' += Pg^T @ dO. A publishes transposed Pg in dp_a while S keeps
      // dO resident in state_b, matching the official fused barrier 01/02
      // handoff and removing the standalone chunk-dV kernel.
      mma_from_tsm<Tile6412864>(
          storage.dp_a.data(), storage.state_b.data(),
          thread, warp, output, false);

      const float g_last = storage.gate[kChunk - 1];
      for (int index = 0; index < size(output); ++index) {
        auto coordinate = coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        const int destination = value_base + row * value_stride + column;
        const float value = output(index);
        params.dv_output[destination] = value;
        storage.state_b[bf16_tsm_offset(column, row, kDim)] = Element(value);
      }
      __syncthreads();  // 02

      for (int linear = thread; linear < kChunk * kDim; linear += 128) {
        const int row = linear / kDim;
        const int column = linear - row * kDim;
        storage.direct_a[bf16_tsm_offset(row, column, kChunk)] = Element(
            params.do_value[value_base + row * value_stride + column]);
      }
      for (int linear = thread; linear < kDim * kDim; linear += 128) {
        const int row = linear / kDim;
        const int column = linear - row * kDim;
        storage.direct_b[bf16_tsm_offset(row, column, kDim)] = Element(
            params.history[history_base + row * kDim + column]);
      }
      __syncthreads();  // 03
      mma_from_tsm<Tile64128128>(
          storage.direct_a.data(), storage.direct_b.data(),
          thread, warp, output, true);
      __syncthreads();  // 04
      for (int index = 0; index < size(output); ++index) {
        auto coordinate = coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        params.dq[value_base + row * value_stride + column] = output(index)
            * params.scale * expf(storage.gate[row]);
      }
      __syncthreads();  // 05

      for (int linear = thread; linear < kChunk * kDim; linear += 128) {
        const int row = linear / kDim;
        const int column = linear - row * kDim;
        storage.direct_a[bf16_tsm_offset(row, column, kChunk)] = Element(
            params.dv_output[value_base + row * value_stride + column]);
      }
      __syncthreads();  // 06
      mma_from_tsm<Tile64128128>(
          storage.direct_a.data(), storage.direct_b.data(),
          thread, warp, output, true);
      __syncthreads();  // 07
      for (int index = 0; index < size(output); ++index) {
        auto coordinate = coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        params.dw[value_base + row * value_stride + column] = -output(index);
      }
      __syncthreads();  // 08

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
            storage.dstate_b[bf16_tsm_offset(column, row, kDim)];
      }
      __syncthreads();  // 09
      mma_from_tsm<Tile64128128>(
          storage.direct_a.data(), storage.direct_b.data(),
          thread, warp, output, true);
      __syncthreads();  // 10
      for (int index = 0; index < size(output); ++index) {
        auto coordinate = coordinates(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        params.dk[value_base + row * value_stride + column] = output(index)
            * stable_exp_difference(g_last - storage.gate[row]);
      }
      __syncthreads();  // 11

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
        float state_sum = 0.0f;
        for (int linear = thread; linear < kDim * kDim; linear += kChunk) {
          const int row = linear / kDim;
          const int column = linear - row * kDim;
          state_sum += params.history[history_base + linear]
              * float(storage.dstate_b[
                    bf16_tsm_offset(column, row, kDim)]);
        }
        storage.dg[thread] = q_dq - k_dk + row_sum - column_sum;
        storage.k_reduction[thread] = k_dk;
        storage.state_reduction[thread] = state_sum;
      }
      __syncthreads();  // 12
      if (thread == 0) {
        float total = 0.0f;
        for (int row = 0; row < kChunk; ++row) {
          total += storage.k_reduction[row]
              + expf(g_last) * storage.state_reduction[row];
        }
        storage.dg_last = total;
      }
      __syncthreads();  // 13
      __syncthreads();  // 14
      __syncthreads();  // 15
      if (thread < kChunk) {
        const int destination = ((batch * params.tokens + token_base + thread)
            * params.value_heads + head);
        params.dg[destination] = storage.dg[thread]
            + (thread == kChunk - 1 ? storage.dg_last : 0.0f);
      }
      __syncthreads();  // 16
    }
  }

  static __attribute__((__noinline__)) __device__ void run_a(
      Params const& params, SharedStorage& storage) {
    const int group = blockIdx.x;
    const int batch = group / params.value_heads;
    const int head = group - batch * params.value_heads;
    const int q_head = head / (params.value_heads / params.q_heads);
    const int chunks = params.tokens / kChunk;
    const int thread = threadIdx.x - 256;
    const int warp = thread / 32;
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread);

    for (int chunk = chunks - 1; chunk >= 0; --chunk) {
      const int token_base = chunk * kChunk;
      const int value_base = (((batch * params.tokens + token_base)
          * params.value_heads + head) * kDim);
      const int q_base = (((batch * params.tokens + token_base)
          * params.q_heads + q_head) * kDim);
      const int value_stride = params.value_heads * kDim;
      const int q_stride = params.q_heads * kDim;

      {
        Tensor output64 = partition_fragment_C(
            tiled_mma, Shape<Int<64>, Int<64>>{});
        Tensor coordinates64 = thr_mma.partition_C(
            make_identity_tensor(Shape<Int<64>, Int<64>>{}));
        for (int linear = thread; linear < kChunk * kDim; linear += 128) {
          const int row = linear / kDim;
          const int column = linear - row * kDim;
          storage.dp_a[bf16_tsm_offset(row, column, kChunk)] = Element(
              params.q[q_base + row * q_stride + column]);
          storage.dp_b[bf16_tsm_offset(row, column, kChunk)] = Element(
              params.k[q_base + row * q_stride + column]);
        }
        __syncthreads();  // 00
        mma_from_tsm<Tile6464128>(
            storage.dp_a.data(), storage.dp_b.data(),
            thread, warp, output64, true);
        for (int index = 0; index < size(output64); ++index) {
          auto coordinate = coordinates64(index);
          const int row = get<0>(coordinate);
          const int column = get<1>(coordinate);
          const int offset = bf16_tsm_offset(row, column, kChunk);
          const float decay = column <= row
              ? stable_exp_difference(
                    storage.gate[row] - storage.gate[column])
              : 0.0f;
          const Element pg = Element(
              output64(index) * params.scale * decay);
          storage.gate_product[offset] = float(pg);
          storage.dp_a[bf16_tsm_offset(column, row, kChunk)] = pg;
        }
        __syncthreads();  // 01
        // K consumes transposed Pg before dp_a is reused for dO.
        __syncthreads();  // 02

        for (int linear = thread; linear < kChunk * kDim; linear += 128) {
          const int row = linear / kDim;
          const int column = linear - row * kDim;
          storage.dp_a[bf16_tsm_offset(row, column, kChunk)] = Element(
              params.do_value[value_base + row * value_stride + column]);
          storage.dp_b[bf16_tsm_offset(row, column, kChunk)] = Element(
              params.v_corrected[
                  value_base + row * value_stride + column]);
        }
        __syncthreads();  // 03
        mma_from_tsm<Tile6464128>(
            storage.dp_a.data(), storage.dp_b.data(),
            thread, warp, output64, true);
        __syncthreads();  // 04
        for (int index = 0; index < size(output64); ++index) {
          auto coordinate = coordinates64(index);
          const int row = get<0>(coordinate);
          const int column = get<1>(coordinate);
          const int offset = bf16_tsm_offset(row, column, kChunk);
          const float decay = column <= row
              ? stable_exp_difference(
                    storage.gate[row] - storage.gate[column])
              : 0.0f;
          storage.gate_product[offset] *= output64(index);
          storage.dp[offset] = Element(
              output64(index) * params.scale * decay);
        }
        __syncthreads();  // 05
      }

      __syncthreads();  // 06
      __syncthreads();  // 07
      __syncthreads();  // 08

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
      __syncthreads();  // 09

      Tensor output128 = partition_fragment_C(
          tiled_mma, Shape<Int<64>, Int<128>>{});
      Tensor coordinates128 = thr_mma.partition_C(
          make_identity_tensor(Shape<Int<64>, Int<128>>{}));
      mma_from_tsm<Tile6412864>(
          storage.dp_a.data(), storage.dp_b.data(),
          thread, warp, output128, true);
      __syncthreads();  // 10
      __syncthreads();  // 11
      __syncthreads();  // 12
      for (int index = 0; index < size(output128); ++index) {
        auto coordinate = coordinates128(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        params.dq[value_base + row * value_stride + column] +=
            output128(index);
      }
      __syncthreads();  // 13

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
      __syncthreads();  // 14
      mma_from_tsm<Tile6412864>(
          storage.dp_a.data(), storage.dp_b.data(),
          thread, warp, output128, true);
      __syncthreads();  // 15
      for (int index = 0; index < size(output128); ++index) {
        auto coordinate = coordinates128(index);
        const int row = get<0>(coordinate);
        const int column = get<1>(coordinate);
        params.dk[value_base + row * value_stride + column] +=
            output128(index);
      }
      __syncthreads();  // 16
    }
  }

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    const int group = blockIdx.x;
    if (group >= params.batch_size * params.value_heads) {
      return;
    }
    auto& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    const int consumer = threadIdx.x / 128;
    if (consumer == 0) {
      run_s(params, storage);
    } else if (consumer == 1) {
      run_k(params, storage);
    } else {
      run_a(params, storage);
    }
  }
};
