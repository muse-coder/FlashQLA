#include <hggc_runtime.h>
#include <stdint.h>


__global__ void flash_qla_prepare_dqkwg_head_major_kernel(
    const float* q,
    const float* k,
    const float* v,
    const float* do_value,
    const float* dv,
    const float* g,
    float* q_head,
    float* k_head,
    float* v_head,
    float* do_head,
    float* dv_head,
    float* g_head,
    int total,
    int tokens,
    int q_heads,
    int value_heads,
    int chunks) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += blockDim.x * gridDim.x) {
    const int vector_column = index % 32;
    const int row_group = index / 32;
    const int row = row_group % 64;
    const int head_group = row_group / 64;
    const int head = head_group % value_heads;
    const int chunk_group = head_group / value_heads;
    const int chunk = chunk_group % chunks;
    const int batch = chunk_group / chunks;
    const int token = chunk * 64 + row;
    const int q_head_index = head / (value_heads / q_heads);
    const int q_source = ((batch * tokens + token) * q_heads
        + q_head_index) * 32 + vector_column;
    const int value_source = ((batch * tokens + token) * value_heads
        + head) * 32 + vector_column;
    reinterpret_cast<float4*>(q_head)[index] =
        reinterpret_cast<const float4*>(q)[q_source];
    reinterpret_cast<float4*>(k_head)[index] =
        reinterpret_cast<const float4*>(k)[q_source];
    reinterpret_cast<float4*>(v_head)[index] =
        reinterpret_cast<const float4*>(v)[value_source];
    reinterpret_cast<float4*>(do_head)[index] =
        reinterpret_cast<const float4*>(do_value)[value_source];
    reinterpret_cast<float4*>(dv_head)[index] =
        reinterpret_cast<const float4*>(dv)[value_source];
    if (vector_column == 0) {
      g_head[row_group] = g[(batch * tokens + token) * value_heads + head];
    }
  }
}


extern "C" void launch_prepare_dqkwg_head_major(
    void* q,
    void* k,
    void* v,
    void* do_value,
    void* dv,
    void* g,
    void* q_head,
    void* k_head,
    void* v_head,
    void* do_head,
    void* dv_head,
    void* g_head,
    int batch_size,
    int tokens,
    int q_heads,
    int value_heads,
    void* raw_stream,
    int& result) {
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  const int chunks = tokens / 64;
  const int total = batch_size * tokens * value_heads * 32;
  const int blocks = (total + 255) / 256;
  flash_qla_prepare_dqkwg_head_major_kernel<<<blocks, 256, 0, stream>>>(
      reinterpret_cast<const float*>(q),
      reinterpret_cast<const float*>(k),
      reinterpret_cast<const float*>(v),
      reinterpret_cast<const float*>(do_value),
      reinterpret_cast<const float*>(dv),
      reinterpret_cast<const float*>(g),
      reinterpret_cast<float*>(q_head),
      reinterpret_cast<float*>(k_head),
      reinterpret_cast<float*>(v_head),
      reinterpret_cast<float*>(do_head),
      reinterpret_cast<float*>(dv_head),
      reinterpret_cast<float*>(g_head),
      total,
      tokens,
      q_heads,
      value_heads,
      chunks);
  result = int(hggcGetLastError());
}


__global__ void flash_qla_grouped_state_dv_update_kernel(
    float* dv,
    const float* update,
    const float* decay,
    int total,
    int chunks,
    int q_heads,
    int value_heads,
    int chunk) {
  const int head_repeat = value_heads / q_heads;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += blockDim.x * gridDim.x) {
    const int vector_column = index % 32;
    const int row_group = index / 32;
    const int head = row_group % value_heads;
    const int token_group = row_group / value_heads;
    const int row = token_group % 64;
    const int batch = token_group / 64;
    const int q_head = head / head_repeat;
    const int repeat = head - q_head * head_repeat;
    const int source = ((((batch * q_heads + q_head) * head_repeat
        + repeat) * 64 + row) * 32) + vector_column;
    const int destination = ((((batch * chunks + chunk) * 64 + row)
        * value_heads + head) * 32) + vector_column;
    const int decay_index = (((batch * chunks + chunk) * 64 + row)
        * value_heads) + head;
    const float factor = decay[decay_index];
    float4 value = reinterpret_cast<float4*>(dv)[destination];
    const float4 delta = reinterpret_cast<const float4*>(update)[source];
    value.x += delta.x * factor;
    value.y += delta.y * factor;
    value.z += delta.z * factor;
    value.w += delta.w * factor;
    reinterpret_cast<float4*>(dv)[destination] = value;
  }
}


extern "C" void launch_grouped_state_dv_update(
    void* dv,
    void* update,
    void* decay,
    int batch_size,
    int chunks,
    int q_heads,
    int value_heads,
    int chunk,
    void* raw_stream,
    int& result) {
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  const int total = batch_size * 64 * value_heads * 32;
  const int blocks = (total + 255) / 256;
  flash_qla_grouped_state_dv_update_kernel<<<blocks, 256, 0, stream>>>(
      reinterpret_cast<float*>(dv),
      reinterpret_cast<const float*>(update),
      reinterpret_cast<const float*>(decay),
      total,
      chunks,
      q_heads,
      value_heads,
      chunk);
  result = int(hggcGetLastError());
}


__global__ void flash_qla_prepare_gate_beta_kernel(
    const float* g,
    const float* beta,
    float* g_cumsum,
    float* beta_chunks,
    int batch_size,
    int tokens,
    int heads,
    int chunks) {
  __shared__ float gate[64];
  const int group = blockIdx.x;
  const int row = threadIdx.x;
  const int chunk = group % chunks;
  const int head_group = group / chunks;
  const int head = head_group % heads;
  const int batch = head_group / heads;
  if (batch >= batch_size || row >= 64) {
    return;
  }
  const int source = ((batch * tokens + chunk * 64 + row) * heads) + head;
  gate[row] = g[source];
  beta_chunks[group * 64 + row] = beta[source];
  __syncthreads();
  for (int offset = 1; offset < 64; offset *= 2) {
    const float addend = row >= offset ? gate[row - offset] : 0.0f;
    __syncthreads();
    gate[row] += addend;
    __syncthreads();
  }
  g_cumsum[group * 64 + row] = gate[row];
}


extern "C" void launch_prepare_gate_beta(
    void* g,
    void* beta,
    void* g_cumsum,
    void* beta_chunks,
    int batch_size,
    int tokens,
    int heads,
    void* raw_stream,
    int& result) {
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  const int chunks = tokens / 64;
  flash_qla_prepare_gate_beta_kernel<<<
      batch_size * heads * chunks, 64, 0, stream>>>(
      reinterpret_cast<const float*>(g),
      reinterpret_cast<const float*>(beta),
      reinterpret_cast<float*>(g_cumsum),
      reinterpret_cast<float*>(beta_chunks),
      batch_size,
      tokens,
      heads,
      chunks);
  result = int(hggcGetLastError());
}


__global__ void flash_qla_reverse_chunk_cumsum_kernel(
    const float* input,
    float* output,
    int groups,
    int tokens,
    int heads,
    int chunks) {
  for (int group = blockIdx.x * blockDim.x + threadIdx.x;
       group < groups;
       group += blockDim.x * gridDim.x) {
    const int head = group % heads;
    const int chunk_group = group / heads;
    const int chunk = chunk_group % chunks;
    const int batch = chunk_group / chunks;
    float accumulated = 0.0f;
    for (int row = 63; row >= 0; --row) {
      const int index = ((batch * tokens + chunk * 64 + row) * heads) + head;
      accumulated += input[index];
      output[index] = accumulated;
    }
  }
}


extern "C" void launch_reverse_chunk_cumsum(
    void* input,
    void* output,
    int batch_size,
    int tokens,
    int heads,
    void* raw_stream,
    int& result) {
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  const int chunks = tokens / 64;
  const int groups = batch_size * chunks * heads;
  const int blocks = (groups + 255) / 256;
  flash_qla_reverse_chunk_cumsum_kernel<<<
      blocks, 256, 0, stream>>>(
      reinterpret_cast<const float*>(input),
      reinterpret_cast<float*>(output),
      groups,
      tokens,
      heads,
      chunks);
  result = int(hggcGetLastError());
}


__global__ void flash_qla_warmup_counts_kernel(
    const float* g,
    int* counts,
    uint8_t* fallback,
    int segments,
    int tokens,
    int heads,
    int chunk_size,
    float threshold) {
  const int segment = blockIdx.x;
  const int head = threadIdx.x;
  if (segment >= segments || head >= heads) {
    return;
  }

  const int chunks = tokens / chunk_size;
  float accumulated = 0.0f;
  int warmup = chunks;
  uint8_t use_fallback = 1;
  for (int reverse_chunk = 0; reverse_chunk < chunks; ++reverse_chunk) {
    const int chunk = chunks - 1 - reverse_chunk;
    float chunk_decay = 0.0f;
    const int token_base = segment * tokens + chunk * chunk_size;
    for (int token = 0; token < chunk_size; ++token) {
      chunk_decay += g[(token_base + token) * heads + head];
    }
    accumulated += chunk_decay;
    if (use_fallback && accumulated < threshold) {
      warmup = reverse_chunk + 1;
      use_fallback = 0;
    }
  }

  const int output = segment * heads + head;
  counts[output] = warmup;
  fallback[output] = use_fallback;
}


extern "C" void launch_warmup_counts(
    void* g,
    void* counts,
    void* fallback,
    int segments,
    int tokens,
    int heads,
    int chunk_size,
    float threshold,
    void* raw_stream,
    int& result) {
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  int threads = ((heads + 31) / 32) * 32;
  flash_qla_warmup_counts_kernel<<<segments, threads, 0, stream>>>(
      reinterpret_cast<const float*>(g),
      reinterpret_cast<int*>(counts),
      reinterpret_cast<uint8_t*>(fallback),
      segments,
      tokens,
      heads,
      chunk_size,
      threshold);
  result = int(hggcGetLastError());
}


__global__ void flash_qla_chunk_coefficients_kernel(
    const float* g_cumsum,
    const float* kkt,
    const float* qk,
    float* weighted_kkt,
    float* attention,
    float* gamma,
    float* reverse_decay,
    int groups,
    int chunk_size) {
  const int group = blockIdx.x;
  if (group >= groups) {
    return;
  }
  const int matrix_elements = chunk_size * chunk_size;
  const int matrix_base = group * matrix_elements;
  const int gate_base = group * chunk_size;
  for (int linear = threadIdx.x; linear < matrix_elements; linear += blockDim.x) {
    const int row = linear / chunk_size;
    const int column = linear - row * chunk_size;
    const float row_gate = g_cumsum[gate_base + row];
    if (column <= row) {
      const float decay = expf(row_gate - g_cumsum[gate_base + column]);
      weighted_kkt[matrix_base + linear] = kkt[matrix_base + linear] * decay;
      attention[matrix_base + linear] = qk[matrix_base + linear] * decay;
    } else {
      weighted_kkt[matrix_base + linear] = 0.0f;
      attention[matrix_base + linear] = 0.0f;
    }
    if (column == 0) {
      gamma[gate_base + row] = expf(row_gate);
      reverse_decay[gate_base + row] = expf(
          g_cumsum[gate_base + chunk_size - 1] - row_gate);
    }
  }
}


extern "C" void launch_chunk_coefficients(
    void* g_cumsum,
    void* kkt,
    void* qk,
    void* weighted_kkt,
    void* attention,
    void* gamma,
    void* reverse_decay,
    int groups,
    int chunk_size,
    void* raw_stream,
    int& result) {
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  flash_qla_chunk_coefficients_kernel<<<groups, 256, 0, stream>>>(
      reinterpret_cast<const float*>(g_cumsum),
      reinterpret_cast<const float*>(kkt),
      reinterpret_cast<const float*>(qk),
      reinterpret_cast<float*>(weighted_kkt),
      reinterpret_cast<float*>(attention),
      reinterpret_cast<float*>(gamma),
      reinterpret_cast<float*>(reverse_decay),
      groups,
      chunk_size);
  result = int(hggcGetLastError());
}


__device__ inline int chunk_gate_offset(
    int dense_index,
    int heads,
    int chunks,
    int chunk_size,
    int width,
    int chunk) {
  const int group_elements = chunk_size * width;
  const int group = dense_index / group_elements;
  const int row = (dense_index - group * group_elements) / width;
  const int batch = group / heads;
  const int head = group - batch * heads;
  return ((batch * heads + head) * chunks + chunk) * chunk_size + row;
}


__global__ void flash_qla_prepare_w_kernel(
    const float* u,
    const float* v_chunks,
    const float* beta_chunks,
    const float* gamma_chunks,
    float* w,
    int total,
    int heads,
    int chunks,
    int chunk_size,
    int value_dim,
    int chunk) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += blockDim.x * gridDim.x) {
    const int gate = chunk_gate_offset(
        index, heads, chunks, chunk_size, value_dim, chunk);
    const int column = index % value_dim;
    const int row = (index / value_dim) % chunk_size;
    const int group = index / (chunk_size * value_dim);
    const int batch = group / heads;
    const int head = group - batch * heads;
    const int value =
        ((((batch * heads + head) * chunks + chunk) * chunk_size + row)
         * value_dim) + column;
    w[index] = beta_chunks[gate]
        * (v_chunks[value] - gamma_chunks[gate] * u[index]);
  }
}


extern "C" void launch_prepare_w(
    void* u,
    void* v_chunks,
    void* beta_chunks,
    void* gamma_chunks,
    void* w,
    int batch,
    int heads,
    int chunks,
    int chunk_size,
    int value_dim,
    int chunk,
    void* raw_stream,
    int& result) {
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  const int total = batch * heads * chunk_size * value_dim;
  const int blocks = (total + 255) / 256;
  flash_qla_prepare_w_kernel<<<blocks, 256, 0, stream>>>(
      reinterpret_cast<const float*>(u),
      reinterpret_cast<const float*>(v_chunks),
      reinterpret_cast<const float*>(beta_chunks),
      reinterpret_cast<const float*>(gamma_chunks),
      reinterpret_cast<float*>(w),
      total,
      heads,
      chunks,
      chunk_size,
      value_dim,
      chunk);
  result = int(hggcGetLastError());
}


__global__ void flash_qla_combine_output_kernel(
    const float* q_state,
    const float* attended,
    const float* gamma_chunks,
    float* output,
    int total,
    int heads,
    int chunks,
    int chunk_size,
    int value_dim,
    int chunk,
    float scale) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += blockDim.x * gridDim.x) {
    const int gate = chunk_gate_offset(
        index, heads, chunks, chunk_size, value_dim, chunk);
    output[index] = gamma_chunks[gate] * q_state[index] * scale
        + attended[index];
  }
}


extern "C" void launch_combine_output(
    void* q_state,
    void* attended,
    void* gamma_chunks,
    void* output,
    int batch,
    int heads,
    int chunks,
    int chunk_size,
    int value_dim,
    int chunk,
    float scale,
    void* raw_stream,
    int& result) {
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  const int total = batch * heads * chunk_size * value_dim;
  const int blocks = (total + 255) / 256;
  flash_qla_combine_output_kernel<<<blocks, 256, 0, stream>>>(
      reinterpret_cast<const float*>(q_state),
      reinterpret_cast<const float*>(attended),
      reinterpret_cast<const float*>(gamma_chunks),
      reinterpret_cast<float*>(output),
      total,
      heads,
      chunks,
      chunk_size,
      value_dim,
      chunk,
      scale);
  result = int(hggcGetLastError());
}


__global__ void flash_qla_decay_values_kernel(
    const float* values,
    const float* reverse_decay_chunks,
    float* output,
    int total,
    int heads,
    int chunks,
    int chunk_size,
    int value_dim,
    int chunk) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += blockDim.x * gridDim.x) {
    const int gate = chunk_gate_offset(
        index, heads, chunks, chunk_size, value_dim, chunk);
    output[index] = reverse_decay_chunks[gate] * values[index];
  }
}


extern "C" void launch_decay_values(
    void* values,
    void* reverse_decay_chunks,
    void* output,
    int batch,
    int heads,
    int chunks,
    int chunk_size,
    int value_dim,
    int chunk,
    void* raw_stream,
    int& result) {
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  const int total = batch * heads * chunk_size * value_dim;
  const int blocks = (total + 255) / 256;
  flash_qla_decay_values_kernel<<<blocks, 256, 0, stream>>>(
      reinterpret_cast<const float*>(values),
      reinterpret_cast<const float*>(reverse_decay_chunks),
      reinterpret_cast<float*>(output),
      total,
      heads,
      chunks,
      chunk_size,
      value_dim,
      chunk);
  result = int(hggcGetLastError());
}


__global__ void flash_qla_update_state_kernel(
    const float* state,
    const float* delta,
    const float* gamma_last_chunks,
    float* output,
    int total,
    int heads,
    int chunks,
    int matrix_elements,
    int chunk,
    int scale_delta) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += blockDim.x * gridDim.x) {
    const int group = index / matrix_elements;
    const int batch = group / heads;
    const int head = group - batch * heads;
    const int gate = (batch * heads + head) * chunks + chunk;
    const float gamma = gamma_last_chunks[gate];
    output[index] = scale_delta
        ? gamma * (state[index] + delta[index])
        : gamma * state[index] + delta[index];
  }
}


extern "C" void launch_update_state(
    void* state,
    void* delta,
    void* gamma_last_chunks,
    void* output,
    int batch,
    int heads,
    int chunks,
    int key_dim,
    int value_dim,
    int chunk,
    int scale_delta,
    void* raw_stream,
    int& result) {
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  const int matrix_elements = key_dim * value_dim;
  const int total = batch * heads * matrix_elements;
  const int blocks = (total + 255) / 256;
  flash_qla_update_state_kernel<<<blocks, 256, 0, stream>>>(
      reinterpret_cast<const float*>(state),
      reinterpret_cast<const float*>(delta),
      reinterpret_cast<const float*>(gamma_last_chunks),
      reinterpret_cast<float*>(output),
      total,
      heads,
      chunks,
      matrix_elements,
      chunk,
      scale_delta);
  result = int(hggcGetLastError());
}


template <bool StoreHistory>
__global__ void flash_qla_reverse_state_update_kernel(
    const float* state,
    const float* inter,
    const float* product,
    const float* decay,
    float* output,
    float* history,
    int total,
    int heads,
    int chunks,
    int matrix_elements,
    int chunk) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += blockDim.x * gridDim.x) {
    const int group = index / matrix_elements;
    const int matrix_index = index - group * matrix_elements;
    const int batch = group / heads;
    const int head = group - batch * heads;
    const int chunk_group = (batch * chunks + chunk) * heads + head;
    const int chunk_index = chunk_group * matrix_elements + matrix_index;
    const float current = state[index];
    if constexpr (StoreHistory) {
      history[chunk_index] = current;
    }
    output[index] = decay[chunk_group] * current
        + inter[chunk_index] - product[index];
  }
}


extern "C" void launch_reverse_state_update(
    void* state,
    void* inter,
    void* product,
    void* decay,
    void* output,
    int batch,
    int heads,
    int chunks,
    int key_dim,
    int value_dim,
    int chunk,
    void* raw_stream,
    int& result) {
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  const int matrix_elements = key_dim * value_dim;
  const int total = batch * heads * matrix_elements;
  const int blocks = (total + 255) / 256;
  flash_qla_reverse_state_update_kernel<false><<<blocks, 256, 0, stream>>>(
      reinterpret_cast<const float*>(state),
      reinterpret_cast<const float*>(inter),
      reinterpret_cast<const float*>(product),
      reinterpret_cast<const float*>(decay),
      reinterpret_cast<float*>(output),
      nullptr,
      total,
      heads,
      chunks,
      matrix_elements,
      chunk);
  result = int(hggcGetLastError());
}


// Official S-consumer handoff: publish the current chunk dH and advance the
// reverse recurrence in the same launch. Updating state in place mirrors the
// register-resident recurrence and avoids a Python list/stack materialization.
extern "C" void launch_reverse_state_update_store_inplace(
    void* state,
    void* inter,
    void* product,
    void* decay,
    void* history,
    int batch,
    int heads,
    int chunks,
    int key_dim,
    int value_dim,
    int chunk,
    void* raw_stream,
    int& result) {
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  const int matrix_elements = key_dim * value_dim;
  const int total = batch * heads * matrix_elements;
  const int blocks = (total + 255) / 256;
  flash_qla_reverse_state_update_kernel<true><<<blocks, 256, 0, stream>>>(
      reinterpret_cast<const float*>(state),
      reinterpret_cast<const float*>(inter),
      reinterpret_cast<const float*>(product),
      reinterpret_cast<const float*>(decay),
      reinterpret_cast<float*>(state),
      reinterpret_cast<float*>(history),
      total,
      heads,
      chunks,
      matrix_elements,
      chunk);
  result = int(hggcGetLastError());
}


__global__ void flash_qla_wy_decay_da_kernel(
    float* dA,
    const float* g,
    int total) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += blockDim.x * gridDim.x) {
    const int matrix_index = index % (64 * 64);
    const int group = index / (64 * 64);
    const int row = matrix_index / 64;
    const int column = matrix_index - row * 64;
    float value = 0.0f;
    if (row > column) {
      const int gate_base = group * 64;
      value = -dA[index]
          * expf(g[gate_base + row] - g[gate_base + column]);
    }
    dA[index] = value;
  }
}


extern "C" void launch_wy_decay_da(
    void* dA,
    void* g,
    int groups,
    void* raw_stream,
    int& result) {
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  const int total = groups * 64 * 64;
  const int blocks = (total + 255) / 256;
  flash_qla_wy_decay_da_kernel<<<blocks, 256, 0, stream>>>(
      reinterpret_cast<float*>(dA),
      reinterpret_cast<const float*>(g),
      total);
  result = int(hggcGetLastError());
}


__global__ void flash_qla_dqkwg_decay_dp_kernel(
    float* dp,
    const float* g,
    int total,
    float scale) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += blockDim.x * gridDim.x) {
    const int matrix_index = index % (64 * 64);
    const int group = index / (64 * 64);
    const int row = matrix_index / 64;
    const int column = matrix_index - row * 64;
    float value = 0.0f;
    if (row >= column) {
      const int gate_base = group * 64;
      value = dp[index] * scale
          * expf(g[gate_base + row] - g[gate_base + column]);
    }
    dp[index] = value;
  }
}


extern "C" void launch_dqkwg_decay_dp(
    void* dp,
    void* g,
    int groups,
    float scale,
    void* raw_stream,
    int& result) {
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  const int total = groups * 64 * 64;
  const int blocks = (total + 255) / 256;
  flash_qla_dqkwg_decay_dp_kernel<<<blocks, 256, 0, stream>>>(
      reinterpret_cast<float*>(dp),
      reinterpret_cast<const float*>(g),
      total,
      scale);
  result = int(hggcGetLastError());
}


__global__ void flash_qla_dqkwg_gate_product_kernel(
    const float* dp,
    const float* p,
    float* dg,
    int rows) {
  for (int output = blockIdx.x * blockDim.x + threadIdx.x;
       output < rows;
       output += blockDim.x * gridDim.x) {
    const int group = output / 64;
    const int row = output - group * 64;
    const int matrix_base = group * 64 * 64;
    float row_sum = 0.0f;
    float column_sum = 0.0f;
    for (int column = 0; column < 64; ++column) {
      const int row_index = matrix_base + row * 64 + column;
      const int column_index = matrix_base + column * 64 + row;
      row_sum += dp[row_index] * p[row_index];
      column_sum += dp[column_index] * p[column_index];
    }
    dg[output] += row_sum - column_sum;
  }
}


extern "C" void launch_dqkwg_gate_product(
    void* dp,
    void* p,
    void* dg,
    int groups,
    void* raw_stream,
    int& result) {
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  const int rows = groups * 64;
  const int blocks = (rows + 255) / 256;
  flash_qla_dqkwg_gate_product_kernel<<<blocks, 256, 0, stream>>>(
      reinterpret_cast<const float*>(dp),
      reinterpret_cast<const float*>(p),
      reinterpret_cast<float*>(dg),
      rows);
  result = int(hggcGetLastError());
}


__global__ void flash_qla_kkt_system_kernel(
    const float* gram,
    const float* beta,
    float* lower,
    float* identity,
    int groups,
    int chunk_size) {
  const int group = blockIdx.x;
  const int elements = chunk_size * chunk_size;
  const int matrix_base = group * elements;
  const int vector_base = group * chunk_size;
  for (int linear = threadIdx.x; linear < elements; linear += blockDim.x) {
    const int row = linear / chunk_size;
    const int column = linear - row * chunk_size;
    const bool diagonal = row == column;
    identity[matrix_base + linear] = diagonal ? 1.0f : 0.0f;
    if (row > column) {
      lower[matrix_base + linear] =
          beta[vector_base + row] * gram[matrix_base + linear];
    } else {
      lower[matrix_base + linear] = diagonal ? 1.0f : 0.0f;
    }
  }
}


extern "C" void launch_kkt_system(
    void* gram,
    void* beta,
    void* lower,
    void* identity,
    int groups,
    int chunk_size,
    void* raw_stream,
    int& result) {
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  flash_qla_kkt_system_kernel<<<groups, 256, 0, stream>>>(
      reinterpret_cast<const float*>(gram),
      reinterpret_cast<const float*>(beta),
      reinterpret_cast<float*>(lower),
      reinterpret_cast<float*>(identity),
      groups,
      chunk_size);
  result = int(hggcGetLastError());
}


__global__ void flash_qla_kkt_inverse_kernel(
    const float* gram,
    const float* beta,
    float* output,
    int groups) {
  constexpr int chunk_size = 64;
  __shared__ float inverse[chunk_size * chunk_size];
  const int group = blockIdx.x;
  if (group >= groups) {
    return;
  }
  const int column = threadIdx.x;
  const int matrix_base = group * chunk_size * chunk_size;
  const int vector_base = group * chunk_size;
  for (int linear = column; linear < chunk_size * chunk_size;
       linear += blockDim.x) {
    inverse[linear] = 0.0f;
  }
  __syncthreads();

  for (int row = 0; row < chunk_size; ++row) {
    if (column == row) {
      inverse[row * chunk_size + column] = 1.0f;
    } else if (column < row) {
      float sum = 0.0f;
      const float row_beta = beta[vector_base + row];
      for (int inner = column; inner < row; ++inner) {
        sum += row_beta * gram[matrix_base + row * chunk_size + inner]
            * inverse[inner * chunk_size + column];
      }
      inverse[row * chunk_size + column] = -sum;
    }
    __syncthreads();
  }

  for (int linear = column; linear < chunk_size * chunk_size;
       linear += blockDim.x) {
    output[matrix_base + linear] = inverse[linear];
  }
}


extern "C" void launch_kkt_inverse(
    void* gram,
    void* beta,
    void* output,
    int groups,
    void* raw_stream,
    int& result) {
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  flash_qla_kkt_inverse_kernel<<<groups, 64, 0, stream>>>(
      reinterpret_cast<const float*>(gram),
      reinterpret_cast<const float*>(beta),
      reinterpret_cast<float*>(output),
      groups);
  result = int(hggcGetLastError());
}


__global__ void flash_qla_scale_rows_negative_kernel(
    const float* values,
    const float* scale_rows,
    float* output,
    int total,
    int width) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += blockDim.x * gridDim.x) {
    output[index] = -scale_rows[index / width] * values[index];
  }
}


extern "C" void launch_scale_rows_negative(
    void* values,
    void* scale_rows,
    void* output,
    int total,
    int width,
    void* raw_stream,
    int& result) {
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  const int blocks = (total + 255) / 256;
  flash_qla_scale_rows_negative_kernel<<<blocks, 256, 0, stream>>>(
      reinterpret_cast<const float*>(values),
      reinterpret_cast<const float*>(scale_rows),
      reinterpret_cast<float*>(output),
      total,
      width);
  result = int(hggcGetLastError());
}


__global__ void flash_qla_affine_y_kernel(
    const float* u,
    const float* v_chunks,
    const float* gamma_last_chunks,
    const float* reverse_decay_chunks,
    float* output,
    int total,
    int heads,
    int chunks,
    int chunk_size,
    int value_dim,
    int chunk) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += blockDim.x * gridDim.x) {
    const int gate = chunk_gate_offset(
        index, heads, chunks, chunk_size, value_dim, chunk);
    const int group = index / (chunk_size * value_dim);
    const int batch = group / heads;
    const int head = group - batch * heads;
    const int row = (index / value_dim) % chunk_size;
    const int column = index % value_dim;
    const int value =
        ((((batch * heads + head) * chunks + chunk) * chunk_size + row)
         * value_dim) + column;
    const int last_gate = (batch * heads + head) * chunks + chunk;
    output[index] = gamma_last_chunks[last_gate] * u[index]
        - reverse_decay_chunks[gate] * v_chunks[value];
  }
}


extern "C" void launch_affine_y(
    void* u,
    void* v_chunks,
    void* gamma_last_chunks,
    void* reverse_decay_chunks,
    void* output,
    int batch,
    int heads,
    int chunks,
    int chunk_size,
    int value_dim,
    int chunk,
    void* raw_stream,
    int& result) {
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  const int total = batch * heads * chunk_size * value_dim;
  const int blocks = (total + 255) / 256;
  flash_qla_affine_y_kernel<<<blocks, 256, 0, stream>>>(
      reinterpret_cast<const float*>(u),
      reinterpret_cast<const float*>(v_chunks),
      reinterpret_cast<const float*>(gamma_last_chunks),
      reinterpret_cast<const float*>(reverse_decay_chunks),
      reinterpret_cast<float*>(output),
      total,
      heads,
      chunks,
      chunk_size,
      value_dim,
      chunk);
  result = int(hggcGetLastError());
}


__global__ void flash_qla_fused_chunks_fp32_kernel(
    const float* q,
    const float* k,
    const float* v,
    const float* beta,
    const float* weighted_kkt,
    const float* attention,
    const float* gamma,
    const float* reverse_decay,
    float* state,
    float* output,
    float* scratch_w,
    float* scratch_vp,
    int groups,
    int chunks,
    int chunk_size,
    int key_dim,
    int value_dim,
    float scale) {
  const int group = blockIdx.x;
  if (group >= groups) {
    return;
  }

  const int token_values = chunk_size * value_dim;
  const int state_values = key_dim * value_dim;
  const int chunk_keys = chunk_size * key_dim;
  const int chunk_values = chunk_size * value_dim;
  const int chunk_matrix = chunk_size * chunk_size;
  float* group_state = state + group * state_values;
  float* group_w = scratch_w + group * token_values;
  float* group_vp = scratch_vp + group * token_values;

  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int qk_base = (group * chunks + chunk) * chunk_keys;
    const int value_base = (group * chunks + chunk) * chunk_values;
    const int gate_base = (group * chunks + chunk) * chunk_size;
    const int matrix_base = (group * chunks + chunk) * chunk_matrix;

    // Official V consumer, stages 1-2: U=K@S, W=beta*(V-gamma*U).
    for (int index = threadIdx.x; index < token_values; index += blockDim.x) {
      const int row = index / value_dim;
      const int column = index - row * value_dim;
      float u = 0.0f;
      for (int inner = 0; inner < key_dim; ++inner) {
        u += k[qk_base + row * key_dim + inner]
            * group_state[inner * value_dim + column];
      }
      group_w[index] = beta[gate_base + row]
          * (v[value_base + index] - gamma[gate_base + row] * u);
    }
    __syncthreads();

    // Official V consumer, stage 3: Vd=(Gamma o A)@W.
    for (int index = threadIdx.x; index < token_values; index += blockDim.x) {
      const int row = index / value_dim;
      const int column = index - row * value_dim;
      float value = 0.0f;
      for (int inner = 0; inner < chunk_size; ++inner) {
        value += weighted_kkt[matrix_base + row * chunk_size + inner]
            * group_w[inner * value_dim + column];
      }
      group_vp[index] = value;
    }
    __syncthreads();

    // Official O consumer: O=scale*gamma*Q@S + Pg@Vd.
    for (int index = threadIdx.x; index < token_values; index += blockDim.x) {
      const int row = index / value_dim;
      const int column = index - row * value_dim;
      float q_state = 0.0f;
      for (int inner = 0; inner < key_dim; ++inner) {
        q_state += q[qk_base + row * key_dim + inner]
            * group_state[inner * value_dim + column];
      }
      float attended = 0.0f;
      for (int inner = 0; inner < chunk_size; ++inner) {
        attended += attention[matrix_base + row * chunk_size + inner]
            * group_vp[inner * value_dim + column];
      }
      output[value_base + index] =
          scale * gamma[gate_base + row] * q_state + attended;
    }
    __syncthreads();

    // Official state consumer: S=gamma_last*S+K^T@(reverse_decay*Vd).
    const float gamma_last = gamma[gate_base + chunk_size - 1];
    for (int index = threadIdx.x; index < state_values; index += blockDim.x) {
      const int row = index / value_dim;
      const int column = index - row * value_dim;
      float delta = 0.0f;
      for (int inner = 0; inner < chunk_size; ++inner) {
        delta += k[qk_base + inner * key_dim + row]
            * reverse_decay[gate_base + inner]
            * group_vp[inner * value_dim + column];
      }
      group_state[index] = gamma_last * group_state[index] + delta;
    }
    __syncthreads();
  }
}


extern "C" void launch_fused_chunks_fp32(
    void* q,
    void* k,
    void* v,
    void* beta,
    void* weighted_kkt,
    void* attention,
    void* gamma,
    void* reverse_decay,
    void* state,
    void* output,
    void* scratch_w,
    void* scratch_vp,
    int batch,
    int heads,
    int chunks,
    int chunk_size,
    int key_dim,
    int value_dim,
    float scale,
    void* raw_stream,
    int& result) {
  auto stream = reinterpret_cast<hggcStream_t>(raw_stream);
  const int groups = batch * heads;
  flash_qla_fused_chunks_fp32_kernel<<<groups, 256, 0, stream>>>(
      reinterpret_cast<const float*>(q),
      reinterpret_cast<const float*>(k),
      reinterpret_cast<const float*>(v),
      reinterpret_cast<const float*>(beta),
      reinterpret_cast<const float*>(weighted_kkt),
      reinterpret_cast<const float*>(attention),
      reinterpret_cast<const float*>(gamma),
      reinterpret_cast<const float*>(reverse_decay),
      reinterpret_cast<float*>(state),
      reinterpret_cast<float*>(output),
      reinterpret_cast<float*>(scratch_w),
      reinterpret_cast<float*>(scratch_vp),
      groups,
      chunks,
      chunk_size,
      key_dim,
      value_dim,
      scale);
  result = int(hggcGetLastError());
}
