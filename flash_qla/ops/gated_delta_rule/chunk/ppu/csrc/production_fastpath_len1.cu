#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

__device__ __forceinline__ float bf16f(const __nv_bfloat16 x) {
  return __bfloat162float(x);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}

__global__ void gdn_len1_batch_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ g,
    const __nv_bfloat16* __restrict__ beta,
    const __nv_bfloat16* __restrict__ initial,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ final_state,
    int count) {
  constexpr int HQ = 4, HV = 16, D = 128, REP = 4;
  const int request = blockIdx.x;
  const int h = blockIdx.y;
  const int tile = blockIdx.z;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int hq = h / REP;
  if (request >= count || h >= HV || tile >= 4) return;

  __shared__ float qv[D], kv[D], qsum[D], ksum[D];
  if (tid < D) {
    float qx = bf16f(q[((size_t)request * HQ + hq) * D + tid]);
    float kx = bf16f(k[((size_t)request * HQ + hq) * D + tid]);
    qv[tid] = qx;
    kv[tid] = kx;
    qsum[tid] = qx * qx;
    ksum[tid] = kx * kx;
  }
  __syncthreads();
  for (int stride = D / 2; stride; stride >>= 1) {
    if (tid < stride) {
      qsum[tid] += qsum[tid + stride];
      ksum[tid] += ksum[tid + stride];
    }
    __syncthreads();
  }
  if (tid < D) {
    qv[tid] *= rsqrtf(qsum[0] + 1e-6f);
    kv[tid] *= rsqrtf(ksum[0] + 1e-6f);
  }
  __syncthreads();

  constexpr float scale = 0.08838834764831845f;
  for (int row_group = 0; row_group < 4; ++row_group) {
    const int value_row = tile * 32 + row_group * 8 + warp;
    const size_t state_row = (((size_t)request * HV + h) * D + value_row) * D;
    float recon = 0.0f;
#pragma unroll
    for (int x = lane; x < D; x += 32) {
      recon += bf16f(initial[state_row + x]) * kv[x];
    }
    recon = warp_sum(recon);
    float decay = 0.0f, delta = 0.0f;
    if (lane == 0) {
      decay = expf(bf16f(g[(size_t)request * HV + h]));
      delta =
          (bf16f(v[((size_t)request * HV + h) * D + value_row]) - recon * decay) *
          bf16f(beta[(size_t)request * HV + h]);
    }
    decay = __shfl_sync(0xffffffffu, decay, 0);
    delta = __shfl_sync(0xffffffffu, delta, 0);
    float out = 0.0f;
#pragma unroll
    for (int x = lane; x < D; x += 32) {
      const float next = bf16f(initial[state_row + x]) * decay + delta * kv[x];
      final_state[state_row + x] = next;
      out += next * qv[x];
    }
    out = warp_sum(out);
    if (lane == 0)
      output[((size_t)request * HV + h) * D + value_row] = __float2bfloat16_rn(out * scale);
  }
}

void forward_len1_into(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                       torch::Tensor g, torch::Tensor beta, torch::Tensor initial,
                       torch::Tensor output, torch::Tensor final_state, int64_t count) {
  auto stream = at::cuda::getCurrentCUDAStream();
  gdn_len1_batch_kernel<<<dim3((unsigned)count, 16, 4), 256, 0, stream>>>(
      (const __nv_bfloat16*)q.data_ptr(),
      (const __nv_bfloat16*)k.data_ptr(),
      (const __nv_bfloat16*)v.data_ptr(),
      (const __nv_bfloat16*)g.data_ptr(),
      (const __nv_bfloat16*)beta.data_ptr(),
      (const __nv_bfloat16*)initial.data_ptr(),
      (__nv_bfloat16*)output.data_ptr(), final_state.data_ptr<float>(), (int)count);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_len1_into", &forward_len1_into);
}
