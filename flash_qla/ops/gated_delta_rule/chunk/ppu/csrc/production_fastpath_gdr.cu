// chunk_gated_delta_rule — single fused CUDA kernel with inline PTX (WMMA m16n16k16 bf16 MMA).
//
// Algorithm (derived from reference.py, chunked with BT=64, the FLA WY form):
//   per (request r, value head h, V-row block of VB rows):
//     S in R^{VB x K} fp32, carried in registers across chunks (MMA accumulator fragments)
//     per chunk (L <= 64 tokens):
//       gc[i]   = inclusive prefix sum of g            (log decay, <= 0)
//       A[i][j] = beta_i (k_i . k_j) exp(gc_i-gc_j)     j<i   (strictly lower)
//       P[i][v] = k_i . S[v,:]                          (state read-out)
//       Z[i][v] = beta_i (v_i[v] - exp(gc_i) P[i][v])
//       U       = (I+A)^{-1} Z                          (blocked forward substitution)
//       S[v,:] <- exp(gc_{L-1}) S[v,:] + sum_j U[j][v] exp(gc_{L-1}-gc_j) k_j
//       R[i][j] = (q_i . k_j) exp(gc_i-gc_j)            j<=i
//       out_i   = K^-0.5 ( exp(gc_i) S0 q_i + sum_j R[i][j] U[j] )
//   final_state = S after the last chunk.
// Recurrent state is accumulated in fp32 throughout.
//
// v02 structure: the four big GEMM stages (A=KK^T, R=QK^T, P=K S0^T, T1=Q S0^T) run in ONE
// register-blocked phase so every warp keeps 3..6 independent accumulators live and each A-operand
// fragment feeds several MMAs (probe: 1 acc/warp = 5.47 cyc/mma, 4 acc = 1.67 cyc/mma).
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <c10/cuda/CUDAStream.h>

#define BT 64
#define KD 128
#define VD 128
#define NW 8            // warps per block == KD/16 (one k-tile of the state per warp)
#define LOG2E 1.4426950408889634f

// ---------------------------------------------------------------- inline PTX helpers
__device__ __forceinline__ uint4 ldg128(const void *p) {
  uint4 r;
  asm volatile("ld.global.nc.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ float2 ldg64f(const void *p) {
  float2 r;
  asm volatile("ld.global.nc.v2.f32 {%0,%1}, [%2];" : "=f"(r.x), "=f"(r.y) : "l"(p));
  return r;
}
__device__ __forceinline__ float ldg32f(const void *p) {
  float r;
  asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ unsigned short ldg16(const void *p) {
  unsigned short r;
  asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ void stg64(void *p, uint2 v) {
  asm volatile("st.global.v2.b32 [%0], {%1,%2};" ::"l"(p), "r"(v.x), "r"(v.y) : "memory");
}
__device__ __forceinline__ float ex2f(float x) {
  float r;
  asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ uint32_t cvt_bf16x2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ void mma_rc(float *d, const uint32_t *a, const uint32_t *b) {
  asm volatile(
      "wmma.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]),
        "r"(b[3]));
}
__device__ __forceinline__ void mma_rr(float *d, const uint32_t *a, const uint32_t *b) {
  asm volatile(
      "wmma.mma.sync.aligned.m16n16k16.row.row.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]),
        "r"(b[3]));
}
// v133 QDIR: one lane-word of a wmma A-fragment straight out of GLOBAL (L2) memory, no TSM stop.
__device__ __forceinline__ uint32_t ldg32u(const void *p) {
  uint32_t r;
  asm volatile("ld.global.nc.b32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t lds32(const void *p) {
  uint32_t r;
  asm volatile("ld.shared.b32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ unsigned short lds16(const void *p) {
  unsigned short r;
  asm volatile("ld.shared.u16 %0, [%1];" : "=h"(r) : "l"(p));
  return r;
}

// `row.col` mma whose B operand comes straight out of a matrix_a-order fragment load of an
// n x k row-major tile: that fragment needs regs 1 and 2 swapped, expressed here as an
// operand-list permutation so the compiler emits zero register moves.

// ===== varlen-v2 contract helpers ==========================================================
// `g` and `initial_state` are bf16 on the live rows and fp32 on the *_fp32_state_compat rows.
// Both are read a handful of times per BLOCK (the cumsum runs on one warp; s0 is read once
// before the chunk loop), so the dtype is a RUNTIME uniform flag -- templating it would double
// the instantiation count of the translation unit for no measurable gain (v121 priced unused
// instantiations at up to +0.9 % of op_lat).
__device__ __forceinline__ float ldg_gv(const void *__restrict__ p, size_t idx, int gf) {
  if (gf) return ldg32f((const float *)p + idx);
  return __bfloat162float(__ushort_as_bfloat16(ldg16((const __nv_bfloat16 *)p + idx)));
}
// two adjacent state elements; idx is even everywhere it is called from (cc = (lane&3)<<1 and
// KD = 128), so the bf16 form is a naturally aligned 4-byte load.
__device__ __forceinline__ float2 ldg_s2v(const void *__restrict__ p, size_t idx, int sd) {
  if (sd) return ldg64f((const float *)p + idx);
  uint32_t w = ldg32u((const __nv_bfloat16 *)p + idx);
  float2 r;
  r.x = __bfloat162float(__ushort_as_bfloat16((unsigned short)(w & 0xffffu)));
  r.y = __bfloat162float(__ushort_as_bfloat16((unsigned short)(w >> 16)));
  return r;
}

__device__ __forceinline__ void gdr_unpack4(uint2 w, float *f) {
  f[0] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(w.x & 0xffffu)));
  f[1] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(w.x >> 16)));
  f[2] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(w.y & 0xffffu)));
  f[3] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(w.y >> 16)));
}

// use_qk_l2norm_in_kernel=True.  Normalise the staged K and Q tiles IN PLACE and publish the
// normalised rows so the scan kernel reads them instead of the raw inputs.  fp32 accumulation of
// exactly-representable bf16 inputs + rsqrt(s + 1e-6) reproduces reference_v2.l2norm.
//
// g2 (MEASURED problem it attacks): the g1 form gave one whole 128-wide row to a warp, so a
// 256-thread block ran BT/8 = 8 SEQUENTIAL passes, each ending in a dependent 5-level shuffle
// butterfly -- 40 back-to-back `shfl` latencies inside a kernel `athanor profile` reports as
// latency-bound (long pole salu, 14.9 % util).  That cost ~4 us per prologue wave: g1's prologue
// is 160.62 us at T=15360 (6.15 waves) against the un-normalised v171 ancestor's 135.32.  Giving
// each lane 8 bf16 (one uint4) makes 16 lanes own a row, so a warp normalises TWO rows per pass:
// 4 passes x 4 butterfly levels = 16 dependent shuffles, and `unroll 2` overlaps two chains.
template <int NWP>
__device__ __forceinline__ void gdr_l2norm_stage(
    __nv_bfloat16 *Kb, __nv_bfloat16 *Qb, int LDK, int L, int warp, int lane,
    __nv_bfloat16 *__restrict__ kn, __nv_bfloat16 *__restrict__ qn, size_t gbase, int gstride) {
  const int sub = lane >> 4;          // which of the warp's two rows this half-warp owns
  const int sl = lane & 15;           // 16 lanes x 8 bf16 = KD = 128
  const int i0 = warp * 2 + sub;
#pragma unroll 2
  for (int j = 0; j < BT / (2 * NWP); ++j) {
    const int i = i0 + j * (2 * NWP);
    const int off = i * LDK + sl * 8;               // LDK = KD+8 = 136 halves, so off is 16 B aligned
    const uint4 kv = *(const uint4 *)(Kb + off);
    const uint4 qv = *(const uint4 *)(Qb + off);
    float kf[8], qf[8];
    gdr_unpack4(make_uint2(kv.x, kv.y), kf);
    gdr_unpack4(make_uint2(kv.z, kv.w), kf + 4);
    gdr_unpack4(make_uint2(qv.x, qv.y), qf);
    gdr_unpack4(make_uint2(qv.z, qv.w), qf + 4);
    float sk = 0.0f, sq = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      sk += kf[e] * kf[e];
      sq += qf[e] * qf[e];
    }
#pragma unroll
    for (int o = 8; o; o >>= 1) {     // laneMask <= 8 never crosses the 16-lane half
      sk += __shfl_xor_sync(0xffffffffu, sk, o);
      sq += __shfl_xor_sync(0xffffffffu, sq, o);
    }
    const float rk = rsqrtf(sk + 1e-6f), rq = rsqrtf(sq + 1e-6f);
    uint4 ko, qo;
    ko.x = cvt_bf16x2(kf[0] * rk, kf[1] * rk);
    ko.y = cvt_bf16x2(kf[2] * rk, kf[3] * rk);
    ko.z = cvt_bf16x2(kf[4] * rk, kf[5] * rk);
    ko.w = cvt_bf16x2(kf[6] * rk, kf[7] * rk);
    qo.x = cvt_bf16x2(qf[0] * rq, qf[1] * rq);
    qo.y = cvt_bf16x2(qf[2] * rq, qf[3] * rq);
    qo.z = cvt_bf16x2(qf[4] * rq, qf[5] * rq);
    qo.w = cvt_bf16x2(qf[6] * rq, qf[7] * rq);
    *(uint4 *)(Kb + off) = ko;
    *(uint4 *)(Qb + off) = qo;
    if (i < L) {
      const size_t go = gbase + (size_t)i * gstride + sl * 8;
      *(uint4 *)(kn + go) = ko;
      *(uint4 *)(qn + go) = qo;
    }
  }
}

__device__ __forceinline__ void mma_rc_t(float *d, const uint32_t *a, const uint32_t *t) {
  asm volatile(
      "wmma.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(t[0]), "r"(t[2]), "r"(t[1]),
        "r"(t[3]));
}
// ===== v43: the native warp-collective fragment load, applied ONLY where it measured faster ====
// SASS motivation (v41 NOTES): v28's object has 2448 `tsm.ld.b32` vs 372 `v.mma...m16n16k16`, i.e.
// 6.6 scalar shared loads per MMA (4 per fragment).  ppu15lab i:tsm.ld.mat.b32x4 / t:ldmatrix_feed:
// the `__ppu_ldmatrix_sync_*` intrinsics emit a dedicated warp-collective `tsm.ld.mat[.trans].b32x4`
// (1 instruction, 84 cyc latency) that is 2.5-3x cheaper per op than the scalar fan-out; RAW inline
// asm cannot reach the opcode (d:native_ldmatrix_raw_asm), which is why v28's `ldmatrix...` asm
// produced ZERO `tsm.ld.mat` in the ISA dump.
// Probe-verified bit-exact drop-in (workspace/v41/probe/{ldm.cu,run_ldm.py}): at per-lane address
// `base + (lane&15)*ld + ((lane>>4)<<3)`, x4 reproduces `ldA` register-for-register and x4.trans is
// bit-identical to the raw-asm `ldmatrix.x4.trans`, with NO tile-layout change (the addressing is
// per lane, so the existing PAD strides -- all multiples of 8 elements = 16 B -- stay
// bank-conflict-free: lane l<16 hits banks {4l..4l+3} and lane 16+j hits {4j+4..4j+7}, i.e. exactly
// 4 word-accesses per bank).
// MEASURED per stage (workspace/v43/varsweep.py, sid0): applying it EVERYWHERE (v42) is a net LOSS
// (op_lat 1213 -> 1279) because its 84-cyc latency is exposed in short chains -- prologue +20 %,
// triangular U solve +21 %, state update +4 % -- while the 8-k-step P/T1 GEMM gains -15..-24 %.
// So v43 applies it to the P/T1 GEMM only.
#ifndef GDR_LDM
#define GDR_LDM 1
#endif
#if GDR_LDM && defined(__HGGCCC__)
extern "C" {
typedef unsigned int gdr_v4u __attribute__((ext_vector_type(4)));
__device__ gdr_v4u __ppu_ldmatrix_sync_m8n8_x4_shared_b16(const int *p);
}
#define GDR_HAVE_LDM 1
#else
#define GDR_HAVE_LDM 0
#endif

__device__ __forceinline__ void ldm4t(uint32_t *r, const __nv_bfloat16 *p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
// fragment loaders (lane maps probe-verified on this device)
//   lane l: r = l>>2, c = 2*(l&3)
__device__ __forceinline__ void ldA(uint32_t *r, const __nv_bfloat16 *base, int ld, int lane) {
  int rr = lane >> 2, cc = (lane & 3) << 1;
  const __nv_bfloat16 *p = base + rr * ld + cc;
  r[0] = lds32(p);
  r[1] = lds32(p + 8 * ld);
  r[2] = lds32(p + 8);
  r[3] = lds32(p + 8 * ld + 8);
}
// same fragment as `ldA`, one instruction instead of four (see the v43 note above)
__device__ __forceinline__ void ldAm(uint32_t *r, const __nv_bfloat16 *base, int ld, int lane) {
#if GDR_HAVE_LDM
  gdr_v4u d = __ppu_ldmatrix_sync_m8n8_x4_shared_b16(
      (const int *)(base + (lane & 15) * ld + ((lane >> 4) << 3)));
  r[0] = d.x; r[1] = d.y; r[2] = d.z; r[3] = d.w;
#else
  ldA(r, base, ld, lane);
#endif
}
// ===== v142: the `row.col` B fragment, loaded ALREADY IN MMA OPERAND ORDER ====================
// ISA MOTIVATION (v141 census, `hgobjdump --dump-isa workspace/v141/build/gdr.cuda.o`):
// `gdr_scan2` emits **66 `v.madl.i32 vregD, c0x1, vregS, c0x0`** -- integer multiply-by-1-add-0,
// i.e. PURE REGISTER COPIES -- against only 32 `v.mma`.  62 of them sit inside the chunk loop
// (32 P/T1 + 14 U solve + 16 state+out) and every one of them is the compiler materialising
// `mma_rc_t`'s `{t[0],t[2],t[1],t[3]}` operand-list permutation:
//     tsm.ld.mat.b32x4  vreg[92:95], [...]           # the fragment as `ldAm` delivers it
//     v.madl.i32        vreg108, c0x1, vreg92, c0x0  # 92 -> 108
//     v.madl.i32        vreg109, c0x1, vreg94, c0x0  # 94 -> 109   <-- regs 1 and 2 SWAPPED
//     v.madl.i32        vreg110, c0x1, vreg93, c0x0  # 93 -> 110
//     v.madl.i32        vreg111, c0x1, vreg95, c0x0  # 95 -> 111
//     v.mma...          vreg[96:103], vreg[92:95], vreg[116:119], vreg[96:103]
// `mma_rc_t`'s source comment asserts the operand-list permutation makes the compiler "emit zero
// register moves".  It is FALSE: the MMA's B slot is a CONTIGUOUS register quad, so an operand-list
// permutation cannot be free -- it forces a 4-instruction copy of every B fragment.  1.94 copies
// per MMA, on FALU (`i:v.madl.i32`: v.madl/mull.i32 count on FALU, ~128 ops/cyc/SM).
//
// The fix is to permute the LANE->ADDRESS MAP instead, which is free.  `ldmatrix.x4` assigns result
// register k to the 8x8 sub-matrix whose 8 rows are addressed by lanes 8k..8k+7, so re-ordering
// which lane addresses which row re-orders the destination registers at zero cost:
//     ldAm : row = lane&15,                  col = (lane>>4)*8   -> r = (r0,r2,r1,r3) of the tile
//     ldBm : row = (lane&7) + (lane>>4)*8,   col = ((lane>>3)&1)*8 -> r = (r0,r1,r2,r3) IN ORDER
// so `ldBm` + `mma_rc` is BIT-IDENTICAL to `ldAm` + `mma_rc_t` with the copies deleted.  The scalar
// twin already existed and was never used in the scan: `ldBc` is exactly `ldA` with the same swap.
// (Every one of these must be verified numerically before it is believed -- v52 closed a whole
// stage for 90 versions on an operand-mapping bug it read as a performance verdict.)
__device__ __forceinline__ void ldBc(uint32_t *r, const __nv_bfloat16 *base, int ld, int lane) {
  int rr = lane >> 2, cc = (lane & 3) << 1;
  const __nv_bfloat16 *p = base + rr * ld + cc;
  r[0] = lds32(p);
  r[1] = lds32(p + 8);
  r[2] = lds32(p + 8 * ld);
  r[3] = lds32(p + 8 * ld + 8);
}
// B operand for `row.row` (stored k x n row-major): one warp-collective `ldmatrix.x4.trans`
// replaces the eight `ld.shared.u16` 2-byte gathers this used to need.
__device__ __forceinline__ void ldBr(uint32_t *r, const __nv_bfloat16 *base, int ld, int lane) {
  ldm4t(r, base + (lane & 15) * ld + ((lane >> 4) << 3));
}
// v142: `ldBc`'s single-instruction twin -- the same fragment `ldAm` returns, but with result
// registers 1 and 2 already exchanged, so a `row.col` MMA can consume it with NO operand-list
// permutation and therefore NO register copies.  See the v142 note above `ldBc`.
__device__ __forceinline__ void ldBm(uint32_t *r, const __nv_bfloat16 *base, int ld, int lane) {
#if GDR_HAVE_LDM
  gdr_v4u d = __ppu_ldmatrix_sync_m8n8_x4_shared_b16(
      (const int *)(base + ((lane & 7) + ((lane >> 4) << 3)) * ld + (((lane >> 3) & 1) << 3)));
  r[0] = d.x; r[1] = d.y; r[2] = d.z; r[3] = d.w;
#else
  ldBc(r, base, ld, lane);
#endif
}

#define ACC_ROW(e) (((e)&2) ? 8 : 0)
#define ACC_COL(e) (((e)&1) + (((e)&4) ? 8 : 0))

// ---------------------------------------------------------------- the kernel
template <int VB>
__global__ __launch_bounds__(NW * 32) void gdr_fused(
    const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v, const float *__restrict__ g,
    const __nv_bfloat16 *__restrict__ beta, const float *__restrict__ s0,
    const int *__restrict__ cu, __nv_bfloat16 *__restrict__ out, float *__restrict__ sf,
    int HQ, int HV, int rep, float scale, int mask) {
  constexpr int LDK = KD + 8;
  constexpr int LDS = KD + 8;
  constexpr int LDA = BT + 8;
  constexpr int LDU = BT + 8;
  constexpr int LDV = VB + 8;
  constexpr int LDX = 16 + 8;
  constexpr int NVT = VB / 16;
  constexpr int NBT = BT / 16;

  extern __shared__ char smem_raw[];
  char *sp = smem_raw;
  __nv_bfloat16 *Sb = (__nv_bfloat16 *)sp;  sp += VB * LDS * 2;
  __nv_bfloat16 *Kb = (__nv_bfloat16 *)sp;  sp += BT * LDK * 2;
  __nv_bfloat16 *Qb = (__nv_bfloat16 *)sp;  sp += BT * LDK * 2;
  __nv_bfloat16 *Vb = (__nv_bfloat16 *)sp;  sp += BT * LDV * 2;
  __nv_bfloat16 *Am = (__nv_bfloat16 *)sp;  sp += BT * LDA * 2;
  __nv_bfloat16 *Rm = (__nv_bfloat16 *)sp;  sp += BT * LDA * 2;
  __nv_bfloat16 *ZT = (__nv_bfloat16 *)sp;  sp += VB * LDU * 2;
  __nv_bfloat16 *UT = (__nv_bfloat16 *)sp;  sp += VB * LDU * 2;
  __nv_bfloat16 *US = (__nv_bfloat16 *)sp;  sp += VB * LDU * 2;
  __nv_bfloat16 *Xi = (__nv_bfloat16 *)sp;  sp += NBT * 16 * LDX * 2;
  float *gc = (float *)sp;   sp += BT * 4;
  float *dd = (float *)sp;   sp += BT * 4;
  float *dh = (float *)sp;   sp += BT * 4;
  float *rh = (float *)sp;   sp += BT * 4;
  float *ar = (float *)sp;   sp += BT * 4;
  float *ex = (float *)sp;   sp += BT * 4;
  float *bt = (float *)sp;   sp += BT * 4;
  float *shd = (float *)sp;  sp += 4;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int rr = lane >> 2, cc = (lane & 3) << 1;
  const int v0 = blockIdx.x * VB;
  const int h = blockIdx.y;
  const int rq = blockIdx.z;
  const int hq = h / rep;

  const int t_start = cu[rq];
  const int t_end = cu[rq + 1];
  const int nchunk = (t_end - t_start + BT - 1) / BT;

  // phase-1 role: warps 0..3 -> {A = K K^T, P = K S0^T}; warps 4..7 -> {R = Q K^T, T1 = Q S0^T}
  const int bi = warp & (NBT - 1);
  const bool grp1 = (warp >= NBT);

  // ---- state accumulator: warp `warp` owns k-tile `warp`, all NVT v-tiles ----
  float S[NVT][8];
  {
    const float *base = s0 + (((size_t)rq * HV + h) * VD + v0) * KD + warp * 16;
#pragma unroll
    for (int vt = 0; vt < NVT; ++vt)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        int vrow = vt * 16 + rr + half * 8;
        float2 a = ldg64f(base + (size_t)vrow * KD + cc);
        float2 b = ldg64f(base + (size_t)vrow * KD + cc + 8);
        S[vt][0 + half * 2] = a.x;
        S[vt][1 + half * 2] = a.y;
        S[vt][4 + half * 2] = b.x;
        S[vt][5 + half * 2] = b.y;
      }
  }

  for (int c = 0; c < nchunk; ++c) {
    const int t0 = t_start + c * BT;
    const int L = min(BT, t_end - t0);

    __syncthreads();
    // ---- (1) stage k, q, v tiles ----
    if (mask & 128) for (int idx = tid; idx < BT * 16; idx += NW * 32) {
      int i = idx >> 4, c8 = (idx & 15) << 3;
      uint4 kv = make_uint4(0, 0, 0, 0), qv = make_uint4(0, 0, 0, 0);
      if (i < L) {
        kv = ldg128(k + ((size_t)(t0 + i) * HQ + hq) * KD + c8);
        qv = ldg128(q + ((size_t)(t0 + i) * HQ + hq) * KD + c8);
      }
      *(uint4 *)(Kb + i * LDK + c8) = kv;
      *(uint4 *)(Qb + i * LDK + c8) = qv;
    }
    if (mask & 256) for (int idx = tid; idx < BT * (VB / 8); idx += NW * 32) {
      int i = idx / (VB / 8), c8 = (idx % (VB / 8)) << 3;
      uint4 vv = make_uint4(0, 0, 0, 0);
      if (i < L) vv = ldg128(v + ((size_t)(t0 + i) * HV + h) * VD + v0 + c8);
      *(uint4 *)(Vb + i * LDV + c8) = vv;
    }
    // ---- (2) bf16 snapshot of the chunk-start state ----
    if (mask & 1024)
#pragma unroll
    for (int vt = 0; vt < NVT; ++vt)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        __nv_bfloat16 *d = Sb + (vt * 16 + rr + half * 8) * LDS + warp * 16 + cc;
        *(uint32_t *)d = cvt_bf16x2(S[vt][0 + half * 2], S[vt][1 + half * 2]);
        *(uint32_t *)(d + 8) = cvt_bf16x2(S[vt][4 + half * 2], S[vt][5 + half * 2]);
      }
    // ---- (3) cumulative log-decay + per-row factors (warp 0) ----
    if (warp == 0 && (mask & 512)) {
      float a = (lane < L) ? ldg32f(g + (size_t)(t0 + lane) * HV + h) : 0.f;
      float b = (lane + 32 < L) ? ldg32f(g + (size_t)(t0 + lane + 32) * HV + h) : 0.f;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        float t = __shfl_up_sync(0xffffffffu, a, d);
        if (lane >= d) a += t;
      }
      float tot = __shfl_sync(0xffffffffu, a, 31);
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        float t = __shfl_up_sync(0xffffffffu, b, d);
        if (lane >= d) b += t;
      }
      b += tot;
      gc[lane] = a;
      gc[lane + 32] = b;
      float glast = __shfl_sync(0xffffffffu, (L > 32) ? b : a, (L - 1) & 31);
      float bl = (lane < L) ? __bfloat162float(__ushort_as_bfloat16(
                                  ldg16(beta + (size_t)(t0 + lane) * HV + h)))
                            : 0.f;
      float bh = (lane + 32 < L)
                     ? __bfloat162float(__ushort_as_bfloat16(
                           ldg16(beta + (size_t)(t0 + lane + 32) * HV + h)))
                     : 0.f;
      bt[lane] = bl;
      bt[lane + 32] = bh;
      float d0 = ex2f(a * LOG2E), d1 = ex2f(b * LOG2E);
      dd[lane] = d0;
      dd[lane + 32] = d1;
      ex[lane] = ex2f((glast - a) * LOG2E);
      ex[lane + 32] = ex2f((glast - b) * LOG2E);
      if (lane == 0) *shd = ex2f(glast * LOG2E);
    }
    __syncthreads();
    const float dlast = *shd;

    // ---- (4) register-blocked GEMM phase ----
    //   warps 0..3 : accM[bj] = (K K^T)[bi][bj] (bj<=bi) , accT[bv] = (K S0^T)[bi][bv]
    //   warps 4..7 : accM[bj] = (Q K^T)[bi][bj] (bj<=bi) , accT[bv] = (Q S0^T)[bi][bv]
    float accM[NBT][8];
    float accT[NVT][8];
#pragma unroll
    for (int t = 0; t < NBT; ++t)
#pragma unroll
      for (int e = 0; e < 8; ++e) accM[t][e] = 0.f;
#pragma unroll
    for (int t = 0; t < NVT; ++t)
#pragma unroll
      for (int e = 0; e < 8; ++e) accT[t][e] = 0.f;
    if (mask & 1) {
      const __nv_bfloat16 *Ab = (grp1 ? Qb : Kb) + bi * 16 * LDK;
#pragma unroll
      for (int kk = 0; kk < KD / 16; ++kk) {
        uint32_t ra[4];
        ldA(ra, Ab + kk * 16, LDK, lane);
#pragma unroll
        for (int bj = 0; bj < NBT; ++bj) {
          if (bj <= bi) {
            uint32_t rb[4];
            ldBc(rb, Kb + bj * 16 * LDK + kk * 16, LDK, lane);
            mma_rc(accM[bj], ra, rb);
          }
        }
#pragma unroll
        for (int bv = 0; bv < NVT; ++bv) {
          uint32_t rb[4];
          ldBc(rb, Sb + bv * 16 * LDS + kk * 16, LDS, lane);
          mma_rc(accT[bv], ra, rb);
        }
      }
    }
    // ---- (5) epilogue of A / R ----
    if (mask & 2) {
      __nv_bfloat16 *dst = grp1 ? Rm : Am;
      float f0 = grp1 ? 1.f : bt[bi * 16 + rr];
      float f1 = grp1 ? 1.f : bt[bi * 16 + rr + 8];
      float ai0 = gc[bi * 16 + rr], ai1 = gc[bi * 16 + rr + 8];
#pragma unroll
      for (int bj = 0; bj < NBT; ++bj) {
        if (bj <= bi) {
          float g0 = gc[bj * 16 + cc], g1 = gc[bj * 16 + cc + 1];
          float g2 = gc[bj * 16 + cc + 8], g3 = gc[bj * 16 + cc + 9];
          const float gv[4] = {g0, g1, g2, g3};
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            int i = bi * 16 + rr + ACC_ROW(e);
            int j = bj * 16 + cc + ACC_COL(e);
            float fr = (e & 2) ? f1 : f0;
            float aj = gv[((e & 4) >> 1) | (e & 1)];
            float ai = (e & 2) ? ai1 : ai0;
            float dij = ex2f(fminf(ai - aj, 0.f) * LOG2E);
            bool keep = grp1 ? (j <= i) : (j < i);
            dst[i * LDA + j] = __float2bfloat16_rn(keep ? fr * dij * accM[bj][e] : 0.f);
          }
        }
      }
    }
    // ---- (6) Z = beta (V - exp(gc) P) transposed (warps 0..3 only) ----
    if (!grp1 && (mask & 4)) {
#pragma unroll
      for (int bv = 0; bv < NVT; ++bv) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          int i = bi * 16 + rr + ACC_ROW(e);
          int vv = bv * 16 + cc + ACC_COL(e);
          float z = bt[i] * (__bfloat162float(Vb[i * LDV + vv]) - dd[i] * accT[bv][e]);
          ZT[vv * LDU + i] = __float2bfloat16_rn(z);
        }
      }
    }
    __syncwarp();
    // ---- (7) inverses of the NBT diagonal 16x16 unit-lower blocks (fp32 substitution) ----
    if (!grp1 && lane < 16 && (mask & 8)) {
      const __nv_bfloat16 *Nb = Am + (bi * 16) * LDA + bi * 16;
      float X[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        float a = (lane == i) ? 1.f : 0.f;
#pragma unroll
        for (int j = 0; j < 16; ++j)
          if (j < i) a -= __bfloat162float(Nb[i * LDA + j]) * X[j];
        X[i] = a;
      }
#pragma unroll
      for (int i = 0; i < 16; ++i)
        Xi[bi * 16 * LDX + i * LDX + lane] = __float2bfloat16_rn(X[i]);
    }
    __syncthreads();

    // ---- (8) solve (I+A) U = Z, blocked forward substitution ----
    for (int b = 0; b < NBT; ++b) {
      if (warp < NVT && (mask & 16)) {
        const int bv = warp;
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int s = 0; s < NBT; ++s) {
          if (s < b) {
            uint32_t ra[4], rb[4];
            ldA(ra, Am + (b * 16) * LDA + s * 16, LDA, lane);
            ldBc(rb, UT + bv * 16 * LDU + s * 16, LDU, lane);
            mma_rc(acc, ra, rb);
          }
        }
        float y[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          int j = b * 16 + rr + ACC_ROW(e);
          int vv = bv * 16 + cc + ACC_COL(e);
          y[e] = __bfloat162float(ZT[vv * LDU + j]) - acc[e];
        }
        __syncwarp();
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          int j = b * 16 + rr + ACC_ROW(e);
          int vv = bv * 16 + cc + ACC_COL(e);
          ZT[vv * LDU + j] = __float2bfloat16_rn(y[e]);
        }
        __syncwarp();
        float u[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        uint32_t ra[4], rb[4];
        ldA(ra, Xi + b * 16 * LDX, LDX, lane);
        ldBc(rb, ZT + bv * 16 * LDU + b * 16, LDU, lane);
        mma_rc(u, ra, rb);
        __syncwarp();
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          int j = b * 16 + rr + ACC_ROW(e);
          int vv = bv * 16 + cc + ACC_COL(e);
          UT[vv * LDU + j] = __float2bfloat16_rn(u[e]);
          US[vv * LDU + j] = __float2bfloat16_rn(u[e] * ex[j]);
        }
        __syncwarp();
      }
    }
    __syncthreads();

    // ---- (9) state update: S = dlast*S + (U*ex)^T K ----
    if (mask & 32) {
#pragma unroll
      for (int vt = 0; vt < NVT; ++vt)
#pragma unroll
        for (int e = 0; e < 8; ++e) S[vt][e] *= dlast;
#pragma unroll
      for (int s = 0; s < NBT; ++s) {
        uint32_t rb[4];
        ldBr(rb, Kb + s * 16 * LDK + warp * 16, LDK, lane);
#pragma unroll
        for (int vt = 0; vt < NVT; ++vt) {
          uint32_t ra[4];
          ldA(ra, US + vt * 16 * LDU + s * 16, LDU, lane);
          mma_rr(S[vt], ra, rb);
        }
      }
    }

    // ---- (10) out = scale*( exp(gc_i) q_i S0^T + R U )   (warps 4..7, T1 still in regs) ----
    if (grp1 && (mask & 64)) {
      float d0 = dd[bi * 16 + rr], d1 = dd[bi * 16 + rr + 8];
#pragma unroll
      for (int bv = 0; bv < NVT; ++bv)
#pragma unroll
        for (int e = 0; e < 8; ++e) accT[bv][e] *= (e & 2) ? d1 : d0;
#pragma unroll
      for (int s = 0; s < NBT; ++s) {
        if (s <= bi) {
          uint32_t ra[4];
          ldA(ra, Rm + bi * 16 * LDA + s * 16, LDA, lane);
#pragma unroll
          for (int bv = 0; bv < NVT; ++bv) {
            uint32_t rb[4];
            ldBc(rb, UT + bv * 16 * LDU + s * 16, LDU, lane);
            mma_rc(accT[bv], ra, rb);
          }
        }
      }
#pragma unroll
      for (int bv = 0; bv < NVT; ++bv)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          int i = bi * 16 + rr + half * 8;
          if (i < L) {
            __nv_bfloat16 *dst = out + ((size_t)(t0 + i) * HV + h) * VD + v0 + bv * 16 + cc;
            *(uint32_t *)dst =
                cvt_bf16x2(accT[bv][0 + half * 2] * scale, accT[bv][1 + half * 2] * scale);
            *(uint32_t *)(dst + 8) =
                cvt_bf16x2(accT[bv][4 + half * 2] * scale, accT[bv][5 + half * 2] * scale);
          }
        }
    }
  }

  // ---- final state ----
  {
    float *base = sf + (((size_t)rq * HV + h) * VD + v0) * KD + warp * 16;
#pragma unroll
    for (int vt = 0; vt < NVT; ++vt)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float *d = base + (size_t)(vt * 16 + rr + half * 8) * KD + cc;
        uint2 a, b;
        a.x = __float_as_uint(S[vt][0 + half * 2]);
        a.y = __float_as_uint(S[vt][1 + half * 2]);
        b.x = __float_as_uint(S[vt][4 + half * 2]);
        b.y = __float_as_uint(S[vt][5 + half * 2]);
        stg64(d, a);
        stg64(d + 8, b);
      }
  }
}

// ================================================================ long-sequence 2-kernel path
// The monolith above owns one (request, value-head, VB rows of the VxK state) slice and marches
// every chunk SERIALLY, so on T=14784/HV=32 it runs 64 blocks on 39 SMs each serialising 231
// chunks -- great for NT<=2, a 0.7x regression for NT>=36.  For long sequences the chunk-local
// half of the algorithm is split off into a kernel with NT*HV*N blocks:
//   K1 `gdr_prologue`  : cumsum(g) -> A = strict_lower(diag(beta) K D K^T) -> (I+A)^-1, emitted
//                        DIRECTLY as bf16 (no fp32 A, no cast kernel, no zeros_like fill).
//   K2 `gdr_scan<VB>`  : the inter-chunk state scan fused with the output projection, state kept
//                        on-chip in fp32 MMA accumulators; U = (I+A)^-1 Z is now a single MMA
//                        chain instead of a 4-step serial forward substitution.
#define NF 6  // per-chunk row-factor planes: dd, ex, gc, -, bt, [dlast]

// ===== v24 (SEGSCAN): GQA Gram sharing at FULL parallelism =====================================
// A7 says the raw Grams (k_i.k_j) and (q_i.k_j) depend only on the KEY head, so with rep = HV/KH
// they are computed 4x redundantly by g0's (NT,HV) grid.  v21 shared them by keying the grid on
// the key head and looping the 4 value-head epilogues SERIALLY -- that was REFUTED (sid0 prologue
// 307.7 -> 342.0) because the prologue is critical-path bound on the sequential (I+A)^-1, so a 4x
// longer per-block chain costs more than the 4x saved Gram MMAs.
//
// v24 keeps the FLOP saving and the 4x smaller K/Q read while restoring the original critical
// path: one block per (chunk, KEY head) with 1024 threads = 32 warps = FOUR independent 8-warp
// value-head groups.  Warps 0-7 build the two raw Grams once into shared memory (packed bf16 pairs
// in exactly the wmma accumulator element order, so consumers re-read them with `ldA` and hit no
// bank conflict); then all four groups run g0's per-head epilogue CONCURRENTLY, each on its own
// named barrier.  Same warps/SM (32) and same chain depth as g0, 4x fewer Gram MMAs, 4x less K/Q
// traffic.
#define LDG (BT + 8)      // shared stride of the raw Gram planes, in bf16 elements

// v44: NGRP is now a template parameter.  MEASURED problem it attacks: `gdr_prologue` is 296.5 us
// = 25 % of op_lat at 7.8x its own HBM roof (94 MB of traffic = 38 us at 2.5 TB/s), and `occ.py`
// says it runs at **1 block/SM**, register-limited (184 regs x 512 threads = 94208 of
// q:chip/regfile.per_sm_32b = 131072).  With one block per SM there is no second block to cover
// its 6 barriers per value head, which is exactly the failure mode v24 measured (1024-thread
// prologue, 1 block/SM, 0.37x).  NGRP=2 halves the block to 256 threads: 184 x 256 = 47104 regs and
// 64512 B smem => 2 blocks/SM, so the same 16 warps/SM but each barrier only drains 8 warps and the
// sibling block keeps the SM issuing.  The cost is that each block now runs its 4 value-head
// epilogues in 2 sequential passes instead of 4 concurrent groups.
// ===== v111: register-budget knobs for `gdr_prologue` ==========================================
// MEASURED lead (LEDGER "Open residual lead"): v105's `gdr_prologue<...,NOR>` (no R epilogue) hits
// 144 regs and 153.8 us, vs v51's 184 regs / 221.1 us, while v47's stage attribution prices the R
// epilogue + emit at only 29.9 us.  184 x 256 = 47104 => 2 blocks/SM; the 3-blocks/SM threshold is
// 131072/(256*3) = 170.7 regs.  So getting <=170 regs SPILL-FREE while still emitting R should buy
// the same barrier-drain win v44 measured going 1 -> 2 blocks/SM (-19 %).
//   RV bitmask (register-reduction variants, all arithmetically identical unless noted):
//     bit0 (1)  GT2   : the two raw Grams accumulate TWO output tiles at a time (acc[2][8] = 16
//                       fp32) instead of all four (acc[4][8] = 32 fp32), reloading the A fragment
//                       for the second pass.  Same MMA count, same k-order => bit-identical.
//     bit1 (2)  GT1   : one output tile at a time (acc[8] = 8 fp32).  Bit-identical too.
//     bit2 (4)  XSM   : the 16x16 diagonal inverse keeps its fp32 substitution vector X[16] in a
//                       per-warp fp32 smem scratch instead of 16 registers.  Values stay fp32, so
//                       this is bit-identical.
//     bit3 (8)  RLDM  : native `tsm.ld.mat` (ldAm) for the R epilogue's GQK fragment.
//     bit4 (16) ALDM  : native `tsm.ld.mat` (ldAm) for the A epilogue's GKK fragment.
//     bit5 (32) INU1  : `#pragma unroll 1` on the blocked 64x64 inverse's b loop.
//     bit6 (64) AEU1  : `#pragma unroll 1` on the A epilogue's bj loop.
//     bit7 (128) REU1 : `#pragma unroll 1` on the R epilogue's bj loop.
//     bit8 (256) NOXI : drop the `Xi` scratch entirely.  The four 16x16 diagonal inverses
//                       already write the SAME values into Tv's diagonal blocks, so the
//                       blocked inverse can take its A fragment straight from
//                       `Tv + b*16*LDA + b*16` (stride LDA) instead of `Xi + b*16*LDX`.
//                       Bit-identical, and it takes smem/block from 70656 to **64512** =
//                       262144/4, which is what a <=128-reg variant needs to reach
//                       **4 blocks/SM** (104 regs x 256 x 4 = 106496 < 131072).
//     bit9 (512) U2   : the three U1 knobs use unroll factor 2 instead of 1 (middle ground).
//   MEASURED SK attribution (`workspace/v111/occ2.py`, real cudaFuncGetAttributes) -- this is what
//   picked the knobs.  Dropping one stage from the 184-reg kernel gives:
//     -Gram MMA 184 | -cumsum/fac 168 | -A epilogue 152 | -diag inverses 160 |
//     -blocked 64x64 inverse **88** | -emit T 176 | -R epilogue 152
//   So the register peak is NOT the Gram's 32 fp32 accumulators (blocking them to 2 or 1 tile
//   leaves 184 exactly): it is the FULLY UNROLLED blocked inverse, whose three b steps let the
//   compiler hoist up to nine 4-register `ldA` fragments of Am/Xi above the chain.  The A and R
//   epilogues contribute the same way (four hoisted `gr[4]` fragments each).  Capping the unroll
//   factor is therefore the direct lever; XSM attacks the diag inverse's X[16].
//   SK is a pure INSTRUMENT (never shipped non-zero): it drops one stage so that
//   `cudaFuncGetAttributes` attributes the register peak to a stage.
// v171: VK = value-keyed grid.  When VK, grid.y indexes the VALUE head (HV of them) instead of the
// KEY head (HQ), so the block count is HV/HQ = rep times larger (sid3: 48 -> 192).  Each block then
// owns EXACTLY ONE value head, computing the two 64x64 Grams for its key head redundantly (the
// GQA-shared Gram of v25 is UN-done here) -- but on the small shapes the prologue grid is a fraction
// of one wave (48 blocks / 39 SMs), so the extra blocks fall onto idle SMs and buy latency hiding
// that the shared-Gram, key-keyed grid could not.  Runs NGRP=1 (128 threads / 4 warps): one group,
// one value head, half the per-block critical path of the 2-pass NGRP=2 form.
template <int NGRP, bool GLDM, bool PCPA, int RV, int SK, int PI = 0, int VK = 0>
__device__ __forceinline__ void gdr_prologue_body(
    const __nv_bfloat16 *__restrict__ k, const __nv_bfloat16 *__restrict__ q,
    const void *__restrict__ g, const __nv_bfloat16 *__restrict__ beta,
    const int *__restrict__ cu,
    __nv_bfloat16 *__restrict__ tg, __nv_bfloat16 *__restrict__ rg,
    float *__restrict__ fg, __nv_bfloat16 *__restrict__ kn,
    __nv_bfloat16 *__restrict__ qn, int HQ, int HV, int rep, int gf) {
  constexpr int LDK = KD + 8;
  constexpr int LDA = BT + 8;
  constexpr int LDX = 24;
  constexpr int NBT = BT / 16;
  // Am,Tv,[Xi],Yt per group.  v112 RV bit8 (NOXI) drops Xi: it is a byte-for-byte duplicate
  // of Tv's four diagonal 16x16 blocks, which the blocked inverse never overwrites (it only
  // writes block-column j < b), so the fragment can be read from Tv directly.
  constexpr int NXI = (RV & 256) ? 1 : 2;
  // ===== v151: PI = the 16x16 DIAGONAL-INVERSE knobs ==========================================
  // ISA CENSUS that produced them (`workspace/v151/census.py` on v145's own object, basic block
  // BB3_83 of `gdr_prologue<2,false,true,480,0>`, which is exactly the diagonal-inverse stage --
  // v143 prices it at **32.9 us, the largest single prologue stage**):
  //     391 instructions per warp, of which
  //       120  v.fma.f32.rtte          <- the substitution FLOPs themselves (31 %)
  //       135  v.shll.b32 / v.and.b32 / v.bfe.b32   <- PURE bf16 -> fp32 UNPACKING (35 %)
  //        35  tsm.ld.b16/b32/b32x2/b32x4  <- the 120 broadcast A loads, already vectorised
  //        31  tsm.st.b16 + v.cnvt.bf16    <- the 16-value X writeback
  //        32  s.wait
  // So the stage is NEITHER substitution FLOPs (31 %) NOR register copies (3 of 391) NOR smem
  // round trips: **a third of it is bf16 unpacking and it runs on HALF A WARP** (`lane < 16`),
  // i.e. 4 warps x 16 lanes for 4 x 16 = 64 independent columns.  Two independent knobs:
  //   bit0 (1) DIH : run the four blocks on TWO warps using BOTH half-warps (warp gw<2 takes
  //                  blocks 2*gw and 2*gw+1, lane>>4 selects the block, lane&15 the column).
  //                  Halves the warp-instructions this stage issues per SM.  BIT-IDENTICAL --
  //                  the per-column arithmetic is untouched, only which lane runs it.
  //   bit1 (2) DIF : the A epilogue ALSO stores the four diagonal 16x16 blocks as **widened
  //                  fp32** (`__bfloat162float(__float2bfloat16_rn(v))`, i.e. the exact bf16
  //                  value it already stores, in 4 bytes) into a triangular-packed scratch,
  //                  so the substitution loads fp32 and the 120 unpack ops disappear.  Costs
  //                  ZERO extra smem: it lives in `Yt`, which the blocked 64x64 inverse only
  //                  touches AFTER the barrier that ends the diagonal inverses (768 floats
  //                  available, 4 x 120 = 480 used).  BIT-IDENTICAL for the same reason.
  //   bit2 (4) BI2 : the blocked 64x64 (I+A)^-1 as a **two-level 32x32 split** instead of four
  //                  sequential block-columns.  v143 prices the blocked inverse at 29.6 us; the
  //                  census says it issues only ~570 warp-instructions per group-pass (19 instr/us,
  //                  vs the diagonal inverses' 56) so it is LATENCY-bound, and its critical warp
  //                  (j=0, which owns b=1,2,3) is **6 smem-mediated dependent MMA levels** deep
  //                  at v121's measured ~680 cyc each.  Blocking 64 = 2 x 32 instead of 4 x 16
  //                  makes it 4 levels: (1) the two 32x32 diagonal inverses in parallel on warps
  //                  0 and 1 -- T[1][0] = -Xd1 A10 Xd0 and T[3][2] = -Xd3 A32 Xd2, 2 levels; then
  //                  (2) one 32x32 off-diagonal correction T(2:4,0:2) = -Tinv(2:4) A(2:4,0:2)
  //                  Tinv(0:2), 2 levels, warp gw owning block-column gw so that BOTH of its Y
  //                  tiles stay warp-private and no group barrier is needed inside the step.
  //                  Identical algebra (the standard 2x2 block inverse), identical MMA count (16),
  //                  but the bf16 rounding of the intermediate Y happens at a different place, so
  //                  this one is NOT bit-identical -- it must pass the real validate.
  //   bit3 (8) TRI : the A and R epilogues drop the per-element `j < i` / `j <= i` triangular
  //                  select on the six STRICTLY-lower tiles, where it is unconditionally true
  //                  (j <= bj*16+15 < bi*16 <= i).  Only the diagonal tile keeps it.  8 compares
  //                  + 8 selects saved per strictly-lower tile.  BIT-IDENTICAL.
  constexpr bool DIH = (PI & 1) != 0;
  constexpr bool DIF = (PI & 2) != 0;
  //   bit4 (16) GLM : native `tsm.ld.mat` (`ldAm`) for the two Gram GEMMs' A/B fragments, i.e.
  //                  `GLDM = true`.  RE-TEST, not a new idea: v44/v47/v112 refuted it three times
  //                  (996.6 vs 989 at 184 r, 950.2 vs 943.2 at 144 r, 926.5 vs 921.8 at 104 r) and
  //                  v53 gave the mechanism -- the Gram MMA HIDES UNDER the sequential (I+A)^-1
  //                  chain, so removing load latency from it buys nothing.  **`BI2` just shortened
  //                  that chain from 6 dependent smem-mediated MMA levels to 4**, and the LEDGER's
  //                  standing rule is that `ld.mat` is a per-call-site AND per-ILP-level decision
  //                  that must be re-tested after any change to the covering chain (it has already
  //                  flipped twice).  The census also says this is where the scalar fan-out lives:
  //                  `gdr_prologue<...,480>` issues **196 `tsm.ld.b32` for 30 `v.mma` = 6.5 scalar
  //                  loads per MMA**, the exact signature v131 fixed in the scan's state+out.
  //   bit5 (32) RZE : the R epilogue stops writing the six STRICTLY-UPPER 16x16 tiles of `Am` as
  //                  zeros in accumulator order (24 predicated `tsm.st.b32` per warp); the emit
  //                  loop writes the zeros straight to `rg` instead, skipping both the smem store
  //                  and the smem read.  BIT-IDENTICAL.
  constexpr bool BI2 = (PI & 4) != 0;
  constexpr bool TRI = (PI & 8) != 0;
  constexpr bool RZE = (PI & 32) != 0;
  constexpr int GRPB = BT * LDA * 2 * 2 + NBT * 16 * LDX * 2 * NXI;
  constexpr int NTHR = NGRP * 128;

  extern __shared__ char smem_raw[];
  char *sp = smem_raw;
  __nv_bfloat16 *GKK = (__nv_bfloat16 *)sp;  sp += BT * LDG * 2;   // raw (k_i.k_j), bf16
  __nv_bfloat16 *GQK = (__nv_bfloat16 *)sp;  sp += BT * LDG * 2;   // raw (q_i.k_j), bf16
  char *un = sp;                          // union: phase-1 K/Q tiles vs phase-2 group buffers
  __nv_bfloat16 *Kb = (__nv_bfloat16 *)un;
  __nv_bfloat16 *Qb = (__nv_bfloat16 *)(un + BT * LDK * 2);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int rr = lane >> 2, cc = (lane & 3) << 1;
  const int grp = warp >> 2;        // 0..3, one value head each
  const int gw = warp & 3;          // 0..3 within the group
  const int bi = gw;                // 16-row band this warp owns
  const int gtid = tid & 127;       // 0..127 within the group
  const int c = blockIdx.x;
  const int hq = VK ? (blockIdx.y / rep) : blockIdx.y;   // v171 VK: grid.y indexes the value head
  const int rq = blockIdx.z;

  // per-group scratch, carved out of the union region behind the K/Q tiles
  __nv_bfloat16 *Am = (__nv_bfloat16 *)(un + grp * GRPB);
  __nv_bfloat16 *Tv = Am + BT * LDA;
  __nv_bfloat16 *Xi = Tv + BT * LDA;                       // unused when NOXI
  __nv_bfloat16 *Yt = Tv + BT * LDA + (RV & 256 ? 0 : NBT * 16 * LDX);
  float *fac = (float *)(un + NGRP * GRPB) + grp * (NF * BT);
  // v111 RV bit2: fp32 substitution scratch for the 16x16 diagonal inverse, 16 lanes x 16 entries
  // per warp, laid out lane-major so each lane's 16 slots are 16 floats apart (conflict-free).
  float *Xs = (float *)(un + NGRP * GRPB) + NGRP * (NF * BT) + (grp * 4 + gw) * 256;
  // v151 DIF: the widened-fp32 diagonal blocks of A, triangular-packed (row i at i*(i-1)/2),
  // 128 floats of stride per block, aliased onto `Yt` (dead until the blocked inverse).
  float *Ad = (float *)Yt;

#ifdef FLASHQLA_EXACT_T
  const int t_start = 0;
  const int t_end = FLASHQLA_EXACT_T;
#else
  const int t_start = cu[rq];
  const int t_end = cu[rq + 1];
#endif
  if (t_start + c * BT >= t_end) return;
#ifdef FLASHQLA_EXACT_T
  const int coff = 0;
#else
  int coff = 0;
  for (int r = 0; r < rq; ++r) coff += (cu[r + 1] - cu[r] + BT - 1) / BT;
#endif
  const int cid = coff + c;
  const int t0 = t_start + c * BT;
  const int L = min(BT, t_end - t0);

  // ---- (1) stage the K + Q tiles of this key head ----
  // v46: `cp.async` (AIU, global->TSM, bypasses the register file) instead of the ldg128 -> vreg ->
  // st.shared round trip.  This is the campaign's biggest single lever (v26/v28: -19 % on the scan)
  // and it had never been applied to the prologue, which stages 32 KB per block through 184-register
  // pressure.  Nothing here needs a second stage: there is exactly one chunk per block.
  if (PCPA) {
    for (int idx = tid; idx < BT * 16; idx += NTHR) {
      int i = idx >> 4, c8 = (idx & 15) << 3;
      if (i < L) {
        __pipeline_memcpy_async(Kb + i * LDK + c8, k + ((size_t)(t0 + i) * HQ + hq) * KD + c8, 16);
        __pipeline_memcpy_async(Qb + i * LDK + c8, q + ((size_t)(t0 + i) * HQ + hq) * KD + c8, 16);
      } else {
        *(uint4 *)(Kb + i * LDK + c8) = make_uint4(0, 0, 0, 0);
        *(uint4 *)(Qb + i * LDK + c8) = make_uint4(0, 0, 0, 0);
      }
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
  } else {
    for (int idx = tid; idx < BT * 16; idx += NTHR) {
      int i = idx >> 4, c8 = (idx & 15) << 3;
      uint4 kv = make_uint4(0, 0, 0, 0), qv = make_uint4(0, 0, 0, 0);
      if (i < L) {
        kv = ldg128(k + ((size_t)(t0 + i) * HQ + hq) * KD + c8);
        qv = ldg128(q + ((size_t)(t0 + i) * HQ + hq) * KD + c8);
      }
      *(uint4 *)(Kb + i * LDK + c8) = kv;
      *(uint4 *)(Qb + i * LDK + c8) = qv;
    }
  }
  __syncthreads();

  // ---- (1b) varlen-v2: L2-normalise the staged K/Q rows in place and publish them ----
  // The prologue grid is (nct, HQ, N), i.e. exactly one block per (chunk, key head), so this
  // covers every (token, key head) row of q and k exactly once -- no redundancy, no extra
  // launch, and the scan then reads `kn`/`qn` with the SAME addressing it used for k/q.
  gdr_l2norm_stage<NTHR / 32>(Kb, Qb, LDK, L, warp, lane, kn, qn,
                              ((size_t)t0 * HQ + hq) * KD, HQ * KD);
  __syncthreads();

  // ---- (2) the two RAW Grams, ONCE per key head (warps 0-7), emitted as packed bf16 pairs in
  //          wmma accumulator element order so consumers re-read them with `ldA`.  ----
  // the two Grams need 8 warp-bands; with fewer than 8 warps resident each warp takes several.
  // ldAm (native tsm.ld.mat) is used here because this is an 8-independent-k-step GEMM, the shape
  // v42 measured the native load to win on by 20 %.
  // v111: GT = number of output tiles accumulated simultaneously.  v51 held all NBT=4 tiles live
  // (acc[4][8] = 32 fp32 registers) across the fully unrolled 8-step k loop; GT<4 blocks the bj
  // loop so only GT*8 accumulators are live, at the cost of reloading the A fragment once per
  // block.  The per-tile k order is unchanged, so every variant is BIT-identical.
  constexpr int GT = (RV & 2) ? 1 : ((RV & 1) ? 2 : NBT);
  if (!(SK & 1)) {
  for (int wi = warp; wi < 8; wi += NTHR / 32) {
    const int wb = wi & 3;
    const bool isq = (wi >= 4);
    const __nv_bfloat16 *Ab = (isq ? Qb : Kb) + wb * 16 * LDK;
    __nv_bfloat16 *dst = isq ? GQK : GKK;
#pragma unroll
    for (int b0 = 0; b0 < NBT; b0 += GT) {
      if (b0 > wb) continue;
      float acc[GT][8];
#pragma unroll
      for (int t = 0; t < GT; ++t)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[t][e] = 0.f;
#pragma unroll
      for (int kk = 0; kk < KD / 16; ++kk) {
        uint32_t ra[4];
        if (GLDM) ldAm(ra, Ab + kk * 16, LDK, lane);
        else ldA(ra, Ab + kk * 16, LDK, lane);
#pragma unroll
        for (int t = 0; t < GT; ++t) {
          const int bj = b0 + t;
          if (bj <= wb) {
            uint32_t rb[4];
            if (GLDM) ldAm(rb, Kb + bj * 16 * LDK + kk * 16, LDK, lane);
            else ldA(rb, Kb + bj * 16 * LDK + kk * 16, LDK, lane);
            mma_rc_t(acc[t], ra, rb);
          }
        }
      }
#pragma unroll
      for (int t = 0; t < GT; ++t) {
        const int bj = b0 + t;
        if (bj <= wb) {
#pragma unroll
          for (int p = 0; p < 4; ++p) {
            const int e = p * 2;
            int i = wb * 16 + rr + ACC_ROW(e);
            int j = bj * 16 + cc + ACC_COL(e);
            *(uint32_t *)(dst + i * LDG + j) = cvt_bf16x2(acc[t][e], acc[t][e + 1]);
          }
        }
      }
    }
  }
  }
  __syncthreads();   // Grams committed; K/Q tiles dead -> the group buffers may reuse that space

  // ===== v144 SK bit7 (128) PG: software-pipeline the cumsum's `g`/`beta` GLOBAL LOADS ==========
  // MEASURED MOTIVATION (v143's prologue attribution at 104 regs / 4 blocks/SM): `cumsum/fac` costs
  // **+21.7 us**, the third-largest prologue stage, and it is executed by `gw == 0` ALONE while the
  // group's other 3 warps wait at `bar.sync BAR_G, 128`.  Its critical path is
  //   4 global loads (g x2, beta x2) -> 10 `__shfl_up` levels -> 8 `ex2f` (SFU, 44 cyc) -> 24 stores
  // so it opens with a cold ~500-cycle HBM latency that nothing covers.  PG issues iteration i+1's
  // four loads at the TOP of iteration i (and iteration 0's before the head loop), so the latency
  // hides under the Grams (16.7 us) for the first head and under the whole A/inverse/T/R epilogue
  // (~96 us) for the second.  Cost: 4 registers on `gw == 0` -- the prologue is at 104 of the 128
  // that 256 thr x 4 blocks/SM allows (104*256*4 = 106496 <= 131072), so there is headroom.
  // Bit-identical: the same four values, loaded earlier.
  constexpr bool PG = (SK & 128) != 0;
  float pg_a = 0.f, pg_b = 0.f, pg_bl = 0.f, pg_bh = 0.f;
#define GDR_PGLOAD(VH_)                                                                       \
  if (PG && gw == 0) {                                                                        \
    const int _h = hq * rep + (VH_);                                                          \
    pg_a = (lane < L) ? ldg_gv(g, (size_t)(t0 + lane) * HV + _h, gf) : 0.f;                      \
    pg_b = (lane + 32 < L) ? ldg_gv(g, (size_t)(t0 + lane + 32) * HV + _h, gf) : 0.f;            \
    pg_bl = (lane < L) ? __bfloat162float(__ushort_as_bfloat16(                               \
                             ldg16(beta + (size_t)(t0 + lane) * HV + _h)))                    \
                       : 0.f;                                                                 \
    pg_bh = (lane + 32 < L) ? __bfloat162float(__ushort_as_bfloat16(                          \
                                  ldg16(beta + (size_t)(t0 + lane + 32) * HV + _h)))          \
                            : 0.f;                                                            \
  }
  GDR_PGLOAD(grp)

  // ---- (3) the four value-head epilogues, one per 8-warp group, CONCURRENTLY ----
  // v171 VK: the block owns exactly one value head (blockIdx.y); the loop runs once.
  const int BAR_G = 1 + grp;       // one named barrier per 4-warp group
  const int vh_lo = VK ? (blockIdx.y - hq * rep) : grp;
  const int vh_hi = VK ? (vh_lo + 1) : rep;
  const int vh_st = VK ? rep : NGRP;
  for (int vh = vh_lo; vh < vh_hi; vh += vh_st) {
    const int h = hq * rep + vh;

    if (!(SK & 2) && gw == 0) {
      float a, b, pfl = 0.f, pfh = 0.f;
      if (PG) {
        a = pg_a; b = pg_b; pfl = pg_bl; pfh = pg_bh;
        if (vh + NGRP < rep) { GDR_PGLOAD(vh + NGRP) }
      } else {
      a = (lane < L) ? ldg_gv(g, (size_t)(t0 + lane) * HV + h, gf) : 0.f;
      b = (lane + 32 < L) ? ldg_gv(g, (size_t)(t0 + lane + 32) * HV + h, gf) : 0.f;
      }
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        float t = __shfl_up_sync(0xffffffffu, a, d);
        if (lane >= d) a += t;
      }
      float tot = __shfl_sync(0xffffffffu, a, 31);
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        float t = __shfl_up_sync(0xffffffffu, b, d);
        if (lane >= d) b += t;
      }
      b += tot;
      float glast = __shfl_sync(0xffffffffu, (L > 32) ? b : a, (L - 1) & 31);
      float bl, bh;
      if (PG) {
        bl = pfl; bh = pfh;
      } else {
      bl = (lane < L) ? __bfloat162float(__ushort_as_bfloat16(
                            ldg16(beta + (size_t)(t0 + lane) * HV + h)))
                      : 0.f;
      bh = (lane + 32 < L)
               ? __bfloat162float(__ushort_as_bfloat16(
                     ldg16(beta + (size_t)(t0 + lane + 32) * HV + h)))
               : 0.f;
      }
      float d0 = ex2f(a * LOG2E), d1 = ex2f(b * LOG2E);
      float e0 = ex2f((glast - a) * LOG2E), e1 = ex2f((glast - b) * LOG2E);
      fac[0 * BT + lane] = d0;       fac[0 * BT + lane + 32] = d1;
      fac[1 * BT + lane] = e0;       fac[1 * BT + lane + 32] = e1;
      fac[2 * BT + lane] = a;        fac[2 * BT + lane + 32] = b;   // raw cumsum
      fac[3 * BT + lane] = 0.f;      fac[3 * BT + lane + 32] = 0.f;
      fac[4 * BT + lane] = bl;       fac[4 * BT + lane + 32] = bh;
      fac[5 * BT + lane] = 0.f;      fac[5 * BT + lane + 32] = 0.f;
      float *fo = fg + (size_t)(cid * HV + h) * (NF * BT);
      fo[0 * BT + lane] = d0;        fo[0 * BT + lane + 32] = d1;
      fo[1 * BT + lane] = e0;        fo[1 * BT + lane + 32] = e1;
      fo[2 * BT + lane] = a;         fo[2 * BT + lane + 32] = b;
      fo[3 * BT + lane] = 0.f;       fo[3 * BT + lane + 32] = 0.f;
      fo[4 * BT + lane] = bl;        fo[4 * BT + lane + 32] = bh;
      fo[5 * BT + lane] = 0.f;       fo[5 * BT + lane + 32] = 0.f;
      if (lane == 0) fo[5 * BT] = ex2f(glast * LOG2E);  // dlast
    }
    {
      const uint4 z4 = make_uint4(0, 0, 0, 0);
      for (int idx = gtid; idx < BT * (LDA / 8); idx += 128)
        *(uint4 *)(Tv + (idx / (LDA / 8)) * LDA + (idx % (LDA / 8)) * 8) = z4;
    }
    asm volatile("bar.sync %0, 128;" ::"r"(BAR_G) : "memory");

    {
      // ---- A = strict_lower( beta_i * exp(gc_i - gc_j) * GKK[i][j] ) ----
      constexpr int AUF = (RV & 64) ? ((RV & 512) ? 2 : 1) : NBT;
      if (!(SK & 4)) {
        float f0 = fac[4 * BT + bi * 16 + rr], f1 = fac[4 * BT + bi * 16 + rr + 8];
        float ai0 = fac[2 * BT + bi * 16 + rr], ai1 = fac[2 * BT + bi * 16 + rr + 8];
#pragma unroll AUF
        for (int bj = 0; bj < NBT; ++bj)
          if (bj <= bi) {
            uint32_t gr[4];
            if (RV & 16) ldAm(gr, GKK + bi * 16 * LDG + bj * 16, LDG, lane);
            else ldA(gr, GKK + bi * 16 * LDG + bj * 16, LDG, lane);
            const float gv[4] = {fac[2 * BT + bj * 16 + cc], fac[2 * BT + bj * 16 + cc + 1],
                                 fac[2 * BT + bj * 16 + cc + 8], fac[2 * BT + bj * 16 + cc + 9]};
            // v151 DIF: on the DIAGONAL tile the bf16 copy in `Am` has no consumer at all -- the
            // substitution below is its only reader and it reads j < i only, the blocked inverse
            // reads block-columns s < b, and the R epilogue overwrites every tile of `Am` later.
            // So the diagonal tile is written ONCE, as widened fp32, instead of once as bf16:
            // no extra store, and the substitution loses its 120 bf16-unpack ops.  The two
            // triangular row bases are hoisted out of the element loop so the packed index
            // costs a select, not a multiply.
            const bool dg = DIF && (bj == bi);
            float *ad0 = Ad + bi * 128 + ((rr * (rr - 1)) >> 1) + cc;
            float *ad1 = Ad + bi * 128 + (((rr + 8) * (rr + 7)) >> 1) + cc;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              int i = bi * 16 + rr + ACC_ROW(e);
              int j = bj * 16 + cc + ACC_COL(e);
              float fr = (e & 2) ? f1 : f0;
              float aj = gv[((e & 4) >> 1) | (e & 1)];
              float gj = ex2f(fminf(((e & 2) ? ai1 : ai0) - aj, 0.f) * LOG2E);
              float gm = __bfloat162float(
                  __ushort_as_bfloat16((unsigned short)(gr[e >> 1] >> ((e & 1) << 4))));
              // v152 TRI: on a strictly-lower tile `j <= bj*16+15 < bi*16 <= i` always holds,
              // so the select is dead there; only the diagonal tile needs it.
              float pv = (TRI && bj < bi) ? fr * gj * gm : ((j < i) ? fr * gj * gm : 0.f);
              __nv_bfloat16 av = __float2bfloat16_rn(pv);
              if (dg) {
                if (j < i) *(((e & 2) ? ad1 : ad0) + ((e & 4) ? 8 : 0) + (e & 1)) =
                    __bfloat162float(av);
              } else {
                Am[i * LDA + j] = av;
              }
            }
          }
      }
      asm volatile("bar.sync %0, 128;" ::"r"(BAR_G) : "memory");

      // ---- the four 16x16 diagonal inverses, fp32 substitution ----
      // v111 RV bit2 (XSM): the substitution vector lives in the fp32 smem scratch `Xs` rather
      // than 16 registers.  Values stay fp32 and the arithmetic order is unchanged, so the result
      // is BIT-identical; the cost is 120 extra smem loads on a chain that already has 16
      // dependent levels (the extra loads within one level are independent of each other).
      // v151 PI: `DIH` moves the four blocks onto TWO warps x both half-warps; `DIF` reads the
      // widened-fp32 copy the A epilogue staged, deleting the 120 bf16 unpack ops.  Both are
      // BIT-IDENTICAL (same per-column arithmetic, same values).
      if (!(SK & 8) && (DIH ? (gw < 2 && true) : (lane < 16))) {
        const int b = DIH ? ((gw << 1) | (lane >> 4)) : gw;
        const int col = DIH ? (lane & 15) : lane;
        const __nv_bfloat16 *Nb = Am + (b * 16) * LDA + b * 16;
        const float *Ab = Ad + b * 128;
        if ((RV & 4) && !DIH && !DIF) {
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            float a = (lane == i) ? 1.f : 0.f;
#pragma unroll
            for (int j = 0; j < 16; ++j)
              if (j < i) a -= __bfloat162float(Nb[i * LDA + j]) * Xs[j * 16 + lane];
            Xs[i * 16 + lane] = a;
          }
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            __nv_bfloat16 xv = __float2bfloat16_rn(Xs[i * 16 + lane]);
            if (!(RV & 256)) Xi[b * 16 * LDX + i * LDX + lane] = xv;
            Tv[(b * 16 + i) * LDA + b * 16 + lane] = xv;
          }
        } else {
          float X[16];
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            float a = (col == i) ? 1.f : 0.f;
#pragma unroll
            for (int j = 0; j < 16; ++j)
              if (j < i)
                a -= (DIF ? Ab[((i * (i - 1)) >> 1) + j]
                          : __bfloat162float(Nb[i * LDA + j])) * X[j];
            X[i] = a;
          }
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            __nv_bfloat16 xv = __float2bfloat16_rn(X[i]);
            if (!(RV & 256)) Xi[b * 16 * LDX + i * LDX + col] = xv;
            Tv[(b * 16 + i) * LDA + b * 16 + col] = xv;
          }
        }
      }
      asm volatile("bar.sync %0, 128;" ::"r"(BAR_G) : "memory");

      // ---- full (I+A)^-1 ----
      constexpr int IUF = (RV & 32) ? ((RV & 512) ? 2 : 1) : NBT;
      // v152 BI2: the two-level 32x32 split.  L = [[L11,0],[A21,L22]] with 32x32 blocks gives
      // L^-1 = [[L11^-1,0],[-L22^-1 A21 L11^-1, L22^-1]], so the whole 64x64 inverse is
      //   step 1  the two 32x32 diagonal inverses, INDEPENDENT -> warps 0 and 1 in parallel
      //   step 2  Y = A(2:4,0:2) . Tinv(0:2)   then   T(2:4,0:2) = -Tinv(2:4) . Y
      // Step 2's warp owns a block-COLUMN j and computes BOTH of its Y row-tiles, so Y never
      // crosses a warp and only the step1/step2 boundary needs a group barrier.  Depth 4 dependent
      // smem-mediated MMA levels instead of 6; MMA count is unchanged at 16 per group-pass
      // (step1 2+2; step2 j=0: 4 Y + 3 T, j=1: 2 Y + 3 T).
      if (BI2 && !(SK & 16)) {
        if (gw < 2) {
          const int pp = gw, br = 2 * pp + 1, bc = 2 * pp;
          float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
          uint32_t ra[4], rb[4];
          ldA(ra, Am + (br * 16) * LDA + bc * 16, LDA, lane);      // A[br][bc]
          ldBr(rb, Tv + (bc * 16) * LDA + bc * 16, LDA, lane);     // Xd[bc]
          mma_rr(acc, ra, rb);
          __syncwarp();
#pragma unroll
          for (int e = 0; e < 8; ++e)
            Yt[pp * 16 * LDX + (rr + ACC_ROW(e)) * LDX + cc + ACC_COL(e)] =
                __float2bfloat16_rn(acc[e]);
          __syncwarp();
          float o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
          ldA(ra, Tv + (br * 16) * LDA + br * 16, LDA, lane);      // Xd[br]
          ldBr(rb, Yt + pp * 16 * LDX, LDX, lane);
          mma_rr(o, ra, rb);
          __syncwarp();
#pragma unroll
          for (int e = 0; e < 8; ++e)
            Tv[(br * 16 + rr + ACC_ROW(e)) * LDA + bc * 16 + cc + ACC_COL(e)] =
                __float2bfloat16_rn(-o[e]);
        }
        asm volatile("bar.sync %0, 128;" ::"r"(BAR_G) : "memory");
        if (gw < 2) {
          const int j = gw;
          float y2[8] = {0, 0, 0, 0, 0, 0, 0, 0}, y3[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
          for (int sq = 0; sq < 2; ++sq)
            if (sq >= j) {                       // Tinv[s][j] == 0 for s < j
              uint32_t rb[4], ra2[4], ra3[4];
              ldBr(rb, Tv + (sq * 16) * LDA + j * 16, LDA, lane);
              ldA(ra2, Am + (2 * 16) * LDA + sq * 16, LDA, lane);
              ldA(ra3, Am + (3 * 16) * LDA + sq * 16, LDA, lane);
              mma_rr(y2, ra2, rb);
              mma_rr(y3, ra3, rb);
            }
          __syncwarp();
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int ro = (rr + ACC_ROW(e)) * LDX + cc + ACC_COL(e);
            Yt[(j * 2 + 0) * 16 * LDX + ro] = __float2bfloat16_rn(y2[e]);
            Yt[(j * 2 + 1) * 16 * LDX + ro] = __float2bfloat16_rn(y3[e]);
          }
          __syncwarp();
          float t2[8] = {0, 0, 0, 0, 0, 0, 0, 0}, t3[8] = {0, 0, 0, 0, 0, 0, 0, 0};
          uint32_t rby2[4], rby3[4], ra[4];
          ldBr(rby2, Yt + (j * 2 + 0) * 16 * LDX, LDX, lane);
          ldBr(rby3, Yt + (j * 2 + 1) * 16 * LDX, LDX, lane);
          ldA(ra, Tv + (2 * 16) * LDA + 2 * 16, LDA, lane);        // Tinv[2][2]
          mma_rr(t2, ra, rby2);
          ldA(ra, Tv + (3 * 16) * LDA + 2 * 16, LDA, lane);        // Tinv[3][2]
          mma_rr(t3, ra, rby2);
          ldA(ra, Tv + (3 * 16) * LDA + 3 * 16, LDA, lane);        // Tinv[3][3]
          mma_rr(t3, ra, rby3);
          __syncwarp();
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int co = j * 16 + cc + ACC_COL(e);
            Tv[(2 * 16 + rr + ACC_ROW(e)) * LDA + co] = __float2bfloat16_rn(-t2[e]);
            Tv[(3 * 16 + rr + ACC_ROW(e)) * LDA + co] = __float2bfloat16_rn(-t3[e]);
          }
        }
      }
      // ---- v51 form: warp j owns block-column j, sequential over b>j ----
      if (!BI2 && !(SK & 16)) {
        const int j = gw;
#pragma unroll IUF
        for (int b = 1; b < NBT; ++b) {
          if (b > j) {
            float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int s = 0; s < NBT; ++s)
              if (s >= j && s < b) {
                uint32_t ra[4], rb[4];
                ldA(ra, Am + (b * 16) * LDA + s * 16, LDA, lane);
                ldBr(rb, Tv + (s * 16) * LDA + j * 16, LDA, lane);
                mma_rr(acc, ra, rb);
              }
            __syncwarp();
#pragma unroll
            for (int e = 0; e < 8; ++e)
              Yt[j * 16 * LDX + (rr + ACC_ROW(e)) * LDX + cc + ACC_COL(e)] =
                  __float2bfloat16_rn(acc[e]);
            __syncwarp();
            float o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            uint32_t ra[4], rb[4];
            if (RV & 256) ldA(ra, Tv + (b * 16) * LDA + b * 16, LDA, lane);
            else ldA(ra, Xi + b * 16 * LDX, LDX, lane);
            ldBr(rb, Yt + j * 16 * LDX, LDX, lane);
            mma_rr(o, ra, rb);
            __syncwarp();
#pragma unroll
            for (int e = 0; e < 8; ++e)
              Tv[(b * 16 + rr + ACC_ROW(e)) * LDA + j * 16 + cc + ACC_COL(e)] =
                  __float2bfloat16_rn(-o[e]);
            __syncwarp();
          }
        }
      }
      asm volatile("bar.sync %0, 128;" ::"r"(BAR_G) : "memory");

      // ---- emit the inverse: 64x64 bf16, row-major, stride 64 ----
      if (!(SK & 32)) {
        __nv_bfloat16 *dst = tg + (size_t)(cid * HV + h) * (BT * BT);
        for (int idx = gtid; idx < BT * (BT / 8); idx += 128) {
          int i = idx / (BT / 8), c8 = (idx % (BT / 8)) << 3;
          *(uint4 *)(dst + i * BT + c8) = *(const uint4 *)(Tv + i * LDA + c8);
        }
      }
    }
    if (!(SK & 64)) {
      // ---- R = exp(gc_i - gc_j) * GQK[i][j] -> staged in Am, then emitted 16 B/lane ----
      constexpr int RUF = (RV & 128) ? ((RV & 512) ? 2 : 1) : NBT;
      float fA0 = fac[2 * BT + bi * 16 + rr], fA1 = fac[2 * BT + bi * 16 + rr + 8];
#pragma unroll RUF
      for (int bj = 0; bj < NBT; ++bj) {
        if (bj <= bi) {
          uint32_t gr[4];
          if (RV & 8) ldAm(gr, GQK + bi * 16 * LDG + bj * 16, LDG, lane);
          else ldA(gr, GQK + bi * 16 * LDG + bj * 16, LDG, lane);
          const float a0 = fac[2 * BT + bj * 16 + cc], a1 = fac[2 * BT + bj * 16 + cc + 1];
          const float a2 = fac[2 * BT + bj * 16 + cc + 8],
                      a3 = fac[2 * BT + bj * 16 + cc + 9];
#pragma unroll
          for (int p = 0; p < 4; ++p) {
            const int e = p * 2;
            int i = bi * 16 + rr + ACC_ROW(e);
            int j = bj * 16 + cc + ACC_COL(e);
            float ai = (e & 2) ? fA1 : fA0;
            float ga = ex2f(fminf(ai - ((e & 4) ? a2 : a0), 0.f) * LOG2E);
            float gb = ex2f(fminf(ai - ((e & 4) ? a3 : a1), 0.f) * LOG2E);
            float m0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(gr[p] & 0xffffu)));
            float m1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(gr[p] >> 16)));
            // v152 TRI: strictly-lower tiles satisfy j+1 <= i unconditionally (see the A
            // epilogue note); only the diagonal tile keeps the two selects.
            float lo = (TRI && bj < bi) ? ga * m0 : ((j <= i) ? ga * m0 : 0.f);
            float hi = (TRI && bj < bi) ? gb * m1 : ((j + 1 <= i) ? gb * m1 : 0.f);
            *(uint32_t *)(Am + i * LDA + j) = cvt_bf16x2(lo, hi);
          }
        } else if (!RZE) {
#pragma unroll
          for (int p = 0; p < 4; ++p) {
            const int e = p * 2;
            int i = bi * 16 + rr + ACC_ROW(e);
            int j = bj * 16 + cc + ACC_COL(e);
            *(uint32_t *)(Am + i * LDA + j) = 0u;
          }
        }
      }
      asm volatile("bar.sync %0, 128;" ::"r"(BAR_G) : "memory");
      __nv_bfloat16 *dst = rg + (size_t)(cid * HV + h) * (BT * BT);
      for (int idx = gtid; idx < BT * (BT / 8); idx += 128) {
        int i = idx / (BT / 8), c8 = (idx % (BT / 8)) << 3;
        // v153 RZE: the strictly-upper 16x16 tiles of R are exact zeros; emit them without ever
        // staging them through `Am` (one uint4 of zeros instead of a predicated smem store in
        // accumulator order plus this read).
        *(uint4 *)(dst + i * BT + c8) =
            (RZE && c8 >= ((i >> 4) + 1) * 16) ? make_uint4(0, 0, 0, 0)
                                               : *(const uint4 *)(Am + i * LDA + c8);
      }
    }
    asm volatile("bar.sync %0, 128;" ::"r"(BAR_G) : "memory");
  }
}
// v111: three launch-bound flavours of the same body.  `__launch_bounds__(256, 3)` tells ptxas to
// budget 131072/(256*3) = 170 registers, which is exactly the 3-blocks/SM threshold.  v22 showed a
// forced 170-reg budget is catastrophic WHEN IT SPILLS (0.67x), so every flavour is reported with
// its `localSizeBytes` by `occ2()` before it is timed.
#define GDR_PROLOGUE_WRAP(NAME, LB)                                                              \
  template <int NGRP, bool GLDM, bool PCPA, int RV, int SK, int PI = 0, int VK = 0>              \
  __global__ LB void NAME(                                                                       \
      const __nv_bfloat16 *__restrict__ k, const __nv_bfloat16 *__restrict__ q,                  \
      const void *__restrict__ g, const __nv_bfloat16 *__restrict__ beta,                        \
      const int *__restrict__ cu, __nv_bfloat16 *__restrict__ tg,                                \
      __nv_bfloat16 *__restrict__ rg, float *__restrict__ fg,                                    \
      __nv_bfloat16 *__restrict__ kn, __nv_bfloat16 *__restrict__ qn,                            \
      int HQ, int HV, int rep, int gf) {                                                          \
    gdr_prologue_body<NGRP, GLDM, PCPA, RV, SK, PI, VK>(k, q, g, beta, cu, tg, rg, fg, kn, qn,    \
                                                        HQ, HV, rep, gf);                        \
  }
GDR_PROLOGUE_WRAP(gdr_prologue, __launch_bounds__(NGRP * 128))
#undef GDR_PROLOGUE_WRAP

template <int NGRP, int RV>
static size_t smem_prologue() {
  constexpr int LDK = KD + 8, LDA = BT + 8, LDX = 24, NBT = BT / 16;
  size_t grams = (size_t)BT * LDG * 2 * 2;
  size_t grpb = (size_t)BT * LDA * 2 * 2 + (size_t)NBT * 16 * LDX * 2 * ((RV & 256) ? 1 : 2);
  size_t ph1 = (size_t)BT * LDK * 2 * 2;
  size_t ph2 = NGRP * grpb + (size_t)NGRP * NF * BT * 4;
  if (RV & 4) ph2 += (size_t)NGRP * 4 * 256 * 4;  // v111 XSM scratch
  return grams + (ph1 > ph2 ? ph1 : ph2);
}

//   * A-operand fragments K[bi] and Q[bi] are each loaded ONCE per k-tile and feed BOTH
//     P = K S0^T and T1 = Q S0^T for this warp's VH = NVT/2 v-tiles, and the R = Q K^T tiles
//     whose column band matches vh's parity.  Sharing the Sb B-fragment between P and T1 is what
//     cuts the critical warp from 9 fragment loads / 8 MMA per k-tile (v05, where warps 0-3 did
//     all of P and warps 4-7 all of R+T1) to 6 / 6, and it balances all eight warps across the
//     barrier instead of leaving a 1.8x split.
//   * shared-memory epilogue stores are packed: the wmma accumulator's elements e and e+1 are
//     adjacent columns of the same row, so `cvt.rn.bf16x2` + one `st.shared.b32` replaces two
//     `st.shared.u16` for Rm, UT and US.
// ===== v26 (SEGSCAN): cp.async / AIU staging with a 2-stage software pipeline ==================
// MEASURED motivation (workspace/v23/ablate.py, sid0, the kernel's own stage mask):
//     loop + 3x __syncthreads only ................  141 us
//   + cross-chunk GLOBAL prefetch ................. +373 us   <-- 100% ADDITIVE, hidden by nothing
//   + the 4 smem commit loops ..................... + 88 us
//   + every MMA stage ............................. +580 us
//                                                   =1173 us  (== the measured gdr_scan)
// g0 prefetches chunk c+1 into a 14 x uint4 REGISTER array (56 vreg) and copies it to shared at
// the top of the next iteration, so the warps themselves pay for the move and it does not overlap
// the MMA stream at all.  ppu15lab has the cure measured: `cp.async` (AIU, global->TSM, bypasses
// the register file, 2172 GB/s, knee at 2 stages) is worth 2.047x over synchronous smem staging
// (t:techniques/async_double_buffer, i:ppu.cp.async.*).
//
// Pipeline built here:
//   * K, Q, V are DOUBLE-buffered and issued one chunk ahead (group A), so the AIU moves chunk
//     c+1 while the MMAs chew chunk c.
//   * T, R, fac stay single-buffered -- they are first touched only in the U / output phases, so
//     issuing them (group B) right after the top barrier already buys 1/3..2/3 of an iteration of
//     overlap without a second copy.  `__pipeline_wait_prior(1)` waits for B while A stays in
//     flight, which is the whole point of ordering B before A.
//   * `US` is aliased onto `Sb`: Sb's last read (the P/T1 GEMM) is separated from US's first write
//     (the U phase) by a __syncthreads, and US's last read (state update) from Sb's next write by
//     the top-of-loop barrier.  That alias is what buys the smem back for the K/Q/V second stage
//     and keeps the block at 2 blocks/SM.
template <int VB, int VAR = 2>
__global__ __launch_bounds__(NW * 32) void gdr_scan(
    const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v, const __nv_bfloat16 *__restrict__ tg,
    const __nv_bfloat16 *__restrict__ rg,
    const float *__restrict__ fg, const float *__restrict__ s0, const int *__restrict__ cu,
    __nv_bfloat16 *__restrict__ out, float *__restrict__ sf, int HQ, int HV, int rep,
    float scale, int mask) {
  constexpr int LDK = KD + 8;
  constexpr int LDS = KD + 8;
  constexpr int LDA = BT + 8;
  constexpr int LDU = BT + 8;
  constexpr int LDV = VB + 8;
  constexpr int NVT = VB / 16;
  constexpr int NBT = BT / 16;
  constexpr int VH = (NVT >= 2) ? (NVT / 2) : 1;  // v-tiles per warp

  extern __shared__ char smem_raw[];
  char *sp = smem_raw;
  __nv_bfloat16 *Sb = (__nv_bfloat16 *)sp;  sp += VB * LDS * 2;
  __nv_bfloat16 *Kb = (__nv_bfloat16 *)sp;  sp += BT * LDK * 2 * 2;   // 2 stages
  __nv_bfloat16 *Qb = (__nv_bfloat16 *)sp;  sp += BT * LDK * 2 * 2;   // 2 stages
  __nv_bfloat16 *Vb = (__nv_bfloat16 *)sp;  sp += BT * LDV * 2 * 2;   // 2 stages
  __nv_bfloat16 *Tm = (__nv_bfloat16 *)sp;  sp += BT * LDA * 2 * 2;   // 2 stages
  __nv_bfloat16 *Rm = (__nv_bfloat16 *)sp;  sp += BT * LDA * 2 * 2;   // 2 stages
  float *fac = (float *)sp;                 sp += NF * BT * 4 * 2;    // 2 stages
  // ZT+UT normally live inside Qb[cur]; only a V tile so wide that they no longer fit needs a
  // separate allocation (VB=32/16 alias; the alias is what pays for the T/R/fac second stage).
  constexpr bool ALIAS_ZU = (3 * VB * LDU) <= (BT * LDK);
  __nv_bfloat16 *ZUx = (__nv_bfloat16 *)sp;
  sp += ALIAS_ZU ? 0 : (3 * VB * LDU * 2);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int rr = lane >> 2, cc = (lane & 3) << 1;
  const int v0 = blockIdx.x * VB;
  const int h = blockIdx.y;
  const int rq = blockIdx.z;
  const int hq = h / rep;

#ifdef FLASHQLA_EXACT_T
  const int t_start = 0;
  const int t_end = FLASHQLA_EXACT_T;
  const int nchunk = (FLASHQLA_EXACT_T + BT - 1) / BT;
  const int coff = 0;
#else
  const int t_start = cu[rq];
  const int t_end = cu[rq + 1];
  const int nchunk = (t_end - t_start + BT - 1) / BT;
  int coff = 0;
  for (int r = 0; r < rq; ++r) coff += (cu[r + 1] - cu[r] + BT - 1) / BT;
#endif

  const int bi = warp & (NBT - 1);
  const int vh = warp >> 2;
  const int bv0 = vh * VH;              // first v-tile owned by this warp

  float S[NVT][8];
  {
    const float *base = s0 + (((size_t)rq * HV + h) * VD + v0) * KD + warp * 16;
#pragma unroll
    for (int vt = 0; vt < NVT; ++vt)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        int vrow = vt * 16 + rr + half * 8;
        float2 a = ldg64f(base + (size_t)vrow * KD + cc);
        float2 b = ldg64f(base + (size_t)vrow * KD + cc + 8);
        S[vt][0 + half * 2] = a.x;
        S[vt][1 + half * 2] = a.y;
        S[vt][4 + half * 2] = b.x;
        S[vt][5 + half * 2] = b.y;
      }
  }

  constexpr int NKQ = BT * 16 / (NW * 32);
  constexpr int NVI = (BT * (VB / 8) + NW * 32 - 1) / (NW * 32);
  constexpr int NTI = (BT * (BT / 8) + NW * 32 - 1) / (NW * 32);
  const uint4 Z4 = make_uint4(0, 0, 0, 0);

  // group A: K, Q, V of chunk CC into stage BUF (double-buffered, issued one chunk ahead)
#define GDR_STAGE_A(BUF, CC)                                                                 \
  {                                                                                          \
    const int _c = (CC);                                                                     \
    const int _t0 = t_start + _c * BT;                                                       \
    const int _L = min(BT, t_end - _t0);                                                     \
    __nv_bfloat16 *_kb = Kb + (BUF) * (BT * LDK);                                            \
    __nv_bfloat16 *_qb = Qb + (BUF) * (BT * LDK);                                            \
    __nv_bfloat16 *_vb = Vb + (BUF) * (BT * LDV);                                            \
    _Pragma("unroll") for (int u = 0; u < NKQ; ++u) {                                        \
      int idx = tid + u * NW * 32, i = idx >> 4, c8 = (idx & 15) << 3;                       \
      if (i < _L) {                                                                          \
        __pipeline_memcpy_async(_kb + i * LDK + c8,                                          \
                                k + ((size_t)(_t0 + i) * HQ + hq) * KD + c8, 16);            \
        __pipeline_memcpy_async(_qb + i * LDK + c8,                                          \
                                q + ((size_t)(_t0 + i) * HQ + hq) * KD + c8, 16);            \
      } else {                                                                               \
        *(uint4 *)(_kb + i * LDK + c8) = Z4;                                                 \
        *(uint4 *)(_qb + i * LDK + c8) = Z4;                                                 \
      }                                                                                      \
    }                                                                                        \
    _Pragma("unroll") for (int u = 0; u < NVI; ++u) {                                        \
      int idx = tid + u * NW * 32;                                                           \
      if (idx < BT * (VB / 8)) {                                                             \
        int i = idx / (VB / 8), c8 = (idx % (VB / 8)) << 3;                                  \
        if (i < _L)                                                                          \
          __pipeline_memcpy_async(_vb + i * LDV + c8,                                        \
                                  v + ((size_t)(_t0 + i) * HV + h) * VD + v0 + c8, 16);      \
        else                                                                                 \
          *(uint4 *)(_vb + i * LDV + c8) = Z4;                                               \
      }                                                                                      \
    }                                                                                        \
  }

  // group B: T, R, fac of chunk CC into stage BUF
#define GDR_STAGE_B(BUF, CC)                                                                 \
  {                                                                                          \
    const int _cid = coff + (CC);                                                            \
    __nv_bfloat16 *_tm = Tm + (BUF) * (BT * LDA);                                            \
    __nv_bfloat16 *_rm = Rm + (BUF) * (BT * LDA);                                            \
    float *_fc = fac + (BUF) * (NF * BT);                                                     \
    const __nv_bfloat16 *_ts = tg + (size_t)(_cid * HV + h) * (BT * BT);                     \
    const __nv_bfloat16 *_rs = rg + (size_t)(_cid * HV + h) * (BT * BT);                     \
    _Pragma("unroll") for (int u = 0; u < NTI; ++u) {                                        \
      int idx = tid + u * NW * 32;                                                           \
      if (idx < BT * (BT / 8)) {                                                             \
        int i = idx / (BT / 8), c8 = (idx % (BT / 8)) << 3;                                  \
        __pipeline_memcpy_async(_tm + i * LDA + c8, _ts + i * BT + c8, 16);                  \
        __pipeline_memcpy_async(_rm + i * LDA + c8, _rs + i * BT + c8, 16);                  \
      }                                                                                      \
    }                                                                                        \
    if (tid < NF * BT / 4)                                                                   \
      __pipeline_memcpy_async(_fc + tid * 4,                                                 \
                              fg + (size_t)(_cid * HV + h) * (NF * BT) + tid * 4, 16);       \
  }

  GDR_STAGE_A(0, 0)
  GDR_STAGE_B(0, 0)
  __pipeline_commit();
#pragma unroll
    for (int vt = 0; vt < NVT; ++vt)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        __nv_bfloat16 *d = Sb + (vt * 16 + rr + half * 8) * LDS + warp * 16 + cc;
        *(uint32_t *)d = cvt_bf16x2(S[vt][0 + half * 2], S[vt][1 + half * 2]);
        *(uint32_t *)(d + 8) = cvt_bf16x2(S[vt][4 + half * 2], S[vt][5 + half * 2]);
      }


#pragma unroll 1
  for (int c = 0; c < nchunk; ++c) {
    const int t0 = t_start + c * BT;
    const int L = min(BT, t0 < t_end ? t_end - t0 : 0);
    const int cur = c & 1;
    const __nv_bfloat16 *Kc = Kb + cur * (BT * LDK);
    const __nv_bfloat16 *Qc = Qb + cur * (BT * LDK);
    const __nv_bfloat16 *Vc = Vb + cur * (BT * LDV);
    const __nv_bfloat16 *Tc = Tm + cur * (BT * LDA);
    const __nv_bfloat16 *Rc = Rm + cur * (BT * LDA);
    const float *facc = fac + cur * (NF * BT);
    // ZT/UT are carved out of Qb[cur]: its last read is the GEMM phase of THIS iteration and its
    // next write is chunk c+2's stage, issued one whole iteration later -- a __syncthreads apart
    // on both sides.  That alias is what pays for the T/R/fac second stage.
    __nv_bfloat16 *ZT = ALIAS_ZU ? (Qb + cur * (BT * LDK)) : ZUx;
    __nv_bfloat16 *UT = ZT + VB * LDU;
    __nv_bfloat16 *US = UT + VB * LDU;

    __pipeline_wait_prior(0);   // chunk c's whole stage buffer has landed
    __syncthreads();            // ... Sb published, and last iteration's ZT/UT/US reads retired

    if (c + 1 < nchunk) {
      GDR_STAGE_A(cur ^ 1, c + 1)
      GDR_STAGE_B(cur ^ 1, c + 1)
    }
    __pipeline_commit();

    // ---- GEMM phase: P[bi][bv] = K S0^T and T1[bi][bv] = Q S0^T share the Sb B-fragment ----
    float accP[VH][8], accT1[VH][8];
#pragma unroll
    for (int t = 0; t < VH; ++t)
#pragma unroll
      for (int e = 0; e < 8; ++e) { accP[t][e] = 0.f; accT1[t][e] = 0.f; }
    if (mask & 1) {
#pragma unroll
      for (int kk = 0; kk < KD / 16; ++kk) {
        uint32_t rak[4], raq[4];
        if (VAR & 2) {
          ldAm(rak, Kc + bi * 16 * LDK + kk * 16, LDK, lane);
          ldAm(raq, Qc + bi * 16 * LDK + kk * 16, LDK, lane);
        } else {
          ldA(rak, Kc + bi * 16 * LDK + kk * 16, LDK, lane);
          ldA(raq, Qc + bi * 16 * LDK + kk * 16, LDK, lane);
        }
#pragma unroll
        for (int t = 0; t < VH; ++t) {
          if (bv0 + t < NVT) {
            uint32_t rb[4];
            if (VAR & 2) ldAm(rb, Sb + (bv0 + t) * 16 * LDS + kk * 16, LDS, lane);
            else ldA(rb, Sb + (bv0 + t) * 16 * LDS + kk * 16, LDS, lane);
            mma_rc_t(accP[t], rak, rb);
            mma_rc_t(accT1[t], raq, rb);
          }
        }
      }
    }
    __syncthreads();            // Sb's last read retired before ZT/US reuse the same bytes
    const float *fdd = facc + 0 * BT, *fex = facc + 1 * BT, *fbt = facc + 4 * BT;
    const float dlast = facc[5 * BT];

    // ---- Z^T = [ beta_i (v_i - dd_i P_i) ]^T ----
    if (mask & 4) {
#pragma unroll
      for (int t = 0; t < VH; ++t)
        if (bv0 + t < NVT) {
#pragma unroll
          for (int p = 0; p < 4; ++p) {
            const int e = p * 2;
            int i = bi * 16 + rr + ACC_ROW(e);
            int vv = (bv0 + t) * 16 + cc + ACC_COL(e);
            uint32_t vpair = lds32(Vc + i * LDV + vv);   // V[i][vv], V[i][vv+1] in one 32-bit load
            float b = fbt[i], d = fdd[i];
            float z0 = b * (__bfloat162float(__ushort_as_bfloat16((unsigned short)(vpair & 0xffffu)))
                            - d * accP[t][e]);
            float z1 = b * (__bfloat162float(__ushort_as_bfloat16((unsigned short)(vpair >> 16)))
                            - d * accP[t][e + 1]);
            ZT[vv * LDU + i] = __float2bfloat16_rn(z0);
            ZT[(vv + 1) * LDU + i] = __float2bfloat16_rn(z1);
          }
        }
    }
    __syncthreads();

    // ---- U^T = Z^T (I+A)^-T : one MMA chain per (v-tile, row-tile) ----
    if (mask & 16) {
      if (NVT * NBT > NW) {
        const int wg = warp >> 2;
        const int wl = warp & (NBT - 1);
        const int bv1 = wg * VH + (wl >> 1);
        const int bj1 = (NBT - 1) - wl;
        const int bv2 = (1 - wg) * VH + (wl >> 1);
        const int bj2 = wl;
        float u1[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        float u2[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int s = 0; s < NBT; ++s) {
          if (s <= bj1) {
            uint32_t ra[4], rb[4];
            ldA(ra, ZT + bv1 * 16 * LDU + s * 16, LDU, lane);
            ldA(rb, Tc + bj1 * 16 * LDA + s * 16, LDA, lane);
            mma_rc_t(u1, ra, rb);
          }
          if (s <= bj2) {
            uint32_t ra[4], rb[4];
            ldA(ra, ZT + bv2 * 16 * LDU + s * 16, LDU, lane);
            ldA(rb, Tc + bj2 * 16 * LDA + s * 16, LDA, lane);
            mma_rc_t(u2, ra, rb);
          }
        }
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          const int e = p * 2;
          int vv1 = bv1 * 16 + rr + ACC_ROW(e), j1 = bj1 * 16 + cc + ACC_COL(e);
          *(uint32_t *)(UT + vv1 * LDU + j1) = cvt_bf16x2(u1[e], u1[e + 1]);
          *(uint32_t *)(US + vv1 * LDU + j1) = cvt_bf16x2(u1[e] * fex[j1], u1[e + 1] * fex[j1 + 1]);
          int vv2 = bv2 * 16 + rr + ACC_ROW(e), j2 = bj2 * 16 + cc + ACC_COL(e);
          *(uint32_t *)(UT + vv2 * LDU + j2) = cvt_bf16x2(u2[e], u2[e + 1]);
          *(uint32_t *)(US + vv2 * LDU + j2) = cvt_bf16x2(u2[e] * fex[j2], u2[e + 1] * fex[j2 + 1]);
        }
      } else {
        for (int ti = warp; ti < NVT * NBT; ti += NW) {
          const int bv = ti >> 2, bj = ti & (NBT - 1);
          float u[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
          for (int s = 0; s < NBT; ++s)
            if (s <= bj) {
              uint32_t ra[4], rb[4];
              ldA(ra, ZT + bv * 16 * LDU + s * 16, LDU, lane);
              ldA(rb, Tc + bj * 16 * LDA + s * 16, LDA, lane);
              mma_rc_t(u, ra, rb);
            }
#pragma unroll
          for (int p = 0; p < 4; ++p) {
            const int e = p * 2;
            int vv = bv * 16 + rr + ACC_ROW(e);
            int j = bj * 16 + cc + ACC_COL(e);
            *(uint32_t *)(UT + vv * LDU + j) = cvt_bf16x2(u[e], u[e + 1]);
            *(uint32_t *)(US + vv * LDU + j) =
                cvt_bf16x2(u[e] * fex[j], u[e + 1] * fex[j + 1]);
          }
        }
      }
    }
    __syncthreads();

    // ---- interleaved state update + output for ILP ----
    if (mask & 96) {
#pragma unroll
      for (int vt = 0; vt < NVT; ++vt)
#pragma unroll
        for (int e = 0; e < 8; ++e) S[vt][e] *= dlast;
      float d0 = fdd[bi * 16 + rr], d1 = fdd[bi * 16 + rr + 8];
#pragma unroll
      for (int t = 0; t < VH; ++t)
#pragma unroll
        for (int e = 0; e < 8; ++e) accT1[t][e] *= (e & 2) ? d1 : d0;
#pragma unroll
      for (int s = 0; s < NBT; ++s) {
        uint32_t rb_k[4];
        ldBr(rb_k, Kc + s * 16 * LDK + warp * 16, LDK, lane);
#pragma unroll
        for (int vt = 0; vt < NVT; ++vt) {
          uint32_t ra_us[4];
          ldA(ra_us, US + vt * 16 * LDU + s * 16, LDU, lane);
          mma_rr(S[vt], ra_us, rb_k);
        }
        if (s <= bi) {
          uint32_t ra_rm[4];
          ldA(ra_rm, Rc + bi * 16 * LDA + s * 16, LDA, lane);
#pragma unroll
          for (int t = 0; t < VH; ++t)
            if (bv0 + t < NVT) {
              uint32_t rb_ut[4];
              ldA(rb_ut, UT + (bv0 + t) * 16 * LDU + s * 16, LDU, lane);
              mma_rc_t(accT1[t], ra_rm, rb_ut);
            }
        }
      }
#pragma unroll
      for (int t = 0; t < VH; ++t)
        if (bv0 + t < NVT) {
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            int i = bi * 16 + rr + half * 8;
            if (i < L) {
              __nv_bfloat16 *dst =
                  out + ((size_t)(t0 + i) * HV + h) * VD + v0 + (bv0 + t) * 16 + cc;
              *(uint32_t *)dst =
                  cvt_bf16x2(accT1[t][0 + half * 2] * scale, accT1[t][1 + half * 2] * scale);
              *(uint32_t *)(dst + 8) =
                  cvt_bf16x2(accT1[t][4 + half * 2] * scale, accT1[t][5 + half * 2] * scale);
            }
          }
        }
    }
    // ---- bf16 snapshot of the NEXT chunk's entry state; Sb's readers (the P/T1 GEMM) are two
    //      barriers behind, so this needs no barrier of its own. ----
#pragma unroll
    for (int vt = 0; vt < NVT; ++vt)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        __nv_bfloat16 *d = Sb + (vt * 16 + rr + half * 8) * LDS + warp * 16 + cc;
        *(uint32_t *)d = cvt_bf16x2(S[vt][0 + half * 2], S[vt][1 + half * 2]);
        *(uint32_t *)(d + 8) = cvt_bf16x2(S[vt][4 + half * 2], S[vt][5 + half * 2]);
      }
  }

  {
    float *base = sf + (((size_t)rq * HV + h) * VD + v0) * KD + warp * 16;
#pragma unroll
    for (int vt = 0; vt < NVT; ++vt)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float *d = base + (size_t)(vt * 16 + rr + half * 8) * KD + cc;
        uint2 a, b;
        a.x = __float_as_uint(S[vt][0 + half * 2]);
        a.y = __float_as_uint(S[vt][1 + half * 2]);
        b.x = __float_as_uint(S[vt][4 + half * 2]);
        b.y = __float_as_uint(S[vt][5 + half * 2]);
        stg64(d, a);
        stg64(d + 8, b);
      }
  }
#undef GDR_STAGE_A
#undef GDR_STAGE_B
}

// ===== v45: TWO 8-warp groups per block sharing the staged K/Q/T/R/fac tiles ===================
// MEASURED problem (v43 NOTES): the scan issues 841 MB of global traffic for 163 MB of unique data
// (5.2x), and 492 MB of that -- 58 % -- is q/k re-read 15.6x, because the 4 V-blocks x 4 GQA value
// heads of one (chunk, key-head) all stage the same 32 KB.  The `GDR_VB` sweep showed the load path
// is **Little's-law bound, not bandwidth bound**: 27 B/cyc/SM at 8 warps/SM (VB=64, 32 blocks) vs
// 54 B/cyc/SM at 16 warps/SM (VB=32, 64 blocks) -- exactly linear in warps -- and 2.74 TB/s already
// exceeds ppu15lab's measured cp.async ceiling (f:ALT/02, 2172 GB/s).  So fewer bytes only pays if
// 16 warps/SM is preserved, which rules out plain VB=64 (measured 1816 us).
//
// This kernel keeps 16 warps/SM AND halves the bytes: ONE 512-thread block owns 64 V rows as TWO
// INDEPENDENT 8-warp groups of 32 rows each, sharing one staged copy of K, Q, T, R and fac.
//   bytes per SM per chunk: 107 KB (two VB=32 blocks) -> 58.3 KB (one 2-group block)
//   K 16 KB + Q 16 KB + T 8 KB + R 8 KB + fac 1.5 KB shared + V 8 KB for both groups
// The two groups synchronise on their OWN named barrier (`bar.sync grp+1, 256`), so a barrier never
// drains the whole block -- that is what makes this different from v24 (1024 threads, one cohort,
// every barrier stalled all 32 warps with no sibling block: 0.37x).  Only the once-per-chunk
// pipeline barrier is block-wide.  ZT/UT/US get their own per-group region instead of aliasing
// Qb[cur] (Qb is now shared, so the alias would need a block-wide barrier), which also removes
// v28's 4th barrier: 4 barriers/chunk -> 1 block-wide + 2 per-group.
#define NW2 16  // warps per 2-group block

// ===== v51 U-solve knobs (UV bitmask) =========================================================
// MEASURED context (LEDGER per-stage table): the U solve costs 150 us for only 20 MMA-counts
// (7.5 us/MMA) against the P/T1 GEMM's 1.02.  v42/v46 showed it is neither load-bandwidth nor
// load-latency bound; what is left is the ACCUMULATOR dependency chain (1..4 dependent
// `mma_rc_t` into one `u[8]`) plus the 1/2/3/4 work imbalance across the 8 warps of a group.
//   bit0 (1)  UDENSE : run all NBT k-steps, not just s<=bj.  gdr_prologue zero-fills `Tv` before
//                      it writes the blocked inverse, so T's upper tiles are EXACT zeros ->
//                      arithmetically harmless, and every warp gets the same 4 steps.
//   bit1 (2)  2 split accumulators -> dependency chain halves (4 -> 2), +8 vregs.
//   bit2 (4)  4 split accumulators -> chain 1 (independent k-steps), +24 vregs; implies dense.
//   bit3 (8)  native `tsm.ld.mat.b32x4` (ldAm) for the U fragments.  v42 refuted this GLOBALLY
//             because its 84-cyc latency needs >=8 independent k-steps; densify+split is exactly
//             the change that creates independent steps, so it is re-tested here.
// ===== v121: SF = semantically-free FOOTPRINT knobs for `gdr_scan2` ===========================
// MANDATE (LEDGER "CORRECTION"): v112 broke the prologue's occupancy wall with two BIT-IDENTICAL
// levers -- `#pragma unroll 1` where full unrolling let the compiler hoist fragment loads above a
// dependency chain, and deleting a smem buffer that duplicated another.  Neither had ever been
// applied to `gdr_scan2` (168 regs x 512 thr + 173056 B => 1 block/SM).
//   SF bitmask (every variant is arithmetically identical; relerr must be exactly 0):
//     bit0 (1)   SBAL : `Sb` (entry-state snapshot, VB*LDS bf16 = 8704 B/group) ALIASED onto this
//                       group's UT/US region (2*VB*LDU = 9216 B).  Liveness re-derived -- v45
//                       aliased ZT/UT/US onto the SHARED Qb, which forced a block-wide 4th
//                       barrier; here BOTH buffers are per-group, so only a per-group barrier is
//                       needed.  Sb is written at the end of chunk c and read only by the P/T1
//                       GEMM of chunk c+1; UT/US are written by the U solve of c+1, already
//                       separated from P/T1 by the post-Z named barrier.  The reverse hazard
//                       (state+out reads UT/US, then the snapshot overwrites them) costs ONE extra
//                       per-group barrier.  smem 173056 -> 155648.
//     bit1 (2)   PTU1 : `#pragma unroll 1` on the P/T1 GEMM's 8 k-steps.
//     bit2 (4)   SOU1 : `#pragma unroll 1` on the state-update+output s loop (4 steps).
//     bit3 (8)   UU1  : `#pragma unroll 1` on the dense U solve's 4 k-steps.
//     bit4 (16)  U2   : the three U1 knobs use unroll factor 2 instead of 1.
//     bit5 (32)  SAU1 : `#pragma unroll 1` on the Sb snapshot vt loop.
//     bit6 (64)  ZU1  : `#pragma unroll 1` on the Z stage's p loop.
// ===== v131: SO = the state-update+output FRAGMENT-FEED knobs ==================================
// MEASURED MOTIVATION (`hgobjdump --dump-isa workspace/v121/build/gdr.cuda.o`, per-warp-per-chunk
// instruction census of the three MMA stages of `gdr_scan2<32,2,12,0>`):
//   P/T1      : 24 `tsm.ld.mat.b32x4`                          / 16 v.mma = 1.50 -> 0.96 us/MMA
//   U solve   :  8 `tsm.ld.mat.b32x4`                          /  4 v.mma = 2.00 -> 3.14
//   state+out : **64 `tsm.ld.b32`** + 4 `tsm.ld.mat.trans.b32x4` / 12 v.mma = **5.67** -> 2.84
// i.e. the 3x us/MMA gap between P/T1 and state+out is a 3.8x gap in FRAGMENT-LOAD INSTRUCTIONS,
// NOT a `row.row`-vs-`row.col` gap.  The `row.row` B operand (`ldBr`) is ALREADY a single native
// `tsm.ld.mat.trans.b32x4` in the object (v42's "raw asm emits zero tsm.ld.mat" counted only the
// non-`.trans` spelling), and ppu15lab `t:compute/ldmatrix_intrinsic` measures `.trans` as FREE
// ("same or lower cycles than non-transposed").  state+out is simply the ONE MMA stage whose
// A-order fragments never got v43's native loader: all 16 of them are still `ldA`'s 4-instruction
// scalar fan-out, because v42 bundled them with the sites where 84-cyc latency was exposed and
// v52's retry was numerically broken (it swapped the `.trans` feed too).
//   SO bitmask:
//     bit0 (1) SOLMS : `ldAm` for the 8 state-update A fragments (`US`).  -32 `tsm.ld.b32`.
//     bit1 (2) SOLMO : `ldAm` for the output fragments (`Rc`, `UT`).      -32 `tsm.ld.b32`.
//     bit2 (4) SODN  : DENSE output -- run all NBT s-steps of `R.U` instead of `if (s<=bi)`.
//                      EXACT, not approximate: `gdr_prologue`'s R epilogue explicitly writes 0 to
//                      `Am + i*LDA + j` for every `bj > bi` tile before emitting `rg`, so R's
//                      strict-upper 16x16 tiles are true zeros -- the same argument that made
//                      v51's dense U solve exact.  Balances the 1/2/3/4 output MMAs across the 8
//                      warps of a group and removes 4 predicated regions from the unrolled loop.
//     bit3 (8) SOKP  : **PROBE ONLY, NUMERICALLY INVALID** -- feeds the state update's B operand
//                      with the NON-transposing `tsm.ld.mat.b32x4` at the same address.  Identical
//                      instruction count and addressing to the `row.col` form a transposed `K`
//                      tile would give, so it prices the mandate's `row.row -> row.col` conversion
//                      WITHOUT paying for the transpose.  Flat timing => the conversion has zero
//                      upside.  Never shipped.
// ===== v132: GS = the LOAD-SKELETON knobs =====================================================
// MEASURED MOTIVATION.  With v131's SO=7 the ladder is SKEL 391.6 (prologue 157 + load skeleton
// **235**) + P/T1 104.2 + Z 50.5 + U 110.0 + state+out 187.8, so the 0-MMA load skeleton is now the
// biggest single term.  Sizing it: one 2-group block stages per chunk
//   K 16384 B + Q 16384 + V 8192 + T 8192 + R 8192 + fac 1536 = **58880 B**
// and 58880 B / (235 us / 240 chunks) = 60 B/cyc/SM at 1.30 GHz, against ppu15lab `f:ALT/02`'s
// MEASURED cp.async ceiling of 2172 GB/s = 2172e9/39/1.30e9 = **42.8 B/cyc/SM**... i.e. the skeleton
// is *at or past* the AIU ceiling.  So the skeleton is BYTE-bound after all (v47's byte-trim failed
// because it added a per-thread PREDICATE to the issue loop, not because bytes were free), and the
// only thing that can shrink it is staging fewer bytes with the SAME instruction count.
//   GS bitmask:
//     bit0 (1) TRTRI : `T` = (I+A)^-1 is lower-UNItriangular and `R` = tril(...) is lower
//                      triangular, so **6 of their 16 16x16 tiles are EXACT zeros** -- the
//                      prologue's inverse writes only block-columns j<=b and its R epilogue has an
//                      explicit `else` that stores `0u` to every bj>bi tile.  The dense U solve
//                      (v51) and the dense output (v131 SODN) READ those tiles, so they must be
//                      present in TSM -- but they never change, so they are zeroed ONCE before the
//                      chunk loop and the per-chunk `cp.async` stages only the 320 (of 512) 16-B
//                      items that live in a lower-triangular tile.  T+R 16384 -> 10240 B, total
//                      58880 -> **52736 B (-10.4 %)**, with the item count per thread unchanged
//                      (320 <= NTHR, so still one issue per thread) -- no predicate is added to a
//                      loop, the loop's *extent* shrinks, which is what v47 could not do.
//     bit1 (2) AHG   : every per-thread staging address is chunk-invariant except for one runtime
//                      constant stride (`BT*HQ*KD` for k/q, `BT*HV*VD` for v, `HV*BT*BT` for T/R,
//                      `HV*NF*BT` for fac), so hoist the whole address computation out of the chunk
//                      loop and advance by that stride.  The ISA shows the staging address math
//                      scheduled INTO the P/T1 region (35 `v.madl.i32` + 18 `v.madw.i64.i32` +
//                      18+18 `v.add.co/ci.u32` per chunk), which is pure per-chunk overhead.
// ===== v141: WH = warp height.  RE-OPENING v125's `WH=2` (1024 threads / 32 warps per SM) ======
// PRECONDITION CHANGE.  v125 refuted the 1024-thread block for exactly ONE reason: "1024 threads cap
// registers at 131072/1024 = 128 exactly and the WH=1 program needs 168 (144 with UV=9, 128 only
// with UV=9 AND SOU1)", so every WH=2 build SPILLED inside the chunk loop and even the skeleton blew
// up 2.5x.  v133/v135's program needs **exactly 128 regs / 0 spill at WH=1**, so that wall is gone.
// The prize v123 measured on the load skeleton is **-21.6 us** (v45's Little's-law linearity -- 27
// B/cyc/SM at 8 warps, 54 at 16 -- extends to 32 warps/SM).  The counterweights v123 measured (P/T1
// loses the shared `Sb` fragment +5.4, state+out the shared `K` fragment +28.2) are avoided by NOT
// re-splitting the compute: warps 0-7 of each group run the byte-identical WH=1 program with every
// fragment-reuse pattern intact, and warps 8-15 contribute only their share of the `cp.async` issue
// loops plus (optionally) the U solve, which v123 measured as free to move (-0.5).
//   WH=1 : 512 threads, 2 groups x 8 warps   -- byte-identical to v135 (relerr must be exactly 0)
//   WH=2 : 1024 threads, 2 groups x 16 warps -- `bar.sync grp+1, 512`; grid UNCHANGED at 32 blocks.
//   WU   : which half owns the U solve.  0 = the state-owning half (gh==0), 1 = the helper half.
// ===== v142: MP = delete the `row.col` B-operand REGISTER PERMUTATION (see the note above `ldBc`)
//   bit0 (1) MPP : P/T1's `Sb` B fragment      (32 of the 62 in-loop `v.madl.i32` copies)
//   bit1 (2) MPU : the U solve's `T` B fragment (14)
//   bit2 (4) MPO : the output's `UT` B fragment (16)
// Each site swaps `ldAm`+`mma_rc_t` for `ldBm`+`mma_rc`, which is the SAME arithmetic on the SAME
// bytes -- relerr must come out EXACTLY 0, and any variant that does not is an operand-mapping bug,
// not a datapoint (v52).
// ===== v144: OQ = the OUTPUT-STORE knobs ======================================================
// MEASURED MOTIVATION (v143).  Feeding the Z stage from global cost **+145 us** purely because a
// wmma-accumulator lane map (`rr = lane>>2`, `cc = (lane&3)*2`) makes ONE warp instruction touch 8
// token rows 4096 B apart -- 8 sectors for 128 B of payload.  The OUTPUT STORE has the SAME lane
// map and had never been examined: per warp per chunk it issues 4 stores of 4 B/lane, each covering
// 8 rows x 16 B = 8 sectors of 32 B, i.e. **256 B of sectors for 128 B of payload -- 2x write
// amplification on `output`** (63 MB -> 126 MB on sid0).
//   bit0 (1) OCO : make each row's 32 B contiguous inside ONE store.  Lanes 4rr..4rr+3 hold the
//                  pairs (cc,cc+1) in w0 and (cc+8,cc+9) in w1 for cc = 0,2,4,6; two
//                  `__shfl_xor_sync(...,1)` exchanges re-pack them so lane j of the quad owns 4
//                  CONSECUTIVE columns (j=0 -> 0..3, j=2 -> 4..7, j=1 -> 8..11, j=3 -> 12..15).
//                  Each lane then does one 8-B store and the quad covers the row's full 32 B:
//                  **4 store instructions -> 2, 8 half-used sectors -> 8 fully used sectors.**
//                  Arithmetically identical -- relerr must be EXACTLY 0.
//   bit1 (2) NOST: PROBE ONLY, NUMERICALLY INVALID -- drop the output store, to price it.
template <int VB, int VAR = 2, int UV = 0, int SF = 0, int SO = 0, int GS = 0, int WH = 1,
          int WU = 0, int MP = 0, int OQ = 0>
__global__ __launch_bounds__(NW2 * 32 * WH) void gdr_scan2(
    const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v, const __nv_bfloat16 *__restrict__ tg,
    const __nv_bfloat16 *__restrict__ rg,
    const float *__restrict__ fg, const void *__restrict__ s0, const int *__restrict__ cu,
    __nv_bfloat16 *__restrict__ out, float *__restrict__ sf, int HQ, int HV, int rep,
    float scale, int mask, int sd) {
  constexpr int LDK = KD + 8;
  constexpr int LDS = KD + 8;
  constexpr int LDA = BT + 8;
  constexpr int LDU = BT + 8;
  constexpr int VG = 2;              // groups per block
  constexpr int VW = VB * VG;        // V rows owned by the block
  constexpr int LDV = VW + 8;
  constexpr int NVT = VB / 16;
  constexpr int NBT = BT / 16;
  constexpr int VH = (NVT >= 2) ? (NVT / 2) : 1;
  constexpr int NTHR = NW2 * 32 * WH;
  constexpr bool LDMP = (VAR & 2) != 0;
  // v121 footprint knobs (see the SF comment above).  All bit-identical.
  constexpr bool SBAL = (SF & 1) != 0;
  constexpr int UFC = (SF & 16) ? 2 : 1;
  constexpr int PTUF = (SF & 2) ? UFC : (KD / 16);
  constexpr int SOUF = (SF & 4) ? UFC : NBT;
  constexpr int UUF = (SF & 8) ? UFC : NBT;
  constexpr int SAUF = (SF & 32) ? 1 : NVT;
  constexpr int ZUF = (SF & 64) ? 1 : 4;
  // v131 state+out fragment-feed knobs (see the SO comment above).
  constexpr bool SOLMS = (SO & 1) != 0;
  constexpr bool SOLMO = (SO & 2) != 0;
  constexpr bool SODN = (SO & 4) != 0;
  constexpr bool SOKP = (SO & 8) != 0;
  // v132 load-skeleton knobs (see the GS comment above).  Both bit-identical.
  constexpr bool TRTRI = (GS & 1) != 0;
  constexpr bool AHG = (GS & 2) != 0;
  // v142 fragment-order knobs (see the MP comment above the template header).
  constexpr bool MPP = (MP & 1) != 0;
  constexpr bool MPU = (MP & 2) != 0;
  constexpr bool MPO = (MP & 4) != 0;
  // v142 bit3 MPI: hoist the two PURE-REGISTER rescalings (`S *= dlast`, `accT1 *= dd`) out of the
  // state+out block and issue them just BEFORE the U solve.  MOTIVATION: the U solve is the only
  // stage bracketed by two named barriers with a strictly serial body -- 4 dependent
  // (ld.mat, ld.mat, s.wait, mma) steps, ~600 cyc/chunk for 4 MMAs -- so every warp of the group
  // stalls on the same TSM latency at the same instant with nothing else to issue.  This is NOT
  // v46/v134's refuted operand hoisting: it moves ZERO fragment loads and holds ZERO extra
  // fragments (the 24 values are already live in registers); it only relocates independent FALU
  // work into the stall shadow.  Arithmetically identical -- both rescalings depend only on `facc`
  // and on values finalised before the post-Z barrier.
  constexpr bool MPI = (MP & 8) != 0;
  // v144 output-store knobs (see the OQ comment above the template header).
  constexpr bool OCO = (OQ & 1) != 0;
  constexpr bool NOST = (OQ & 2) != 0;
  // v133 bit2 QDIR: generalise v31's `Vb` deletion to the Q tile -- do not stage Q at all and read
  // the `T1 = Q.S^T` A-fragments straight from global (L2) with `ld.global.nc.b32`.  ARITHMETIC
  // PREDICTION, recorded before measuring: Q staging is only 2 `cp.async` warp-instructions per warp
  // per chunk (1024 16-B items / 512 threads), while the direct feed costs 8 kk x 4 = 32
  // `ld.global.nc.b32` warp-instructions -- 16x MORE instructions -- and re-reads Q 4x from L2
  // (warps gw and gw+4 share bi, and the two groups duplicate every tile).  Since v132 measured the
  // skeleton to be INSTRUCTION-bound and not byte-bound (TRTRI cut 10.4 % of the staged bytes for
  // -0.5 %, while AHG cut only address arithmetic for -1.8 %), this is expected to LOSE.
  constexpr bool QDIR = (GS & 4) != 0;
  // ===== v143 bit3 (8) VDIR: delete the `V` cp.async staging and feed the Z stage from GLOBAL ====
  // WHY THIS IS NOT `QDIR` (refuted +872 us on P/T1 in v133).  v133's own post-mortem names the
  // structural reason `QDIR` failed and, in the same sentence, predicts that `V` is the one tensor
  // where it works: "`V` is consumed as **4 scalar `lds32` per warp per chunk** (the Z stage), so a
  // global feed is instruction-neutral there; `Q`/`K` are consumed as 8 wmma FRAGMENTS per warp per
  // chunk = 32 lane-words.  A wmma fragment is the wrong granularity for a global load on this
  // chip."  That prediction has never been built.  The accounting for `V`:
  //   * staging cost deleted: 8192 B of the 58880 B a block stages per chunk (-13.9 %) AND one
  //     `cp.async` warp-instruction per warp per chunk of the ~7 the block issues (-14 %).  Unlike
  //     v132's `TRTRI` -- which cut 10.4 % of the BYTES but left the instruction count unchanged
  //     (320 <= NTHR, so still one predicated issue per thread) and bought only -0.5 % -- this cuts
  //     BOTH, which is the combination v132's "the staging lever is instruction count" implies.
  //   * feed cost added: exactly 4 `ld.global.nc.b32` per warp per chunk, replacing exactly 4
  //     `tsm.ld.b32`.  Instruction-neutral, as v133 predicted.
  //   * NO replication: each block owns its own 64 V columns and each chunk its own 64 rows, so the
  //     global read is 1x unique (unlike Q/K, which are 8x replicated across the 2 V-blocks and 4
  //     GQA value heads of a key head).
  // EXACTNESS.  Rows i >= L used to read the explicit zero-fill `GDR2_STAGE_A` writes.  With VDIR
  // the row index is CLAMPED to `t_end-1` instead (an in-bounds, finite bf16), and the result is
  // still bit-identical because the prologue writes `fac[4*BT + lane] = (lane < L) ? beta : 0` --
  // so `z = fbt[i] * (v - dd*P) = 0 * finite = 0` for every i >= L, exactly as before.  Verified:
  // relerr must be EXACTLY 0.
  constexpr bool VDIR = (GS & 8) != 0;
  // ===== v143 bit4 (16) VPF: VDIR + a ONE-CHUNK-AHEAD REGISTER prefetch of the 4 V words ========
  // Plain VDIR (measured this round) wins **-27.9 us on the load skeleton** -- the largest skeleton
  // saving the campaign has measured, and evidence that the staging path IS byte-proportional once
  // the instruction count falls with the bytes -- but it costs **+145 us on the Z stage**, because
  // the four `ld.global.nc.b32` a warp issues sit at wmma-ACCUMULATOR lane granularity
  // (rr = lane>>2, cc = (lane&3)*2), so ONE warp instruction touches 8 different token rows 4096 B
  // apart -- 8 sectors for 128 B of payload -- with ~500 cyc of L2/HBM latency fully exposed in a
  // stage whose entire dependency chain is load -> fma -> cvt -> store.
  // VPF keeps the staging saving and removes the latency: the four words are only 4 REGISTERS, so
  // chunk c+1's are issued at the top of chunk c, right beside the `cp.async` issue for c+1, and
  // consumed a whole chunk (2.7 us) later.  This is NOT v62's refuted deeper `cp.async` prefetch
  // nor v46/v134's refuted fragment hoisting: it holds 4 SCALAR registers, not a wmma fragment,
  // and it REPLACES a TSM round trip instead of adding one.
  constexpr bool VPF = (GS & 16) != 0;
  constexpr bool VG_ = VDIR || VPF;   // read V from global rather than from the staged tile
  static_assert(!SBAL || VB * LDS <= 2 * VB * LDU, "Sb does not fit in the UT/US region");
  // NOTE (v46, measured): hoisting these stages' TSM operands out of their dependent MMA chains
  // (all (bj+1) pairs of the U solve, or one s-step of software pipelining in state+out) makes them
  // 21 % and 26 % SLOWER -- they are REGISTER-PRESSURE bound, not load-latency bound, at 168 regs x
  // 512 threads.  Both variants were removed after measurement; do not re-add them.

  extern __shared__ char smem_raw[];
  char *sp = smem_raw;
  __nv_bfloat16 *Kb = (__nv_bfloat16 *)sp;  sp += BT * LDK * 2 * 2;      // shared, 2 stages
  __nv_bfloat16 *Qb = (__nv_bfloat16 *)sp;  sp += BT * LDK * 2 * 2;      // shared, 2 stages
  __nv_bfloat16 *Vb = (__nv_bfloat16 *)sp;  sp += BT * LDV * 2 * 2;      // shared, 2 stages
  __nv_bfloat16 *Tm = (__nv_bfloat16 *)sp;  sp += BT * LDA * 2 * 2;      // shared, 2 stages
  __nv_bfloat16 *Rm = (__nv_bfloat16 *)sp;  sp += BT * LDA * 2 * 2;      // shared, 2 stages
  float *fac = (float *)sp;                 sp += NF * BT * 4 * 2;       // shared, 2 stages
  __nv_bfloat16 *Sba = (__nv_bfloat16 *)sp;
  if (!SBAL) sp += VG * VB * LDS * 2;                                    // per group
  __nv_bfloat16 *ZUa = (__nv_bfloat16 *)sp; sp += VG * 3 * VB * LDU * 2; // per group

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int rr = lane >> 2, cc = (lane & 3) << 1;
  // v141 WH: a group is 8*WH warps.  `gh` splits it into the state-owning half (0) and the helper
  // half (1); at WH=1 there is no helper half and `gh` folds to the constant 0, so every expression
  // below collapses to v135's code exactly.
  const int grp = warp / (8 * WH);    // 0 or 1
  const int gh = (WH == 1) ? 0 : ((warp >> 3) & 1);
  const int gw = warp & 7;           // warp index inside the half
  const bool own = (gh == 0);        // owns the recurrent state S and every non-U stage
  const bool uown = (WH == 1) || (WU ? (gh == 1) : (gh == 0));
  const int bi = gw & (NBT - 1);
  const int vh = gw >> 2;
  const int bv0 = vh * VH;
  const int vloc = grp * VB;                 // this group's column base inside Vb
  const int v0 = blockIdx.x * VW + vloc;     // global V row base
  const int h = blockIdx.y;
  const int rq = blockIdx.z;
  const int hq = h / rep;
  const int BAR = grp + 1;
  constexpr int BARN = 256 * WH;             // threads the per-group named barrier drains

  __nv_bfloat16 *ZT = ZUa + grp * (3 * VB * LDU);
  __nv_bfloat16 *UT = ZT + VB * LDU;
  __nv_bfloat16 *US = UT + VB * LDU;
  // v121 SBAL: the entry-state snapshot lives in the UT/US bytes, which are dead from the end of
  // state+out until the U solve of the NEXT chunk (see the SF comment).
  __nv_bfloat16 *Sb = SBAL ? UT : (Sba + grp * (VB * LDS));

  const int t_start = cu[rq];
  const int t_end = cu[rq + 1];
  const int nchunk = (t_end - t_start + BT - 1) / BT;
  int coff = 0;
  for (int r = 0; r < rq; ++r) coff += (cu[r + 1] - cu[r] + BT - 1) / BT;

  float S[NVT][8];
  if (own) {
    const size_t sbase = (((size_t)rq * HV + h) * VD + v0) * KD + gw * 16;
#pragma unroll
    for (int vt = 0; vt < NVT; ++vt)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        int vrow = vt * 16 + rr + half * 8;
        float2 a = ldg_s2v(s0, sbase + (size_t)vrow * KD + cc, sd);
        float2 b = ldg_s2v(s0, sbase + (size_t)vrow * KD + cc + 8, sd);
        S[vt][0 + half * 2] = a.x;
        S[vt][1 + half * 2] = a.y;
        S[vt][4 + half * 2] = b.x;
        S[vt][5 + half * 2] = b.y;
      }
  }

  constexpr int NKQ = (BT * 16 + NTHR - 1) / NTHR;
  constexpr int NVI = (BT * (VW / 8) + NTHR - 1) / NTHR;
  constexpr int NTI = (BT * (BT / 8) + NTHR - 1) / NTHR;
  const uint4 Z4 = make_uint4(0, 0, 0, 0);
  const int vg0 = blockIdx.x * VW;   // the block's V column base in the v tensor

  // ---- v132 TRTRI: this thread's item inside the 320-item lower-triangular enumeration of a
  // 64x64 tile walked in 16-B (8-element) units.  Row i (tile row br = i>>4) owns 2*(br+1) items,
  // so the cumulative counts are 32 / 96 / 192 / 320 and the decode is exact.  Computed ONCE.
  int tr_i = 0, tr_c8 = 0;
  bool tr_act = false;
  if (TRTRI) {
    const int n = tid;
    tr_act = (n < 320);
    if (n < 32) { tr_i = n >> 1; tr_c8 = (n & 1) << 3; }
    else if (n < 96) { const int m = n - 32; tr_i = 16 + (m >> 2); tr_c8 = (m & 3) << 3; }
    else if (n < 192) { const int m = n - 96; tr_i = 32 + m / 6; tr_c8 = (m % 6) << 3; }
    else if (n < 320) { const int m = n - 192; tr_i = 48 + (m >> 3); tr_c8 = (m & 7) << 3; }
  }

  // ---- v132 AHG: chunk-invariant staging addresses + the one runtime stride each tensor advances
  // by per chunk.  All of this is dead code when AHG is off.
  const size_t ah_kq = (size_t)BT * HQ * KD;
  const size_t ah_v = (size_t)BT * HV * VD;
  const size_t ah_tr = (size_t)HV * BT * BT;
  const size_t ah_fc = (size_t)HV * NF * BT;
  const __nv_bfloat16 *ah_gk[NKQ], *ah_gq[NKQ];
  int ah_sk[NKQ], ah_kr[NKQ];
#pragma unroll
  for (int u = 0; u < NKQ; ++u) {
    const int idx = tid + u * NTHR;
    const int i = (idx < BT * 16) ? (idx >> 4) : 0;
    const int c8 = (idx < BT * 16) ? ((idx & 15) << 3) : 0;
    ah_kr[u] = (idx < BT * 16) ? i : BT;      // >= BT is never < L, and the else-branch is guarded
    ah_sk[u] = i * LDK + c8;
    ah_gk[u] = k + ((size_t)(t_start + i) * HQ + hq) * KD + c8;
    ah_gq[u] = q + ((size_t)(t_start + i) * HQ + hq) * KD + c8;
  }
  const __nv_bfloat16 *ah_gv[NVI];
  int ah_sv[NVI], ah_vr[NVI];
#pragma unroll
  for (int u = 0; u < NVI; ++u) {
    const int idx = tid + u * NTHR;
    const int i = (idx < BT * (VW / 8)) ? (idx / (VW / 8)) : 0;
    const int c8 = (idx < BT * (VW / 8)) ? ((idx % (VW / 8)) << 3) : 0;
    ah_vr[u] = (idx < BT * (VW / 8)) ? i : BT;
    ah_sv[u] = i * LDV + c8;
    ah_gv[u] = v + ((size_t)(t_start + i) * HV + h) * VD + vg0 + c8;
  }
  const __nv_bfloat16 *ah_gt[NTI], *ah_gr[NTI];
  int ah_st[NTI];
  bool ah_ta[NTI];
#pragma unroll
  for (int u = 0; u < NTI; ++u) {
    int i, c8;
    if (TRTRI) {
      i = tr_i; c8 = tr_c8; ah_ta[u] = (u == 0) && tr_act;
    } else {
      const int idx = tid + u * NTHR;
      ah_ta[u] = idx < BT * (BT / 8);
      i = ah_ta[u] ? (idx / (BT / 8)) : 0;
      c8 = ah_ta[u] ? ((idx % (BT / 8)) << 3) : 0;
    }
    ah_st[u] = i * LDA + c8;
    ah_gt[u] = tg + (size_t)(coff * HV + h) * (BT * BT) + i * BT + c8;
    ah_gr[u] = rg + (size_t)(coff * HV + h) * (BT * BT) + i * BT + c8;
  }
  const float *ah_gf = fg + (size_t)(coff * HV + h) * (NF * BT) + tid * 4;

  // K, Q, V of chunk CC into stage BUF.  K and Q stay INTERLEAVED in one loop: v43 measured that
  // splitting them into two loops costs +51 us on the load path.
#define GDR2_STAGE_A(BUF, CC)                                                                \
  {                                                                                          \
    const int _c = (CC);                                                                     \
    const int _t0 = t_start + _c * BT;                                                       \
    const int _L = min(BT, t_end - _t0);                                                     \
    __nv_bfloat16 *_kb = Kb + (BUF) * (BT * LDK);                                            \
    __nv_bfloat16 *_qb = Qb + (BUF) * (BT * LDK);                                            \
    __nv_bfloat16 *_vb = Vb + (BUF) * (BT * LDV);                                            \
    const size_t _okq = (size_t)_c * ah_kq;                                                  \
    const size_t _ov = (size_t)_c * ah_v;                                                    \
    _Pragma("unroll") for (int u = 0; u < NKQ; ++u) {                                        \
      if (AHG) {                                                                             \
        if (ah_kr[u] < _L) {                                                                 \
          __pipeline_memcpy_async(_kb + ah_sk[u], ah_gk[u] + _okq, 16);                       \
          if (!QDIR) __pipeline_memcpy_async(_qb + ah_sk[u], ah_gq[u] + _okq, 16);             \
        } else if (ah_kr[u] < BT) {                                                          \
          *(uint4 *)(_kb + ah_sk[u]) = Z4;                                                   \
          if (!QDIR) *(uint4 *)(_qb + ah_sk[u]) = Z4;                                         \
        }                                                                                    \
      } else {                                                                               \
      int idx = tid + u * NTHR, i = idx >> 4, c8 = (idx & 15) << 3;                          \
      if (idx < BT * 16) {                                                                   \
        if (i < _L) {                                                                        \
          __pipeline_memcpy_async(_kb + i * LDK + c8,                                        \
                                  k + ((size_t)(_t0 + i) * HQ + hq) * KD + c8, 16);          \
          __pipeline_memcpy_async(_qb + i * LDK + c8,                                        \
                                  q + ((size_t)(_t0 + i) * HQ + hq) * KD + c8, 16);          \
        } else {                                                                             \
          *(uint4 *)(_kb + i * LDK + c8) = Z4;                                               \
          *(uint4 *)(_qb + i * LDK + c8) = Z4;                                               \
        }                                                                                    \
      } }                                                                                    \
    }                                                                                        \
    _Pragma("unroll") for (int u = 0; u < NVI; ++u) {                                        \
      if (VG_) {                                                                             \
      } else if (AHG) {                                                                      \
        if (ah_vr[u] < _L) __pipeline_memcpy_async(_vb + ah_sv[u], ah_gv[u] + _ov, 16);        \
        else if (ah_vr[u] < BT) *(uint4 *)(_vb + ah_sv[u]) = Z4;                              \
      } else {                                                                               \
      int idx = tid + u * NTHR;                                                              \
      if (idx < BT * (VW / 8)) {                                                             \
        int i = idx / (VW / 8), c8 = (idx % (VW / 8)) << 3;                                  \
        if (i < _L)                                                                          \
          __pipeline_memcpy_async(_vb + i * LDV + c8,                                        \
                                  v + ((size_t)(_t0 + i) * HV + h) * VD + vg0 + c8, 16);     \
        else                                                                                 \
          *(uint4 *)(_vb + i * LDV + c8) = Z4;                                               \
      } }                                                                                    \
    }                                                                                        \
  }

#define GDR2_STAGE_B(BUF, CC)                                                                \
  {                                                                                          \
    const int _cid = coff + (CC);                                                            \
    __nv_bfloat16 *_tm = Tm + (BUF) * (BT * LDA);                                            \
    __nv_bfloat16 *_rm = Rm + (BUF) * (BT * LDA);                                            \
    float *_fc = fac + (BUF) * (NF * BT);                                                    \
    const __nv_bfloat16 *_ts = tg + (size_t)(_cid * HV + h) * (BT * BT);                     \
    const __nv_bfloat16 *_rs = rg + (size_t)(_cid * HV + h) * (BT * BT);                     \
    const size_t _otr = (size_t)(CC) * ah_tr;                                                \
    _Pragma("unroll") for (int u = 0; u < NTI; ++u) {                                        \
      if (AHG) {                                                                             \
        if (ah_ta[u]) {                                                                      \
          __pipeline_memcpy_async(_tm + ah_st[u], ah_gt[u] + _otr, 16);                       \
          __pipeline_memcpy_async(_rm + ah_st[u], ah_gr[u] + _otr, 16);                       \
        }                                                                                    \
      } else if (TRTRI) {                                                                    \
        if (u == 0 && tr_act) {                                                              \
          __pipeline_memcpy_async(_tm + tr_i * LDA + tr_c8, _ts + tr_i * BT + tr_c8, 16);     \
          __pipeline_memcpy_async(_rm + tr_i * LDA + tr_c8, _rs + tr_i * BT + tr_c8, 16);     \
        }                                                                                    \
      } else {                                                                               \
        int idx = tid + u * NTHR;                                                            \
        if (idx < BT * (BT / 8)) {                                                           \
          int i = idx / (BT / 8), c8 = (idx % (BT / 8)) << 3;                                \
          __pipeline_memcpy_async(_tm + i * LDA + c8, _ts + i * BT + c8, 16);                \
          __pipeline_memcpy_async(_rm + i * LDA + c8, _rs + i * BT + c8, 16);                \
        }                                                                                    \
      }                                                                                      \
    }                                                                                        \
    if (tid < NF * BT / 4) {                                                                 \
      if (AHG) __pipeline_memcpy_async(_fc + tid * 4, ah_gf + (size_t)(CC) * ah_fc, 16);      \
      else                                                                                   \
        __pipeline_memcpy_async(_fc + tid * 4,                                               \
                                fg + (size_t)(_cid * HV + h) * (NF * BT) + tid * 4, 16);     \
    }                                                                                        \
  }

  // v132 TRTRI: the 6 strict-upper 16x16 tiles of T and R are EXACT zeros for every chunk, and the
  // dense U solve / dense output read them, so zero both double-buffered tiles ONCE here and never
  // restage them.  Tm and Rm are adjacent in the carve-up, 2 stages each => 4*BT*LDA elements.
  if (TRTRI) {
    for (int idx = tid; idx < 4 * BT * LDA / 8; idx += NTHR) *(uint4 *)(Tm + idx * 8) = Z4;
    __syncthreads();
  }
  GDR2_STAGE_A(0, 0)
  GDR2_STAGE_B(0, 0)
  __pipeline_commit();
  if (own) {
#pragma unroll
    for (int vt = 0; vt < NVT; ++vt)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        __nv_bfloat16 *d = Sb + (vt * 16 + rr + half * 8) * LDS + gw * 16 + cc;
        *(uint32_t *)d = cvt_bf16x2(S[vt][0 + half * 2], S[vt][1 + half * 2]);
        *(uint32_t *)(d + 8) = cvt_bf16x2(S[vt][4 + half * 2], S[vt][5 + half * 2]);
      }
  }

  // v143 VPF: the four V words this warp consumes in the Z stage, fetched one chunk ahead.  Row
  // indices are clamped to `t_end-1` -- rows >= L are multiplied by `fac[4*BT+i] == 0`, so the
  // value is irrelevant but the address must stay in bounds; bit-identical to the zero-fill.
  uint32_t vcur[VH][4], vnxt[VH][4];
#define GDR2_VLOAD(DST, CC)                                                                  \
  if (VPF) {                                                                                 \
    const int _t0 = t_start + (CC) * BT;                                                     \
    _Pragma("unroll") for (int t = 0; t < VH; ++t) {                                         \
      const int _vc = (bv0 + t) * 16 + cc;                                                   \
      const int _i0 = bi * 16 + rr;                                                          \
      const __nv_bfloat16 *_p0 =                                                             \
          v + ((size_t)min(_t0 + _i0, t_end - 1) * HV + h) * VD + v0 + _vc;                  \
      const __nv_bfloat16 *_p1 =                                                             \
          v + ((size_t)min(_t0 + _i0 + 8, t_end - 1) * HV + h) * VD + v0 + _vc;              \
      DST[t][0] = ldg32u(_p0);                                                               \
      DST[t][1] = ldg32u(_p1);                                                               \
      DST[t][2] = ldg32u(_p0 + 8);                                                           \
      DST[t][3] = ldg32u(_p1 + 8);                                                           \
    }                                                                                        \
  }
  GDR2_VLOAD(vcur, 0)

#pragma unroll 1
  for (int c = 0; c < nchunk; ++c) {
    const int t0 = t_start + c * BT;
    const int L = min(BT, t0 < t_end ? t_end - t0 : 0);
    const int cur = c & 1;
    const __nv_bfloat16 *Kc = Kb + cur * (BT * LDK);
    const __nv_bfloat16 *Qc = Qb + cur * (BT * LDK);
    const __nv_bfloat16 *Vc = Vb + cur * (BT * LDV);
    const __nv_bfloat16 *Tc = Tm + cur * (BT * LDA);
    const __nv_bfloat16 *Rc = Rm + cur * (BT * LDA);
    const float *facc = fac + cur * (NF * BT);

    __pipeline_wait_prior(0);
    __syncthreads();   // the only block-wide barrier: stage landed + Sb published

    if (c + 1 < nchunk) {
      GDR2_STAGE_A(cur ^ 1, c + 1)
      GDR2_STAGE_B(cur ^ 1, c + 1)
      GDR2_VLOAD(vnxt, c + 1)
    }
    __pipeline_commit();

    // v133 QDIR: this lane's two global Q rows for the whole chunk.  Rows past the sequence end are
    // CLAMPED (not predicated) -- `accT1` rows >= L are never stored (the epilogue guards `i < L`),
    // so the value is irrelevant, but the address must stay inside the tensor.
    const __nv_bfloat16 *qg0 = q, *qg1 = q;
    if (QDIR) {
      const int _r0 = min(t0 + bi * 16 + rr, t_end - 1);
      const int _r1 = min(t0 + bi * 16 + rr + 8, t_end - 1);
      qg0 = q + ((size_t)_r0 * HQ + hq) * KD + cc;
      qg1 = q + ((size_t)_r1 * HQ + hq) * KD + cc;
    }
    // ---- P = K S0^T and T1 = Q S0^T share the Sb B-fragment ----
    float accP[VH][8], accT1[VH][8];
#pragma unroll
    for (int t = 0; t < VH; ++t)
#pragma unroll
      for (int e = 0; e < 8; ++e) { accP[t][e] = 0.f; accT1[t][e] = 0.f; }
    if ((mask & 1) && own) {
#pragma unroll PTUF
      for (int kk = 0; kk < KD / 16; ++kk) {
        uint32_t rak[4], raq[4];
        if (LDMP) ldAm(rak, Kc + bi * 16 * LDK + kk * 16, LDK, lane);
        else ldA(rak, Kc + bi * 16 * LDK + kk * 16, LDK, lane);
        if (QDIR) {
          // same lane map as `ldA`: r0/r2 from row rr, r1/r3 from row rr+8, columns cc and cc+8.
          raq[0] = ldg32u(qg0 + kk * 16);
          raq[2] = ldg32u(qg0 + kk * 16 + 8);
          raq[1] = ldg32u(qg1 + kk * 16);
          raq[3] = ldg32u(qg1 + kk * 16 + 8);
        } else if (LDMP) {
          ldAm(raq, Qc + bi * 16 * LDK + kk * 16, LDK, lane);
        } else {
          ldA(raq, Qc + bi * 16 * LDK + kk * 16, LDK, lane);
        }
#pragma unroll
        for (int t = 0; t < VH; ++t) {
          if (bv0 + t < NVT) {
            uint32_t rb[4];
            if (MPP) {
              // v142: load `Sb` already in MMA B-operand order -> no register permutation.
              if (LDMP) ldBm(rb, Sb + (bv0 + t) * 16 * LDS + kk * 16, LDS, lane);
              else ldBc(rb, Sb + (bv0 + t) * 16 * LDS + kk * 16, LDS, lane);
              mma_rc(accP[t], rak, rb);
              mma_rc(accT1[t], raq, rb);
            } else {
              if (LDMP) ldAm(rb, Sb + (bv0 + t) * 16 * LDS + kk * 16, LDS, lane);
              else ldA(rb, Sb + (bv0 + t) * 16 * LDS + kk * 16, LDS, lane);
              mma_rc_t(accP[t], rak, rb);
              mma_rc_t(accT1[t], raq, rb);
            }
          }
        }
      }
    }
    const float *fdd = facc + 0 * BT, *fex = facc + 1 * BT, *fbt = facc + 4 * BT;
    const float dlast = facc[5 * BT];

    // ---- Z^T = [ beta_i (v_i - dd_i P_i) ]^T ----
    if ((mask & 4) && own) {
#pragma unroll
      for (int t = 0; t < VH; ++t)
        if (bv0 + t < NVT) {
#pragma unroll ZUF
          for (int p = 0; p < 4; ++p) {
            const int e = p * 2;
            int i = bi * 16 + rr + ACC_ROW(e);
            int vv = (bv0 + t) * 16 + cc + ACC_COL(e);
            uint32_t vpair;
            if (VPF) {
              vpair = vcur[t][p];          // prefetched one chunk ago (v143 VPF)
            } else if (VDIR) {
              // v143: straight from global.  Row clamped (rows >= L carry beta == 0, so the value
              // is multiplied by zero -- bit-identical to the staged zero-fill, and in bounds).
              const int _rt = min(t0 + i, t_end - 1);
              vpair = ldg32u(v + ((size_t)_rt * HV + h) * VD + v0 + vv);
            } else {
              vpair = lds32(Vc + i * LDV + vloc + vv);
            }
            float b = fbt[i], d = fdd[i];
            float z0 = b * (__bfloat162float(__ushort_as_bfloat16((unsigned short)(vpair & 0xffffu)))
                            - d * accP[t][e]);
            float z1 = b * (__bfloat162float(__ushort_as_bfloat16((unsigned short)(vpair >> 16)))
                            - d * accP[t][e + 1]);
            ZT[vv * LDU + i] = __float2bfloat16_rn(z0);
            ZT[(vv + 1) * LDU + i] = __float2bfloat16_rn(z1);
          }
        }
    }
    asm volatile("bar.sync %0, %1;" ::"r"(BAR), "n"(BARN) : "memory");

    // v142 MPI: the two rescalings, moved up into the U solve's stall shadow (see the MPI note).
    float hd0 = 0.f, hd1 = 0.f;
    if (MPI && (mask & 96) && own) {
#pragma unroll
      for (int vt = 0; vt < NVT; ++vt)
#pragma unroll
        for (int e = 0; e < 8; ++e) S[vt][e] *= dlast;
      hd0 = fdd[bi * 16 + rr];
      hd1 = fdd[bi * 16 + rr + 8];
#pragma unroll
      for (int t = 0; t < VH; ++t)
#pragma unroll
        for (int e = 0; e < 8; ++e) accT1[t][e] *= (e & 2) ? hd1 : hd0;
    }

    // ---- U^T = Z^T (I+A)^-T : one MMA chain per (v-tile, row-tile) ----
    if ((mask & 16) && uown) {
      constexpr int UNA = (UV & 4) ? 4 : ((UV & 2) ? 2 : 1);
      constexpr bool UDENSE = (UV & 5) != 0;
      constexpr bool ULDM = (UV & 8) != 0;
      for (int ti = gw; ti < NVT * NBT; ti += 8) {
        const int bv = ti >> 2, bj = ti & (NBT - 1);
        float uu[UNA][8];
#pragma unroll
        for (int a = 0; a < UNA; ++a)
#pragma unroll
          for (int e = 0; e < 8; ++e) uu[a][e] = 0.f;
#pragma unroll UUF
        for (int s = 0; s < NBT; ++s)
          if (UDENSE || s <= bj) {
            uint32_t ra[4], rb[4];
            if (ULDM) ldAm(ra, ZT + bv * 16 * LDU + s * 16, LDU, lane);
            else ldA(ra, ZT + bv * 16 * LDU + s * 16, LDU, lane);
            if (MPU) {
              // v142: `T` already in MMA B-operand order -> no register permutation.
              if (ULDM) ldBm(rb, Tc + bj * 16 * LDA + s * 16, LDA, lane);
              else ldBc(rb, Tc + bj * 16 * LDA + s * 16, LDA, lane);
              mma_rc(uu[s % UNA], ra, rb);
            } else {
              if (ULDM) ldAm(rb, Tc + bj * 16 * LDA + s * 16, LDA, lane);
              else ldA(rb, Tc + bj * 16 * LDA + s * 16, LDA, lane);
              mma_rc_t(uu[s % UNA], ra, rb);
            }
          }
        float u[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          float acc = uu[0][e];
#pragma unroll
          for (int a = 1; a < UNA; ++a) acc += uu[a][e];
          u[e] = acc;
        }
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          const int e = p * 2;
          int vv = bv * 16 + rr + ACC_ROW(e);
          int j = bj * 16 + cc + ACC_COL(e);
          *(uint32_t *)(UT + vv * LDU + j) = cvt_bf16x2(u[e], u[e + 1]);
          *(uint32_t *)(US + vv * LDU + j) = cvt_bf16x2(u[e] * fex[j], u[e + 1] * fex[j + 1]);
        }
      }
    }
    asm volatile("bar.sync %0, %1;" ::"r"(BAR), "n"(BARN) : "memory");

    // ---- interleaved state update + output for ILP ----
    if ((mask & 96) && own) {
      if (!MPI) {
#pragma unroll
        for (int vt = 0; vt < NVT; ++vt)
#pragma unroll
          for (int e = 0; e < 8; ++e) S[vt][e] *= dlast;
        float d0 = fdd[bi * 16 + rr], d1 = fdd[bi * 16 + rr + 8];
#pragma unroll
        for (int t = 0; t < VH; ++t)
#pragma unroll
          for (int e = 0; e < 8; ++e) accT1[t][e] *= (e & 2) ? d1 : d0;
      }
#pragma unroll SOUF
      for (int s = 0; s < NBT; ++s) {
        uint32_t rb_k[4];
        // v131 SOKP is a PROBE: same instruction, same address, non-transposing -- it prices the
        // `row.col` state update a transposed K tile would allow, and is NOT numerically valid.
        if (SOKP) ldAm(rb_k, Kc + s * 16 * LDK + gw * 16, LDK, lane);
        else ldBr(rb_k, Kc + s * 16 * LDK + gw * 16, LDK, lane);
#pragma unroll
        for (int vt = 0; vt < NVT; ++vt) {
          uint32_t ra_us[4];
          if (SOLMS) ldAm(ra_us, US + vt * 16 * LDU + s * 16, LDU, lane);
          else ldA(ra_us, US + vt * 16 * LDU + s * 16, LDU, lane);
          mma_rr(S[vt], ra_us, rb_k);
        }
        if (SODN || s <= bi) {
          uint32_t ra_rm[4];
          if (SOLMO) ldAm(ra_rm, Rc + bi * 16 * LDA + s * 16, LDA, lane);
          else ldA(ra_rm, Rc + bi * 16 * LDA + s * 16, LDA, lane);
#pragma unroll
          for (int t = 0; t < VH; ++t)
            if (bv0 + t < NVT) {
              uint32_t rb_ut[4];
              if (MPO) {
                // v142: `UT` already in MMA B-operand order -> no register permutation.
                if (SOLMO) ldBm(rb_ut, UT + (bv0 + t) * 16 * LDU + s * 16, LDU, lane);
                else ldBc(rb_ut, UT + (bv0 + t) * 16 * LDU + s * 16, LDU, lane);
                mma_rc(accT1[t], ra_rm, rb_ut);
              } else {
                if (SOLMO) ldAm(rb_ut, UT + (bv0 + t) * 16 * LDU + s * 16, LDU, lane);
                else ldA(rb_ut, UT + (bv0 + t) * 16 * LDU + s * 16, LDU, lane);
                mma_rc_t(accT1[t], ra_rm, rb_ut);
              }
            }
        }
      }
#pragma unroll
      for (int t = 0; t < VH; ++t)
        if (bv0 + t < NVT) {
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            int i = bi * 16 + rr + half * 8;
            uint32_t w0 =
                cvt_bf16x2(accT1[t][0 + half * 2] * scale, accT1[t][1 + half * 2] * scale);
            uint32_t w1 =
                cvt_bf16x2(accT1[t][4 + half * 2] * scale, accT1[t][5 + half * 2] * scale);
            if (NOST) {
              // v144 PROBE: prices the store; the output tensor is INVALID.
            } else if (OCO) {
              // v144 OCO: re-pack so each lane owns 4 CONSECUTIVE columns, so the row's whole 32 B
              // is one contiguous transaction.  lane j of the quad holds (2j,2j+1) in w0 and
              // (2j+8,2j+9) in w1; after the xor-1 exchange j=0 -> cols 0..3, j=1 -> 8..11,
              // j=2 -> 4..7, j=3 -> 12..15.
              const uint32_t p0 = __shfl_xor_sync(0xffffffffu, w0, 1);
              const uint32_t p1 = __shfl_xor_sync(0xffffffffu, w1, 1);
              const bool odd = (lane & 1) != 0;
              uint2 pk;
              pk.x = odd ? p1 : w0;
              pk.y = odd ? w1 : p0;
              const int col = (((lane & 3) >> 1) * 4) + (odd ? 8 : 0);
              if (i < L) {
                __nv_bfloat16 *dst =
                    out + ((size_t)(t0 + i) * HV + h) * VD + v0 + (bv0 + t) * 16 + col;
                stg64(dst, pk);
              }
            } else if (i < L) {
              __nv_bfloat16 *dst =
                  out + ((size_t)(t0 + i) * HV + h) * VD + v0 + (bv0 + t) * 16 + cc;
              *(uint32_t *)dst = w0;
              *(uint32_t *)(dst + 8) = w1;
            }
          }
        }
    }
    // snapshot the NEXT chunk's entry state into this group's Sb
    // v121 SBAL: Sb overlays UT/US, which state+out (above) was still reading -- one per-group
    // barrier is the entire cost of reclaiming those 17408 B.
    if (SBAL) asm volatile("bar.sync %0, %1;" ::"r"(BAR), "n"(BARN) : "memory");
    if (VPF) {
#pragma unroll
      for (int t = 0; t < VH; ++t)
#pragma unroll
        for (int p = 0; p < 4; ++p) vcur[t][p] = vnxt[t][p];
    }
    if (own) {
  #pragma unroll SAUF
      for (int vt = 0; vt < NVT; ++vt)
  #pragma unroll
        for (int half = 0; half < 2; ++half) {
          __nv_bfloat16 *d = Sb + (vt * 16 + rr + half * 8) * LDS + gw * 16 + cc;
          *(uint32_t *)d = cvt_bf16x2(S[vt][0 + half * 2], S[vt][1 + half * 2]);
          *(uint32_t *)(d + 8) = cvt_bf16x2(S[vt][4 + half * 2], S[vt][5 + half * 2]);
        }
    }
  }

  if (own) {
    float *base = sf + (((size_t)rq * HV + h) * VD + v0) * KD + gw * 16;
#pragma unroll
    for (int vt = 0; vt < NVT; ++vt)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float *d = base + (size_t)(vt * 16 + rr + half * 8) * KD + cc;
        uint2 a, b;
        a.x = __float_as_uint(S[vt][0 + half * 2]);
        a.y = __float_as_uint(S[vt][1 + half * 2]);
        b.x = __float_as_uint(S[vt][4 + half * 2]);
        b.y = __float_as_uint(S[vt][5 + half * 2]);
        stg64(d, a);
        stg64(d + 8, b);
      }
  }
#undef GDR2_STAGE_A
#undef GDR2_STAGE_B
#undef GDR2_VLOAD
}

template <int VB, int SF = 0>
static size_t smem_scan2() {
  constexpr int LDK = KD + 8, LDS = KD + 8, LDA = BT + 8, LDU = BT + 8;
  constexpr int VG = 2, VW = VB * VG, LDV = VW + 8;
  size_t s = 0;
  s += (size_t)BT * LDK * 2 * 2;          // Kb
  s += (size_t)BT * LDK * 2 * 2;          // Qb
  s += (size_t)BT * LDV * 2 * 2;          // Vb
  s += (size_t)BT * LDA * 2 * 2;          // Tm
  s += (size_t)BT * LDA * 2 * 2;          // Rm
  s += (size_t)NF * BT * 4 * 2;           // fac
  if (!(SF & 1)) s += (size_t)VG * VB * LDS * 2;  // Sb per group (v121 SBAL aliases it away)
  s += (size_t)VG * 3 * VB * LDU * 2;     // ZT/UT/US per group
  return s;
}

// ---------------------------------------------------------------- host side
template <int VB>
static size_t smem_bytes() {
  constexpr int LDK = KD + 8, LDS = KD + 8, LDA = BT + 8, LDU = BT + 8, LDV = VB + 8,
                LDX = 24;
  size_t s = 0;
  s += (size_t)VB * LDS * 2;
  s += (size_t)BT * LDK * 2 * 2;
  s += (size_t)BT * LDV * 2;
  s += (size_t)BT * LDA * 2 * 2;
  s += (size_t)VB * LDU * 2 * 3;
  s += (size_t)(BT / 16) * 16 * LDX * 2;
  s += (size_t)BT * 4 * 7 + 4;
  return s;
}

// varlen-v2: `gdr_forward` (the single-launch gdr_fused monolith) is not built --
// every shape in this tree has nct >= 2 and takes the split path.

std::vector<torch::Tensor> gdr_forward_split(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                             torch::Tensor g, torch::Tensor beta,
                                             torch::Tensor s0, torch::Tensor cu,
                                             torch::Tensor tg, torch::Tensor rg, torch::Tensor fg,
                                             torch::Tensor kn, torch::Tensor qn,
                                             int64_t vb_sel, int64_t mask, int64_t var,
                                             int64_t pgrp, int64_t uv) {
  const int T = q.size(1);
  const int HQ = q.size(2);
  const int HV = v.size(2);
  const int N = s0.size(0);
  const int rep = HV / HQ;
  const float scale = 1.0f / sqrtf((float)KD);
  const int nct = (T + BT - 1) / BT;
  auto out = torch::empty({1, T, HV, VD}, v.options());
  // reference_v2 ALWAYS returns an fp32 final_state, even when initial_state is bf16
  // (registered deviation 7), and the recurrence accumulates in fp32 regardless.
  auto sf = torch::empty(s0.sizes(), s0.options().dtype(torch::kFloat32));
  const int gfv = (g.scalar_type() == torch::kFloat32) ? 1 : 0;
  const int sdv = (s0.scalar_type() == torch::kFloat32) ? 1 : 0;
  auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_PROLOGUE_PI(KN, NG, RVV, SKV, PIV)                                                  \
  {                                                                                        \
    static bool pdones = false;                                                            \
    size_t psm = smem_prologue<NG, RVV>();                                                   \
    if (!pdones) {                                                                         \
      cudaFuncSetAttribute((KN<NG, ((PIV)&16) != 0, true, RVV, SKV, PIV>),                              \
                           cudaFuncAttributeMaxDynamicSharedMemorySize, (int)psm);         \
      pdones = true;                                                                       \
    }                                                                                      \
    KN<NG, ((PIV)&16) != 0, true, RVV, SKV, PIV><<<dim3(nct, HQ, N), (NG) * 128, psm, stream>>>(               \
        (const __nv_bfloat16 *)k.data_ptr(), (const __nv_bfloat16 *)q.data_ptr(),           \
        g.data_ptr(), (const __nv_bfloat16 *)beta.data_ptr(), cu.data_ptr<int>(),           \
        (__nv_bfloat16 *)tg.data_ptr(), (__nv_bfloat16 *)rg.data_ptr(),                     \
        fg.data_ptr<float>(), (__nv_bfloat16 *)kn.data_ptr(),                               \
        (__nv_bfloat16 *)qn.data_ptr(), HQ, HV, rep, gfv);                                   \
  }
  // v171: value-keyed launch -- NGRP=1, 128 threads, grid.y = HV, VK=1.
#define LAUNCH_PROLOGUE_VK(KN, RVV, SKV, PIV)                                                  \
  {                                                                                        \
    static bool pdvk = false;                                                              \
    size_t psm = smem_prologue<1, RVV>();                                                   \
    if (!pdvk) {                                                                           \
      cudaFuncSetAttribute((KN<1, ((PIV)&16) != 0, true, RVV, SKV, PIV, 1>),               \
                           cudaFuncAttributeMaxDynamicSharedMemorySize, (int)psm);         \
      pdvk = true;                                                                         \
    }                                                                                      \
    KN<1, ((PIV)&16) != 0, true, RVV, SKV, PIV, 1><<<dim3(nct, HV, N), 128, psm, stream>>>( \
        (const __nv_bfloat16 *)k.data_ptr(), (const __nv_bfloat16 *)q.data_ptr(),           \
        g.data_ptr(), (const __nv_bfloat16 *)beta.data_ptr(), cu.data_ptr<int>(),           \
        (__nv_bfloat16 *)tg.data_ptr(), (__nv_bfloat16 *)rg.data_ptr(),                     \
        fg.data_ptr<float>(), (__nv_bfloat16 *)kn.data_ptr(),                               \
        (__nv_bfloat16 *)qn.data_ptr(), HQ, HV, rep, gfv);                                   \
  }
  // v111: +64 = __launch_bounds__(...,3); bits 8..17 = RV mask; bits 18..25 = SK (the v143
  // stage-drop instrument); v151: bits 26..29 = PI (the diagonal-inverse knobs).
  {
    const int rv = ((int)pgrp >> 8) & 1023;
    const int skv = ((int)pgrp >> 18) & 255;
    const int piv = ((int)pgrp >> 26) & 15;
    // v151 probe vehicle: PI x {SK=120 (stop after the A epilogue), SK=112 (+ the four 16x16
    // diagonal inverses), SK=0 (FULL)} is exactly the ladder that isolates the stage this round
    // attacks.  `_pick_prv`'s RV=256 flavour (short shapes) keeps its single PI=0 instantiation.
    // v155 MINIMAL BUILD: exactly TWO prologue kernels, and `PI` is bound to the flavour that
    // `_pick_prv` already selects from metadata.  MEASURED reason (v151 `rvprobe.py`, v154
    // `bench.sh`): `PI=13`'s two work-halving knobs (`DIH`, and `BI2`'s 2-warp inverse) pay
    // **only where the prologue grid oversubscribes 4 blocks/SM**, i.e. exactly the `RV=480`
    // flavour.  At `RV=256` (nct*HQ*N <= 78 blocks, sid3's 48) the SM already holds every block
    // it is given, halving the warps that issue a stage removes latency-hiding it cannot replace,
    // and `PI=13` measures **+0.19 %** (T=764, 41-rep interleaved) -- so that flavour keeps
    // `PI=0`, i.e. it stays byte-identical to v145's shipped prologue.  Same metadata-only rule,
    // no tensor values, one extra `constexpr` branch.
    // v171 MINIMAL BUILD: exactly ONE prologue instantiation, gdr_prologue<2,false,true,256,0,13>,
    // for EVERY shape.  MEASURED (workspace/v171/vkprobe.py, 40-rep interleaved, same binary):
    // RV256+PI13 beats the v155-shipped RV480+PI13 on every shape --
    //   sid0 (960 blk) 790.1 vs 798.1  (-1.0 %)   sid2 (80 blk) 80.9 vs 82.1  (-1.5 %)
    //   sid3 (48 blk)  58.8 vs 59.0                RV256 PI0 (v155 sid3) 59.0
    // WHY: `BI2` (in PI13) deletes the fully-unrolled blocked-inverse b loop that was RV256's
    // register peak (v152), taking it 176 -> 120 regs = 4 blocks/SM -- the same occupancy RV480
    // reached by ALSO `#pragma unroll 1`-ing the A/R epilogues (INU1|AEU1|REU1).  With BI2 supplying
    // the occupancy, those unroll caps are pure ILP loss, so keeping the epilogues fully unrolled
    // (RV256 = NOXI alone) is uniformly faster.  The v112/v155 choice of RV480 predates BI2.
    // The metadata split (_pick_prv) collapses: one flavour wins the whole range 48..960 blocks.
    (void)rv; (void)skv; (void)piv;
    // ===== g8: split the prologue's VALUE-HEAD GROUP COUNT on grid occupancy ==================
    // MEASURED problem (bench.sh gen:g2 per-kernel): the prologue costs 20.60 us of T=128's
    // 30.92 us op_lat, and 20.00 / 21.58 / 22.27 / 23.41 us at nct = 4 / 12 / 16 / 20 -- a flat
    // per-BLOCK latency floor, not a throughput cost (T=15360's 154.21 us / 960 blocks is 6.15
    // waves x 25 us).  `athanor profile` on the T=128 launch: bound = latency, long pole salu at
    // 14.9 % util, with the grid only nct*HQ*N = 8 blocks on 39 SMs.
    //
    // With NGRP=2 and rep = HV/HQ = 4, each block runs its four value-head epilogues (A -> four
    // 16x16 diagonal inverses -> blocked 64x64 inverse -> emit T,R) in TWO SEQUENTIAL passes.
    // NGRP=4 gives each of the four 4-warp groups its own value head, so the epilogue runs ONCE.
    // MEASURED (workspace/g7, single NGRP=4 instantiation, same probe):
    //     T=128    25.45 us vs g2's 28.67  (-11.2 %)
    //     T=256    31.39 us vs g2's 34.53  ( -9.1 %)
    //     T=15360 858.32 us vs g2's 803.35 ( +6.8 %)
    // i.e. NGRP=4 buys latency and loses throughput -- at 960 blocks the wide block wastes warps
    // 8..15 during the shared-Gram stage (warps 0..7 build both Grams) and drops to 2 blocks/SM.
    // v44's original rejection of the wide block was measured at 184 regs (1 block/SM, the v24
    // failure mode); RV=256's NOXI + PI=13's BI2 have since taken it to 120, which is why the
    // small-grid half of the trade is now available at all.
    //
    // The predicate is nct*HQ*N, i.e. shape metadata (T, HQ, N) only: no tensor values, no
    // autotune-on-T, both flavours instantiated unconditionally at build time.  32 is the
    // occupancy knee -- below it the key-keyed grid cannot fill 39 SMs even at 1 block/SM.
    // ===== g9: the knee is 64 blocks, not 32 -- MEASURED =====================================
    // g8 set the NGRP=4 predicate at nct*HQ*N <= 32 because that was the largest grid it had
    // A/B'd.  The real knee is the point where the grid stops fitting in ONE wave of resident
    // blocks: NGRP=4 is 512 threads at 120 regs => 120*512 = 61440 of 131072 regs => 2 blocks/SM,
    // so 39 SMs hold 78 NGRP=4 blocks concurrently.  Below that the wide block's shorter
    // per-block critical path (one epilogue pass instead of two) is free; above it the wasted
    // warps 8..15 of the shared-Gram stage start costing throughput.
    // MEASURED (workspace/g9, one object, knee swept through an env var, probe_t128.py):
    //     nct=2   (  8 blocks)  25.44 -> 25.37   (both NGRP=4, control)
    //     nct=4   ( 16 blocks)  31.34 -> 31.22   (both NGRP=4, control)
    //     nct=12  ( 48 blocks)  58.00 -> 55.99   -3.5 %   <- WON, g8 was leaving this on the table
    //     nct=16  ( 64 blocks)  69.02 -> 67.35   -2.4 %   <- WON
    //     nct=20  ( 80 blocks)  81.02 -> 89.36  +10.3 %   <- LOST, 80 > 78 = the 2-blocks/SM wall
    //     nct=240 (960 blocks) 803.27 -> 803.25  (both NGRP=2, control)
    // 64 is therefore the shipped knee: it takes both measured wins and stays clear of the
    // measured loss at 80.  Still shape metadata (T, HQ, N) only -- no tensor values, no
    // autotune-on-T, both flavours instantiated unconditionally at build time.
    if (nct * HQ * N <= 64) {
      LAUNCH_PROLOGUE_PI(gdr_prologue, 4, 256, 0, 13)
    } else {
      LAUNCH_PROLOGUE_PI(gdr_prologue, 2, 256, 0, 13)
    }
#undef PDISP_PI
  }
#undef LAUNCH_PROLOGUE_PI
#undef LAUNCH_PROLOGUE_VK

#define LAUNCH_SCAN(VBV, VARV)                                                                  \
  {                                                                                       \
    static bool done = false;                                                             \
    size_t sm = smem_scan<VBV>();                                                         \
    if (!done) {                                                                          \
      cudaFuncSetAttribute(gdr_scan<VBV, VARV>,                                           \
                           cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);         \
      done = true;                                                                        \
    }                                                                                     \
    gdr_scan<VBV, VARV><<<dim3(VD / VBV, HV, N), NW * 32, sm, stream>>>(                  \
        (const __nv_bfloat16 *)q.data_ptr(), (const __nv_bfloat16 *)k.data_ptr(),          \
        (const __nv_bfloat16 *)v.data_ptr(), (const __nv_bfloat16 *)tg.data_ptr(),         \
        (const __nv_bfloat16 *)rg.data_ptr(),                                              \
        fg.data_ptr<float>(), s0.data_ptr<float>(), cu.data_ptr<int>(),                    \
        (__nv_bfloat16 *)out.data_ptr(), sf.data_ptr<float>(), HQ, HV, rep, scale,          \
        (int)mask);                                                                        \
  }
  // v135 MINIMAL BUILD: `solution.py` always sends VAR=18, i.e. bit 16 set, so the v28-era
  // `gdr_scan<VB,VAR>` (one 8-warp group per block) is unreachable; v45's two-group `gdr_scan2` has
  // superseded it on every shape since. Not instantiated.
  {
    // v45: one 512-thread block = two independent 8-warp groups of VBV V rows, sharing one
    // staged copy of K/Q/T/R/fac.  Grid halves to (VD/(2*VBV), HV, N).
    // v121: the host `uv` word is  UV | (SF << 8).  UV selects the v51 U-solve variant, SF the
    // semantically-free footprint knobs (Sb aliasing + per-site unroll caps).
    // v131: bits 16..23 carry SO, the state+out fragment-feed knobs.  The settled UV (v51) and SF
    // (v121) sweep values are no longer instantiated -- both axes are closed, and every extra
    // instantiation in the translation unit costs up to ~0.9 % of op_lat (v121's measurement
    // hazard), so the TU carries only the shipped UV=12 x {SF=0,SF=4} x SO cross product.
    // v132: the host word is UV | SF<<8 | SO<<16 | GS<<24.  Only a CURATED tuple list is
    // instantiated -- v121 measured unused instantiations at up to +0.9 % of op_lat.
    const int uvm = (int)uv & 255;
    const int sfm = ((int)uv >> 8) & 255;
    const int som = ((int)uv >> 16) & 255;
    const int gsm = ((int)uv >> 24) & 255;
#define LAUNCH_S2(UVV, SFV, SOV, GSV, WHV, WUV, MPV, OQV)                                        \
    {                                                                                       \
      size_t sm2 = smem_scan2<32, SFV>();                                                   \
      static bool d2 = false;                                                               \
      if (!d2) {                                                                            \
        cudaFuncSetAttribute((gdr_scan2<32, 2, UVV, SFV, SOV, GSV, WHV, WUV, MPV, OQV>),      \
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm2);         \
        d2 = true;                                                                           \
      }                                                                                      \
      gdr_scan2<32, 2, UVV, SFV, SOV, GSV, WHV, WUV, MPV, OQV>                                \
          <<<dim3(VD / 64, HV, N), NW2 * 32 * WHV, sm2, stream>>>(                            \
          (const __nv_bfloat16 *)qn.data_ptr(), (const __nv_bfloat16 *)kn.data_ptr(),         \
          (const __nv_bfloat16 *)v.data_ptr(), (const __nv_bfloat16 *)tg.data_ptr(),          \
          (const __nv_bfloat16 *)rg.data_ptr(), fg.data_ptr<float>(), s0.data_ptr(),          \
          cu.data_ptr<int>(), (__nv_bfloat16 *)out.data_ptr(), sf.data_ptr<float>(), HQ, HV,  \
          rep, scale, (int)mask, sdv);                                                        \
    }
#define S2_CASE(U, S, O, G)                                                                 \
    if (uvm == (U) && sfm == (S) && som == (O) && gsm == (G)) { LAUNCH_S2(U, S, O, G) } else
    // v135 MINIMAL BUILD: exactly ONE `gdr_scan2` instantiation -- the v133 winner
    //   UV=9  dense U solve, ONE accumulator, native ld.mat   (v132: v51's four split accumulators
    //         REVERSED once SO=7 changed the surrounding register pressure, -17 us)
    //   SO=7  SOLMS|SOLMO|SODN: all 16 A-order fragments on `tsm.ld.mat.b32x4` + dense output
    //   GS=2  AHG: staging addresses hoisted out of the chunk loop (TRTRI/QDIR refuted)
    (void)uvm; (void)sfm; (void)som; (void)gsm;
    // v141: the host `var` word carries the WH sweep in bits 8..15 -- 0/1 = WH1 (v135, the control),
    // 2 = WH2 WU0 (helper half idles through the U solve), 3 = WH2 WU1 (helper half owns the U solve).
    // v142: bits 16..23 of the host `var` word carry MP (the row.col B fragment-order knob).
    // ===== v145 MINIMAL BUILD =====================================================
    // v121 measured `op_lat` moving by up to **+0.9 % with kernels that are never launched**, so
    // the shipped winner is re-built with exactly ONE `gdr_scan2` instantiation.  v141-v144 were
    // sweep vehicles carrying 3, 9, 14 and 5 of them; this object carries 1.
    //   UV=9  dense U solve, one accumulator, native `ld.mat`               (v133)
    //   SO=7  SOLMS|SOLMO|SODN -- all 16 A-order fragments native + dense R.U (v131, -8.3 %)
    //   GS=2  AHG: staging addresses hoisted out of the chunk loop           (v132, -1.8 %)
    //   WH=1  512 threads / 16 warps per SM (WH=2 refuted twice: v125, v141)
    //   MP=6  MPU|MPO -- the `row.col` B fragments of the U solve and the output loaded ALREADY
    //         IN MMA OPERAND ORDER by `ldBm`, deleting 26 of the 66 `v.madl.i32` register copies
    //         the operand-list permutation forced                            (v142, -0.7...-0.9 %)
    //   OQ=0  plain output store (OCO refuted, v144)
    (void)var;
    LAUNCH_S2(9, 0, 7, 2, 1, 0, 6, 0)
#undef S2_CASE
#undef LAUNCH_S2
  }
#undef LAUNCH_SCAN
  return {out, sf};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_split", &gdr_forward_split);
}
