#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
using bf16=__nv_bfloat16;
__device__ __forceinline__ float fp(bf16 x){return __bfloat162float(x);}
__device__ __forceinline__ float fp(float x){return x;}
__device__ __forceinline__ bf16 bf(float x){return __float2bfloat16_rn(x);}
__device__ __forceinline__ unsigned pack(bf16 a,bf16 b){return unsigned(*reinterpret_cast<unsigned short*>(&a))|(unsigned(*reinterpret_cast<unsigned short*>(&b))<<16);}
__device__ __forceinline__ void mma16(float (&d)[8],const unsigned (&a)[4],const unsigned (&b)[4]){
 asm volatile("wmma.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, {%0,%1,%2,%3,%4,%5,%6,%7};" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(b[2]),"r"(b[3]));
}
__device__ __forceinline__ int sw(int row,int col,int ld){return row*ld+(col^((row&7)*8));}
__device__ __forceinline__ int u_col4(int c) {
 return (c&~15)|(c&1)|((c&6)<<1)|((c&8)>>2);
}
// Native m16n16 mapping: each lane owns four FP32 values in each 8-column half.
// Both logical input operands remain row-major shared matrices.
template<int M,int N,int K> __device__ __forceinline__ void gemm(const bf16* a,const bf16* b,float (&out)[M*N/4096][8]){
 int lane=threadIdx.x&31,warp=threadIdx.x>>5,r=lane>>2,c=(lane&3)*2;
 #pragma unroll
 for(int tile=0;tile<M*N/4096;tile++){
  int id=warp+tile*16,m=(id/(N/16))*16,n=(id%(N/16))*16;
  #pragma unroll
  for(int e=0;e<8;e++)out[tile][e]=0;
  #pragma unroll
  for(int k=0;k<K;k+=16){
   unsigned ar[4],br[4];
   unsigned pa=(unsigned)__cvta_generic_to_shared(a+sw(m+(lane&15),k+(lane>>4)*8,K));
   unsigned pb=(unsigned)__cvta_generic_to_shared(b+sw(k+(lane&15),n+(lane>>4)*8,N));
   asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(ar[0]),"=r"(ar[1]),"=r"(ar[2]),"=r"(ar[3]):"r"(pa));
   asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(br[0]),"=r"(br[1]),"=r"(br[2]),"=r"(br[3]):"r"(pb));
   mma16(out[tile],ar,br);
  }
 }
}
__device__ __forceinline__ void gemm_update_natural(const bf16* a,const bf16* b,float (&out)[2][8]){
 constexpr int M=128,N=64,K=64;
 int lane=threadIdx.x&31,warp=threadIdx.x>>5,r=lane>>2,c=(lane&3)*2;
 #pragma unroll
 for(int tile=0;tile<M*N/4096;tile++){
  int id=warp+tile*16,m=(id/(N/16))*16,n=(id%(N/16))*16;
  #pragma unroll
  for(int e=0;e<8;e++)out[tile][e]=0;
  #pragma unroll
  for(int k=0;k<K;k+=16){
   unsigned ar[4],br[4];
   unsigned pa=(unsigned)__cvta_generic_to_shared(a+sw(k+(lane&7)+(lane>>4)*8,m+((lane>>3)&1)*8,128));
   unsigned pb=(unsigned)__cvta_generic_to_shared(b+sw(k+(lane&15),n+(lane>>4)*8,N));
   asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(ar[0]),"=r"(ar[1]),"=r"(ar[2]),"=r"(ar[3]):"r"(pa));
   asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(br[0]),"=r"(br[1]),"=r"(br[2]),"=r"(br[3]):"r"(pb));
   mma16(out[tile],ar,br);
  }
 }
}
template<int N> __device__ __forceinline__ void coord(int tile,int e,int& r,int& c){
 int lane=threadIdx.x&31,id=(threadIdx.x>>5)+tile*16;
 r=(id/(N/16))*16+(lane>>2)+((e&2)?8:0);c=(id%(N/16))*16+(lane&3)*2+(e&1)+((e&4)?8:0);
}
__device__ __forceinline__ int chunk_base(int start,int seq){return start/64+seq;}
template<class G> __global__ void prepare(const bf16* q,const bf16* k,const G* g,const int* bounds,bf16* qp,bf16* kp,float* gp,int total,int hq,int hv,int ns,int nc){
 int ch=blockIdx.x,h=blockIdx.y,start=-1,end=0;
 for(int s=0;s<ns;s++){int b=chunk_base(bounds[s],s),n=(bounds[s+1]-bounds[s]+63)/64;if(ch>=b&&ch<b+n){start=bounds[s]+64*(ch-b);end=bounds[s+1];break;}}
 if(start<0)return;
 int lane=threadIdx.x&31,warp=threadIdx.x>>5,qh=h/(hv/hq);
 long off=(long(h)*nc+ch)*8192;
 for(int r=warp;r<64;r+=4){
  float qs[4],ks[4],qn=0,kn=0;int token=start+r;
  #pragma unroll
  for(int z=0;z<4;z++){int d=lane+32*z;qs[z]=token<end?fp(q[(long(token)*hq+qh)*128+d]):0;ks[z]=token<end?fp(k[(long(token)*hq+qh)*128+d]):0;qn+=qs[z]*qs[z];kn+=ks[z]*ks[z];}
  #pragma unroll
  for(int s=16;s;s>>=1){qn+=__shfl_xor_sync(0xffffffff,qn,s);kn+=__shfl_xor_sync(0xffffffff,kn,s);}
  qn=rsqrtf(qn+1e-6f);kn=rsqrtf(kn+1e-6f);
  #pragma unroll
  for(int z=0;z<4;z++){int d=lane+32*z;qp[off+r*128+d]=bf(qs[z]*qn);kp[off+r*128+d]=bf(ks[z]*kn);}
 }
 if(threadIdx.x==0){float sum=0;for(int r=0;r<64;r++){if(start+r<end)sum+=fp(g[long(start+r)*hv+h]);gp[(long(h)*nc+ch)*64+r]=sum;}}
}
struct KktShared{bf16 k[8192];bf16 kt[8192];float gram[4096];float inv[4096];};
__global__ void inverse(const bf16* kp,const bf16* beta,const float* gp,const int* bounds,bf16* ap,int hv,int ns,int nc){
 int ch=blockIdx.x,h=blockIdx.y,start=-1,end=0;
 for(int s=0;s<ns;s++){int b=chunk_base(bounds[s],s),n=(bounds[s+1]-bounds[s]+63)/64;if(ch>=b&&ch<b+n){start=bounds[s]+64*(ch-b);end=bounds[s+1];break;}}
 if(start<0)return;
 __shared__ KktShared s;int t=threadIdx.x;long off=(long(h)*nc+ch)*8192;
 for(int x=t;x<8192;x+=512){s.k[sw(x/128,x%128,128)]=kp[off+x];s.kt[sw(x%128,x/128,64)]=kp[off+x];}
 __syncthreads();float gram[1][8];gemm<64,64,128>(s.k,s.kt,gram);
 #pragma unroll
 for(int i=0;i<1;i++){
  #pragma unroll
  for(int e=0;e<8;e++){int r,c;coord<64>(i,e,r,c);s.gram[r*64+c]=gram[i][e];}
 }
 // MMA fragment ownership differs from the next linear shared-memory traversal.
 __syncthreads();
 for(int x=t;x<4096;x+=512){int r=x/64,c=x%64;float b=start+r<end?fp(beta[long(start+r)*hv+h]):0;s.gram[x]=(r>c)?b*s.gram[x]:0;s.inv[x]=r==c?1:0;}
 __syncthreads();
 // Invert four diagonal 16x16 blocks concurrently in FP32.
 // L = Ldiag + Loff; D=(I+Ldiag)^-1; B=-D*Loff is strictly block lower.
 // B^4=0, so (I+L)^-1=(I+B+B^2+B^3)*D exactly before BF16 rounding.
 for(int rr=1;rr<16;rr++){
  if(t<64){int base=(t/16)*16,col=t%16,row=base+rr;
   if(col<rr){float sum=0;for(int j=col;j<rr;j++)sum=fmaf(s.gram[row*64+base+j],s.inv[(base+j)*64+base+col],sum);s.inv[row*64+base+col]=-sum;}
  }
  __syncthreads();
 }
 for(int x=t;x<4096;x+=512){int r=x/64,c=x%64;
  s.k[sw(r,c,64)]=bf(s.inv[x]);
  s.kt[sw(r,c,64)]=bf(r/16>c/16?s.gram[x]:0.f);
 }
 __syncthreads();
 float product[1][8];gemm<64,64,64>(s.k,s.kt,product);
 __syncthreads();
 #pragma unroll
 for(int e=0;e<8;e++){int r,c;coord<64>(0,e,r,c);
  s.k[sw(r,c,64)]=bf(r/16>c/16?-product[0][e]:0.f);
  s.kt[sw(r,c,64)]=bf(s.inv[r*64+c]);
 }
 __syncthreads();
 #pragma unroll
 for(int stage=0;stage<3;stage++){
  gemm<64,64,64>(s.k,s.kt,product);
  __syncthreads();
  #pragma unroll
  for(int e=0;e<8;e++){int r,c;coord<64>(0,e,r,c);float value=product[0][e]+s.inv[r*64+c];
   if(stage<2)s.kt[sw(r,c,64)]=bf(value);
   else {float weight=r>=c?expf(gp[(long(h)*nc+ch)*64+r]-gp[(long(h)*nc+ch)*64+c]):0.f;ap[(long(h)*nc+ch)*4096+r*64+c]=bf(value*weight);}
  }
  __syncthreads();
 }

}
struct MatrixShared{bf16 a[8192],b[8192];};
__global__ void basis(const bf16* qp,const bf16* kp,const bf16* v,const bf16* beta,const float* gp,const bf16* ap,const int* bounds,bf16* wp,float* up,bf16* pp,int hv,int ns,int nc){
 int ch=blockIdx.x,h=blockIdx.y,start=-1,end=0,t=threadIdx.x;
 for(int seq=0;seq<ns;seq++){int b=chunk_base(bounds[seq],seq),n=(bounds[seq+1]-bounds[seq]+63)/64;if(ch>=b&&ch<b+n){start=bounds[seq]+64*(ch-b);end=bounds[seq+1];break;}}
 if(start<0)return;
 long packed=long(h)*nc+ch,off=packed*8192;
 __shared__ MatrixShared s;
 #pragma unroll 4
 for(int j=0;j<16;j++){unsigned x=unsigned(t)+unsigned(j)*512;int r=x/128,c=x%128;s.a[sw(r,c,128)]=qp[off+x];s.b[sw(c,r,64)]=kp[off+x];}
 __syncthreads();float p[1][8];gemm<64,64,128>(s.a,s.b,p);
 #pragma unroll
 for(int e=0;e<8;e++){int r,c;coord<64>(0,e,r,c);pp[packed*4096+r*64+c]=bf(r>=c?p[0][e]*expf(gp[packed*64+r]-gp[packed*64+c]):0.f);}
 __syncthreads();
 #pragma unroll 4
 for(int j=0;j<8;j++){unsigned x=unsigned(t)+unsigned(j)*512;s.a[sw(x/64,x%64,64)]=ap[packed*4096+x];}
 #pragma unroll 4
 for(int j=0;j<16;j++){unsigned x=unsigned(t)+unsigned(j)*512;int r=x/128,c=x%128;float b=start+r<end?fp(beta[long(start+r)*hv+h]):0.f;s.b[sw(r,c,128)]=bf(b*expf(gp[packed*64+r])*fp(kp[off+x]));}
 __syncthreads();float z[2][8];gemm<64,128,64>(s.a,s.b,z);
 #pragma unroll
 for(int i=0;i<2;i++){
  #pragma unroll
  for(int e=0;e<8;e++){int r,c;coord<128>(i,e,r,c);wp[off+r*128+c]=bf(z[i][e]);}
 }
 __syncthreads();
 #pragma unroll 4
 for(int j=0;j<16;j++){unsigned x=unsigned(t)+unsigned(j)*512;int r=x/128,c=x%128;float b=start+r<end?fp(beta[long(start+r)*hv+h]):0.f;float val=start+r<end?fp(v[(long(start+r)*hv+h)*128+c]):0.f;s.b[sw(r,c,128)]=bf(b*val);}
 __syncthreads();gemm<64,128,64>(s.a,s.b,z);
 #pragma unroll
 for(int i=0;i<2;i++){
  #pragma unroll
  for(int e=0;e<4;e+=2){
   int r,c;coord<128>(i,e,r,c);
   float4 value=make_float4(z[i][e],z[i][e+1],z[i][e+4],z[i][e+5]);
   *reinterpret_cast<float4*>(up+off+r*128+u_col4(c))=value;
  }
 }
}
template<class S> __global__ void state_scan(const bf16* kp,const float* gp,const bf16* wp,const float* up,const S* initial,const int* bounds,bf16* hs,bf16* vn,float* final,int hv,int nc){
 int h=blockIdx.x/2,part=blockIdx.x%2,seq=blockIdx.y,t=threadIdx.x;
 int left=bounds[seq],right=bounds[seq+1],n=(right-left+63)/64;
 long statebase=(long(seq)*hv+h)*16384;
 __shared__ __align__(16) MatrixShared s;
 float state[2][8];
 #pragma unroll
 for(int i=0;i<2;i++){
  #pragma unroll
  for(int e=0;e<8;e++){int r,c;coord<64>(i,e,r,c);state[i][e]=fp(initial[statebase+(part*64+c)*128+r]);}
 }
 for(int chunk=0;chunk<n;chunk++){
  int ch=chunk_base(left,seq)+chunk;long packed=long(h)*nc+ch,off=packed*8192;
  // Each thread transfers two aligned groups of eight BF16 W elements.
  #pragma unroll
  for(int j=0;j<2;j++){
   int row=(t>>4)+j*32,col=(t&15)*8;
   uint4 value=*reinterpret_cast<const uint4*>(wp+off+row*128+col);
   *reinterpret_cast<uint4*>(s.a+sw(row,col,128))=value;
  }
  #pragma unroll
  for(int i=0;i<2;i++){
   #pragma unroll
   for(int e=0;e<8;e++){int r,c;coord<64>(i,e,r,c);bf16 val=bf(state[i][e]);s.b[sw(r,c,64)]=val;hs[packed*16384+r*128+part*64+c]=val;}
  }
  __syncthreads();float pred[1][8];gemm<64,64,128>(s.a,s.b,pred);
  __syncthreads();
  #pragma unroll
  for(int e=0;e<4;e+=2){
   int r,c;coord<64>(0,e,r,c);
   long idx=off+r*128+part*64+c;
   float4 u=*reinterpret_cast<const float4*>(up+off+r*128+u_col4(part*64+c));
   unsigned bits0=pack(bf(u.x-pred[0][e]),bf(u.y-pred[0][e+1]));
   unsigned bits1=pack(bf(u.z-pred[0][e+4]),bf(u.w-pred[0][e+5]));
   *reinterpret_cast<unsigned*>(s.b+sw(r,c,64))=bits0;
   *reinterpret_cast<unsigned*>(vn+idx)=bits0;
   *reinterpret_cast<unsigned*>(s.b+sw(r,c+8,64))=bits1;
   *reinterpret_cast<unsigned*>(vn+idx+8)=bits1;
  }
  #pragma unroll
  for(int j=0;j<2;++j){
   int r=(t>>4)+j*32,k=(t&15)*8;
   uint4 raw=*reinterpret_cast<const uint4*>(kp+off+r*128+k);
   float d=expf(gp[packed*64+63]-gp[packed*64+r]);
   const bf16* v=reinterpret_cast<const bf16*>(&raw);
   uint4 out;
   out.x=pack(bf(fp(v[0])*d),bf(fp(v[1])*d));
   out.y=pack(bf(fp(v[2])*d),bf(fp(v[3])*d));
   out.z=pack(bf(fp(v[4])*d),bf(fp(v[5])*d));
   out.w=pack(bf(fp(v[6])*d),bf(fp(v[7])*d));
   *reinterpret_cast<uint4*>(s.a+sw(r,k,128))=out;
  }
  __syncthreads();float delta[2][8];gemm_update_natural(s.a,s.b,delta);
  float eg=expf(gp[packed*64+63]);
  #pragma unroll
  for(int i=0;i<2;i++){
   #pragma unroll
   for(int e=0;e<8;e++)state[i][e]=fmaf(eg,state[i][e],delta[i][e]);
  }
  __syncthreads();
 }
 #pragma unroll
 for(int i=0;i<2;i++){
  #pragma unroll
  for(int e=0;e<8;e++){int r,c;coord<64>(i,e,r,c);final[statebase+(part*64+c)*128+r]=state[i][e];}
 }
}
__global__ void outputs(const bf16* qp,const float* gp,const bf16* pp,const bf16* hs,const bf16* vn,const int* bounds,bf16* output,int hv,int ns,int nc){
 int ch=blockIdx.x,h=blockIdx.y/2,part=blockIdx.y%2,start=-1,end=0,t=threadIdx.x;
 for(int seq=0;seq<ns;seq++){int b=chunk_base(bounds[seq],seq),n=(bounds[seq+1]-bounds[seq]+63)/64;if(ch>=b&&ch<b+n){start=bounds[seq]+64*(ch-b);end=bounds[seq+1];break;}}
 if(start<0)return;
 long packed=long(h)*nc+ch,off=packed*8192;__shared__ __align__(16) MatrixShared s;
 #pragma unroll
 for(int j=0;j<2;j++){
  int qgroup=t+j*512,qrow=qgroup>>4,qcol=(qgroup&15)*8;
  uint4 qvalue=*reinterpret_cast<const uint4*>(qp+off+qrow*128+qcol);
  *reinterpret_cast<uint4*>(s.a+sw(qrow,qcol,128))=qvalue;
  int hgroup=t+j*512,hrow=hgroup>>3,hcol=(hgroup&7)*8;
  uint4 hvalue=*reinterpret_cast<const uint4*>(hs+packed*16384+hrow*128+part*64+hcol);
  *reinterpret_cast<uint4*>(s.b+sw(hrow,hcol,64))=hvalue;
 }
 __syncthreads();float out[1][8];gemm<64,64,128>(s.a,s.b,out);
 #pragma unroll
 for(int e=0;e<8;e++){int r,c;coord<64>(0,e,r,c);out[0][e]*=expf(gp[packed*64+r]);}
 __syncthreads();
 {int r=t>>3,c=(t&7)*8;
  uint4 pvalue=*reinterpret_cast<const uint4*>(pp+packed*4096+r*64+c);
  uint4 vvalue=*reinterpret_cast<const uint4*>(vn+off+r*128+part*64+c);
  *reinterpret_cast<uint4*>(s.a+sw(r,c,64))=pvalue;
  *reinterpret_cast<uint4*>(s.b+sw(r,c,64))=vvalue;
 }
 __syncthreads();float within[1][8];gemm<64,64,64>(s.a,s.b,within);
 #pragma unroll
 for(int e=0;e<8;e++){int r,c;coord<64>(0,e,r,c);if(start+r<end)output[(long(start+r)*hv+h)*128+part*64+c]=bf((out[0][e]+within[0][e])*0.08838834764831845f);}
}
// The CP geometry is shared by metadata, prefix, and replay.
__device__ __forceinline__ void cp_partition_geometry(int n,int& length,int& count){
 if(n==0){length=0;count=1;}
 else if(n<2048){length=n;count=1;}
 else{length=(n+7)/8;count=(n+length-1)/length;}
}
__global__ void cp_metadata(const float* gp,const int* bounds,int* meta,int hv,int nc){
 int p=blockIdx.x,seq=blockIdx.y,h=threadIdx.x;
 if(h>=hv)return;
 int left=bounds[seq],n=(bounds[seq+1]-left+63)/64,length,count;
 cp_partition_geometry(n,length,count);
 long descriptor=((long(seq)*8+p)*hv+h)*4;
 if(p>=count){meta[descriptor]=0;meta[descriptor+1]=0;meta[descriptor+2]=0;meta[descriptor+3]=3;return;}
 int begin=p*length,end=min(n,begin+length);
 meta[descriptor]=begin;meta[descriptor+1]=end;
 if(p+1==count){meta[descriptor+2]=0;meta[descriptor+3]=2;return;}
 float sum=0.f;int warmup=0,flag=1;
 for(int chunk=end-1;chunk>=begin;--chunk){
  long packed=long(h)*nc+chunk_base(left,seq)+chunk;
  sum+=gp[packed*64+63];++warmup;
  if(sum<-10.f){flag=0;break;}
 }
 meta[descriptor+2]=warmup;meta[descriptor+3]=flag;
}

template<bool Matrix> __global__ void cp_summary(
 const bf16* kp,const float* gp,const bf16* wp,const float* up,const int* bounds,
 const int* meta,float* result,int hv,int nc){
 int h=blockIdx.x/2,part=blockIdx.x%2,seq=blockIdx.y,p=blockIdx.z,t=threadIdx.x;
 long slot=(long(seq)*8+p)*hv+h,descriptor=slot*4;
 int begin=meta[descriptor],end=meta[descriptor+1],warmup=meta[descriptor+2],flag=meta[descriptor+3];
 if(flag>=2||(Matrix&&flag!=1))return;
 if(flag==0)begin=end-warmup;
 int left=bounds[seq];
 __shared__ __align__(16) MatrixShared s;
 float state[2][8];
 #pragma unroll
 for(int i=0;i<2;i++){
  #pragma unroll
  for(int e=0;e<8;e++){int r,c;coord<64>(i,e,r,c);state[i][e]=Matrix&&r==part*64+c?1.f:0.f;}
 }
 for(int chunk=begin;chunk<end;chunk++){
  int ch=chunk_base(left,seq)+chunk;long packed=long(h)*nc+ch,off=packed*8192;
  // Each thread transfers two aligned groups of eight BF16 W elements.
  #pragma unroll
  for(int j=0;j<2;j++){
   int row=(t>>4)+j*32,col=(t&15)*8;
   uint4 value=*reinterpret_cast<const uint4*>(wp+off+row*128+col);
   *reinterpret_cast<uint4*>(s.a+sw(row,col,128))=value;
  }
  #pragma unroll
  for(int i=0;i<2;i++){
   #pragma unroll
   for(int e=0;e<8;e++){int r,c;coord<64>(i,e,r,c);bf16 val=bf(state[i][e]);s.b[sw(r,c,64)]=val;}
  }
  __syncthreads();float pred[1][8];gemm<64,64,128>(s.a,s.b,pred);
  __syncthreads();
  #pragma unroll
  for(int e=0;e<4;e+=2){
   int r,c;coord<64>(0,e,r,c);
   long idx=off+r*128+part*64+c;
   float4 u=make_float4(0.f,0.f,0.f,0.f);
   if(!Matrix)u=*reinterpret_cast<const float4*>(up+off+r*128+u_col4(part*64+c));
   unsigned bits0=pack(bf(u.x-pred[0][e]),bf(u.y-pred[0][e+1]));
   unsigned bits1=pack(bf(u.z-pred[0][e+4]),bf(u.w-pred[0][e+5]));
   *reinterpret_cast<unsigned*>(s.b+sw(r,c,64))=bits0;
   *reinterpret_cast<unsigned*>(s.b+sw(r,c+8,64))=bits1;
  }
  #pragma unroll
  for(int j=0;j<2;++j){
   int r=(t>>4)+j*32,k=(t&15)*8;
   uint4 raw=*reinterpret_cast<const uint4*>(kp+off+r*128+k);
   float d=expf(gp[packed*64+63]-gp[packed*64+r]);
   const bf16* v=reinterpret_cast<const bf16*>(&raw);
   uint4 out;
   out.x=pack(bf(fp(v[0])*d),bf(fp(v[1])*d));
   out.y=pack(bf(fp(v[2])*d),bf(fp(v[3])*d));
   out.z=pack(bf(fp(v[4])*d),bf(fp(v[5])*d));
   out.w=pack(bf(fp(v[6])*d),bf(fp(v[7])*d));
   *reinterpret_cast<uint4*>(s.a+sw(r,k,128))=out;
  }
  __syncthreads();float delta[2][8];gemm_update_natural(s.a,s.b,delta);
  float eg=expf(gp[packed*64+63]);
  #pragma unroll
  for(int i=0;i<2;i++){
   #pragma unroll
   for(int e=0;e<8;e++)state[i][e]=fmaf(eg,state[i][e],delta[i][e]);
  }
  __syncthreads();
 }
 long resultbase=slot*16384;
 #pragma unroll
 for(int i=0;i<2;i++){
  #pragma unroll
  for(int e=0;e<8;e++){int r,c;coord<64>(i,e,r,c);result[resultbase+r*128+part*64+c]=state[i][e];}
 }
}
struct PrefixShared {
 float state[8192];
 float matrix[16384];
};
static_assert(sizeof(PrefixShared)==98304,"CP prefix shared layout");
template<class S> __global__ __launch_bounds__(512,1) void cp_prefix(
 const float* b,const float* m,const int* meta,const S* initial,
 const int* bounds,float* boundary,int hv){
 extern __shared__ __align__(16) unsigned char storage[];
 PrefixShared& s=*reinterpret_cast<PrefixShared*>(storage);
 int h=blockIdx.x/2,part=blockIdx.x%2,seq=blockIdx.y,t=threadIdx.x;
 int n=(bounds[seq+1]-bounds[seq]+63)/64,length,count;
 cp_partition_geometry(n,length,count);
 long initialbase=(long(seq)*hv+h)*16384;
 for(int x=t;x<8192;x+=512){
  s.state[x]=fp(initial[initialbase+(part*64+x%64)*128+x/64]);
 }
 __syncthreads();
 for(int p=0;p<count;++p){
  long slot=(long(seq)*8+p)*hv+h,pos=slot*16384;
  for(int x=t;x<8192;x+=512){
   boundary[pos+(x/64)*128+part*64+x%64]=s.state[x];
  }
  if(p+1==count)break;
  if(meta[slot*4+3]==0){
   for(int x=t;x<8192;x+=512){
    s.state[x]=b[pos+(x/64)*128+part*64+x%64];
   }
   __syncthreads();
   continue;
  }
  for(int x=t;x<16384;x+=512)s.matrix[x]=m[pos+x];
  __syncthreads();
  float next[16];
  #pragma unroll
  for(int z=0;z<16;++z){
   int x=t+512*z,r=x/64,c=x%64;
   float value=b[pos+r*128+part*64+c];
   #pragma unroll 1
   for(int k=0;k<128;++k){
    value=fmaf(s.matrix[r*128+k],s.state[k*64+c],value);
   }
   next[z]=value;
  }
  __syncthreads();
  #pragma unroll
  for(int z=0;z<16;++z)s.state[t+512*z]=next[z];
  __syncthreads();
 }
}

__global__ void cp_replay(
 const bf16* kp,const float* gp,const bf16* wp,const float* up,const float* boundary,
 const int* bounds,const int* meta,bf16* hs,bf16* vn,float* final,int hv,int nc){
 int h=blockIdx.x/2,part=blockIdx.x%2,seq=blockIdx.y,p=blockIdx.z,t=threadIdx.x;
 long slot=(long(seq)*8+p)*hv+h,descriptor=slot*4;
 if(meta[descriptor+3]==3)return;
 int begin=meta[descriptor],end=meta[descriptor+1],left=bounds[seq];
 int n=(bounds[seq+1]-left+63)/64,length,count;
 cp_partition_geometry(n,length,count);
 long statebase=(long(seq)*hv+h)*16384,boundarybase=slot*16384;
 __shared__ __align__(16) MatrixShared s;
 float state[2][8];
 #pragma unroll
 for(int i=0;i<2;i++){
  #pragma unroll
  for(int e=0;e<8;e++){int r,c;coord<64>(i,e,r,c);state[i][e]=boundary[boundarybase+r*128+part*64+c];}
 }
 for(int chunk=begin;chunk<end;chunk++){
  int ch=chunk_base(left,seq)+chunk;long packed=long(h)*nc+ch,off=packed*8192;
  // Each thread transfers two aligned groups of eight BF16 W elements.
  #pragma unroll
  for(int j=0;j<2;j++){
   int row=(t>>4)+j*32,col=(t&15)*8;
   uint4 value=*reinterpret_cast<const uint4*>(wp+off+row*128+col);
   *reinterpret_cast<uint4*>(s.a+sw(row,col,128))=value;
  }
  #pragma unroll
  for(int i=0;i<2;i++){
   #pragma unroll
   for(int e=0;e<8;e++){int r,c;coord<64>(i,e,r,c);bf16 val=bf(state[i][e]);s.b[sw(r,c,64)]=val;hs[packed*16384+r*128+part*64+c]=val;}
  }
  __syncthreads();float pred[1][8];gemm<64,64,128>(s.a,s.b,pred);
  __syncthreads();
  #pragma unroll
  for(int e=0;e<4;e+=2){
   int r,c;coord<64>(0,e,r,c);
   long idx=off+r*128+part*64+c;
   float4 u=*reinterpret_cast<const float4*>(up+off+r*128+u_col4(part*64+c));
   unsigned bits0=pack(bf(u.x-pred[0][e]),bf(u.y-pred[0][e+1]));
   unsigned bits1=pack(bf(u.z-pred[0][e+4]),bf(u.w-pred[0][e+5]));
   *reinterpret_cast<unsigned*>(s.b+sw(r,c,64))=bits0;
   *reinterpret_cast<unsigned*>(vn+idx)=bits0;
   *reinterpret_cast<unsigned*>(s.b+sw(r,c+8,64))=bits1;
   *reinterpret_cast<unsigned*>(vn+idx+8)=bits1;
  }
  #pragma unroll
  for(int j=0;j<2;++j){
   int r=(t>>4)+j*32,k=(t&15)*8;
   uint4 raw=*reinterpret_cast<const uint4*>(kp+off+r*128+k);
   float d=expf(gp[packed*64+63]-gp[packed*64+r]);
   const bf16* v=reinterpret_cast<const bf16*>(&raw);
   uint4 out;
   out.x=pack(bf(fp(v[0])*d),bf(fp(v[1])*d));
   out.y=pack(bf(fp(v[2])*d),bf(fp(v[3])*d));
   out.z=pack(bf(fp(v[4])*d),bf(fp(v[5])*d));
   out.w=pack(bf(fp(v[6])*d),bf(fp(v[7])*d));
   *reinterpret_cast<uint4*>(s.a+sw(r,k,128))=out;
  }
  __syncthreads();float delta[2][8];gemm_update_natural(s.a,s.b,delta);
  float eg=expf(gp[packed*64+63]);
  #pragma unroll
  for(int i=0;i<2;i++){
   #pragma unroll
   for(int e=0;e<8;e++)state[i][e]=fmaf(eg,state[i][e],delta[i][e]);
  }
  __syncthreads();
 }
 if(p+1==count){
  #pragma unroll
  for(int i=0;i<2;i++){
   #pragma unroll
   for(int e=0;e<8;e++){int r,c;coord<64>(i,e,r,c);final[statebase+(part*64+c)*128+r]=state[i][e];}
  }
 }
}

std::vector<torch::Tensor> forward(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor g,torch::Tensor beta,torch::Tensor initial,torch::Tensor bounds){
 TORCH_CHECK(q.is_cuda()&&q.is_contiguous()&&k.is_contiguous()&&v.is_contiguous()&&g.is_contiguous()&&beta.is_contiguous()&&initial.is_contiguous()&&bounds.is_contiguous(),"CUDA contiguous inputs required");
 int total=q.size(1),hq=q.size(2),hv=v.size(2),ns=bounds.numel()-1,nc=(total+63)/64+ns;
 TORCH_CHECK(q.size(0)==1&&q.size(3)==128&&hv%hq==0,"expected packed B1,D128 and grouped heads");
 auto opts=q.options();auto fopts=opts.dtype(torch::kFloat32);
 auto qp=torch::empty({hv,nc,64,128},opts),kp=torch::empty_like(qp),gp=torch::empty({hv,nc,64},fopts),ap=torch::empty({hv,nc,64,64},opts);
 auto wp=torch::empty_like(qp),up=torch::empty({hv,nc,64,128},fopts),pp=torch::empty_like(ap),hs=torch::empty({hv,nc,128,128},opts),vn=torch::empty_like(qp);
 auto out=torch::empty_like(v),last=torch::empty(initial.sizes(),fopts);
 auto stream=at::cuda::getCurrentCUDAStream();dim3 pg(nc,hv);
 #define PREP(GT) prepare<<<pg,128,0,stream>>>((bf16*)q.data_ptr(),(bf16*)k.data_ptr(),(GT*)g.data_ptr(),bounds.data_ptr<int>(),(bf16*)qp.data_ptr(),(bf16*)kp.data_ptr(),gp.data_ptr<float>(),total,hq,hv,ns,nc)
 if(g.scalar_type()==torch::kFloat32){PREP(float);}else{PREP(bf16);}
 #undef PREP
 inverse<<<pg,512,0,stream>>>((bf16*)kp.data_ptr(),(bf16*)beta.data_ptr(),gp.data_ptr<float>(),bounds.data_ptr<int>(),(bf16*)ap.data_ptr(),hv,ns,nc);
 basis<<<pg,512,0,stream>>>((bf16*)qp.data_ptr(),(bf16*)kp.data_ptr(),(bf16*)v.data_ptr(),(bf16*)beta.data_ptr(),gp.data_ptr<float>(),(bf16*)ap.data_ptr(),bounds.data_ptr<int>(),(bf16*)wp.data_ptr(),up.data_ptr<float>(),(bf16*)pp.data_ptr(),hv,ns,nc);
 dim3 sg(hv*2,ns);
 const bool use_cp=hv<=16&&int64_t(total)>=int64_t(131072)*ns;
 if(use_cp){
  auto meta=torch::empty({ns,8,hv,4},opts.dtype(torch::kInt32));
  auto summary_b=torch::empty({ns,8,hv,128,128},fopts),summary_m=torch::empty_like(summary_b),boundary=torch::empty_like(summary_b);
  cp_metadata<<<dim3(8,ns),32,0,stream>>>(gp.data_ptr<float>(),bounds.data_ptr<int>(),meta.data_ptr<int>(),hv,nc);
  dim3 cg(hv*2,ns,8);
  cp_summary<false><<<cg,512,0,stream>>>((bf16*)kp.data_ptr(),gp.data_ptr<float>(),(bf16*)wp.data_ptr(),up.data_ptr<float>(),bounds.data_ptr<int>(),meta.data_ptr<int>(),summary_b.data_ptr<float>(),hv,nc);
  cp_summary<true><<<cg,512,0,stream>>>((bf16*)kp.data_ptr(),gp.data_ptr<float>(),(bf16*)wp.data_ptr(),up.data_ptr<float>(),bounds.data_ptr<int>(),meta.data_ptr<int>(),summary_m.data_ptr<float>(),hv,nc);
  #define CP_PREFIX(ST) do { \
   auto attribute_error=cudaFuncSetAttribute(cp_prefix<ST>,cudaFuncAttributeMaxDynamicSharedMemorySize,98304); \
   TORCH_CHECK(attribute_error==cudaSuccess,cudaGetErrorString(attribute_error)); \
   cp_prefix<ST><<<sg,512,98304,stream>>>(summary_b.data_ptr<float>(),summary_m.data_ptr<float>(),meta.data_ptr<int>(),(ST*)initial.data_ptr(),bounds.data_ptr<int>(),boundary.data_ptr<float>(),hv); \
  } while(0)
  if(initial.scalar_type()==torch::kFloat32){CP_PREFIX(float);}else{CP_PREFIX(bf16);}
  #undef CP_PREFIX
  cp_replay<<<cg,512,0,stream>>>((bf16*)kp.data_ptr(),gp.data_ptr<float>(),(bf16*)wp.data_ptr(),up.data_ptr<float>(),boundary.data_ptr<float>(),bounds.data_ptr<int>(),meta.data_ptr<int>(),(bf16*)hs.data_ptr(),(bf16*)vn.data_ptr(),last.data_ptr<float>(),hv,nc);
 }else{
 #define STATE(ST) state_scan<ST><<<sg,512,0,stream>>>((bf16*)kp.data_ptr(),gp.data_ptr<float>(),(bf16*)wp.data_ptr(),up.data_ptr<float>(),(ST*)initial.data_ptr(),bounds.data_ptr<int>(),(bf16*)hs.data_ptr(),(bf16*)vn.data_ptr(),last.data_ptr<float>(),hv,nc)
 if(initial.scalar_type()==torch::kFloat32){STATE(float);}else{STATE(bf16);}
 #undef STATE
 }

 outputs<<<dim3(nc,hv*2),512,0,stream>>>((bf16*)qp.data_ptr(),gp.data_ptr<float>(),(bf16*)pp.data_ptr(),(bf16*)hs.data_ptr(),(bf16*)vn.data_ptr(),bounds.data_ptr<int>(),(bf16*)out.data_ptr(),hv,ns,nc);
 auto err=cudaGetLastError();TORCH_CHECK(err==cudaSuccess,cudaGetErrorString(err));return {out,last};
}
