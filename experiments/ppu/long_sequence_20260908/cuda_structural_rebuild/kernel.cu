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

#include <c10/cuda/CUDAGuard.h>
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


// Four warps own each logical S/V/O role. One accumulator class is reused.
template<int N> __device__ __forceinline__ void group_coord(int tile,int e,int& r,int& c){
 int local=threadIdx.x%128,lane=local&31,id=(local>>5)+4*tile;
 r=(id/(N/16))*16+(lane>>2)+((e&2)?8:0);
 c=(id%(N/16))*16+(lane&3)*2+(e&1)+((e&4)?8:0);
}
template<int M,int N,int K,bool Clear,bool NaturalTranspose=false>
__device__ __forceinline__ void gemm_group(const bf16* a,const bf16* b,float (&out)[8][8]){
 int local=threadIdx.x%128,lane=local&31,warp=local>>5;
 #pragma unroll
 for(int tile=0;tile<M*N/1024;tile++){
  int id=warp+4*tile,m=(id/(N/16))*16,n=(id%(N/16))*16;
  if(Clear){
   #pragma unroll
   for(int e=0;e<8;e++)out[tile][e]=0.f;
  }
  #pragma unroll
  for(int k=0;k<K;k+=16){
   unsigned ar[4],br[4],pa;
   if(NaturalTranspose){
    pa=(unsigned)__cvta_generic_to_shared(a+sw(k+(lane&7)+(lane>>4)*8,m+((lane>>3)&1)*8,128));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(ar[0]),"=r"(ar[1]),"=r"(ar[2]),"=r"(ar[3]):"r"(pa));
   }else{
    pa=(unsigned)__cvta_generic_to_shared(a+sw(m+(lane&15),k+(lane>>4)*8,K));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(ar[0]),"=r"(ar[1]),"=r"(ar[2]),"=r"(ar[3]):"r"(pa));
   }
   unsigned pb=(unsigned)__cvta_generic_to_shared(b+sw(k+(lane&15),n+(lane>>4)*8,N));
   asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(br[0]),"=r"(br[1]),"=r"(br[2]),"=r"(br[3]):"r"(pb));
   mma16(out[tile],ar,br);
  }
 }
}

struct alignas(16) SvoShared {
 bf16 q[8192],k[8192],xt[8192],h[8192],ab[4096],pg[4096];
 float g[64],gamma[64],reverse[64];
};
struct alignas(16) SummaryShared {
 bf16 h[8192],k[8192],ab[4096],residual[4096],vn[4096];
 float g[64],gamma[64],reverse[64];
};
static_assert(sizeof(SvoShared)==82688,"frozen S/V/O shared allocation");
static_assert(sizeof(SummaryShared)==58112,"frozen S/V summary allocation");

struct SvoParams {
 const bf16 *qp,*kp,*v,*beta,*ap;
 const float *gp,*boundary;
 const int *bounds,*meta;
 bf16* output;
 float* final;
 int hv,nc,use_cp;
};

template<class Initial> __global__ void fused_svo(SvoParams p,const Initial* initial){
 int head=blockIdx.x/2,part=blockIdx.x%2,seq=blockIdx.y,partition=blockIdx.z;
 int role=threadIdx.x/128,local=threadIdx.x%128;
 int left=p.bounds[seq],right=p.bounds[seq+1],n=(right-left+63)/64;
 int begin=0,end=n,count=1;
 long statebase=(long(seq)*p.hv+head)*16384;
 long slot=(long(seq)*8+partition)*p.hv+head;
 if(p.use_cp){
  if(p.meta[slot*4+3]==3)return; // Uniform CTA return before any barrier.
  begin=p.meta[slot*4];end=p.meta[slot*4+1];
  int length;cp_partition_geometry(n,length,count);
 }
 extern __shared__ __align__(16) unsigned char raw_shared[];
 SvoShared& s=*reinterpret_cast<SvoShared*>(raw_shared);
 float accum[8][8];
 if(role==0){
  #pragma unroll
  for(int tile=0;tile<8;tile++){
   #pragma unroll
   for(int e=0;e<8;e++){
    int r,c;group_coord<64>(tile,e,r,c);
    accum[tile][e]=p.use_cp?p.boundary[slot*16384+r*128+part*64+c]
                         :fp(initial[statebase+(part*64+c)*128+r]);
   }
  }
 }
 for(int chunk=begin;chunk<end;chunk++){
  long packed=long(head)*p.nc+chunk_base(left,seq)+chunk,off=packed*8192;
  int token_base=left+chunk*64,valid=min(64,right-token_base);
  // A: producers publish H_before, K/Ab/gates, Q and transposed K.
  if(role==0){
   #pragma unroll
   for(int tile=0;tile<8;tile++){
    #pragma unroll
    for(int e=0;e<8;e++){int r,c;group_coord<64>(tile,e,r,c);s.h[sw(r,c,64)]=bf(accum[tile][e]);}
   }
  }else if(role==1){
   for(int x=local;x<8192;x+=128)s.k[sw(x/128,x%128,128)]=p.kp[off+x];
   for(int x=local;x<4096;x+=128){
    int r=x/64,c=x%64;
    float beta=c<valid?fp(p.beta[long(token_base+c)*p.hv+head]):0.f;
    s.ab[sw(r,c,64)]=bf(r<valid&&c<=r?fp(p.ap[packed*4096+x])*beta:0.f);
   }
   if(local<64){
    float g=p.gp[packed*64+local];s.g[local]=g;s.gamma[local]=expf(g);
    s.reverse[local]=expf(p.gp[packed*64+63]-g);
   }
  }else{
   for(int x=local;x<8192;x+=128){
    int r=x/128,c=x%128;
    s.q[sw(r,c,128)]=p.qp[off+x];s.xt[sw(c,r,64)]=p.kp[off+x];
   }
  }
  __syncthreads(); // 1: all operands ready.
  // B: independent KS and QK.
  if(role==1)gemm_group<64,64,128,true>(s.k,s.h,accum);
  else if(role==2)gemm_group<64,64,128,true>(s.q,s.xt,accum);
  __syncthreads(); // 2: XT is no longer read as K^T.
  // C: decay only S registers; H shared must still hold H_before.
  if(role==0){
   float alpha=s.gamma[63];
   #pragma unroll
   for(int tile=0;tile<8;tile++){
    #pragma unroll
    for(int e=0;e<8;e++)accum[tile][e]*=alpha;
   }
  }else if(role==1){
   #pragma unroll
   for(int tile=0;tile<4;tile++){
    #pragma unroll
    for(int e=0;e<8;e++){
     int r,c;group_coord<64>(tile,e,r,c);
     float v=r<valid?fp(p.v[(long(token_base+r)*p.hv+head)*128+part*64+c]):0.f;
     s.xt[sw(r,c,64)]=bf(r<valid?fmaf(-s.gamma[r],accum[tile][e],v):0.f);
    }
   }
  }else{
   #pragma unroll
   for(int tile=0;tile<4;tile++){
    #pragma unroll
    for(int e=0;e<8;e++){
     int r,c;group_coord<64>(tile,e,r,c);
     s.pg[sw(r,c,64)]=bf(r<valid&&c<=r?(accum[tile][e]*0.08838834764831845f)*expf(s.g[r]-s.g[c]):0.f);
    }
   }
  }
  __syncthreads(); // 3: residual and Pg ready; H_before intact.
  // D: independent Vprime and QS, sharing no output region.
  if(role==1)gemm_group<64,64,64,true>(s.ab,s.xt,accum);
  else if(role==2)gemm_group<64,64,128,true>(s.q,s.h,accum);
  __syncthreads(); // 4: both reads of H_before have completed.
  // E: H's region now holds separate unweighted and reverse-weighted Vprime.
  if(role==1){
   #pragma unroll
   for(int tile=0;tile<4;tile++){
    #pragma unroll
    for(int e=0;e<8;e++){
     int r,c;group_coord<64>(tile,e,r,c);int at=sw(r,c,64);
     s.h[at]=bf(accum[tile][e]);s.h[4096+at]=bf(accum[tile][e]*s.reverse[r]);
    }
   }
  }else if(role==2){
   #pragma unroll
   for(int tile=0;tile<4;tile++){
    #pragma unroll
    for(int e=0;e<8;e++){int r,c;group_coord<64>(tile,e,r,c);accum[tile][e]*=0.08838834764831845f*s.gamma[r];}
   }
  }
  __syncthreads(); // 5: Vd and Vn published.
  // F: independent state update and output. The natural K layout is retained.
  if(role==0)gemm_group<128,64,64,false,true>(s.k,s.h+4096,accum);
  else if(role==2){
   gemm_group<64,64,64,false>(s.pg,s.h,accum);
   #pragma unroll
   for(int tile=0;tile<4;tile++){
    #pragma unroll
    for(int e=0;e<8;e++){
     int r,c;group_coord<64>(tile,e,r,c);
     if(r<valid)p.output[(long(token_base+r)*p.hv+head)*128+part*64+c]=bf(accum[tile][e]);
    }
   }
  }
  __syncthreads(); // 6: every reader completes before next chunk reuses shared.
 }
 if(role==0&&partition+1==count){
  #pragma unroll
  for(int tile=0;tile<8;tile++){
   #pragma unroll
   for(int e=0;e<8;e++){int r,c;group_coord<64>(tile,e,r,c);p.final[statebase+(part*64+c)*128+r]=accum[tile][e];}
  }
 }
}

template<bool Matrix> __global__ void summary_residual(
 const bf16* kp,const bf16* ap,const bf16* v,const bf16* beta,const float* gp,
 const int* bounds,const int* meta,float* result,int hv,int nc){
 int head=blockIdx.x/2,part=blockIdx.x%2,seq=blockIdx.y,partition=blockIdx.z;
 int role=threadIdx.x/128,local=threadIdx.x%128;
 long slot=(long(seq)*8+partition)*hv+head;
 int mode=meta[slot*4+3];
 if(mode>=2||(Matrix&&mode!=1))return;
 int begin=meta[slot*4],end=meta[slot*4+1];
 if(mode==0)begin=end-meta[slot*4+2];
 int left=bounds[seq],right=bounds[seq+1];
 extern __shared__ __align__(16) unsigned char raw_shared[];
 SummaryShared& s=*reinterpret_cast<SummaryShared*>(raw_shared);
 float accum[8][8];
 if(role==0){
  #pragma unroll
  for(int tile=0;tile<8;tile++){
   #pragma unroll
   for(int e=0;e<8;e++){int r,c;group_coord<64>(tile,e,r,c);accum[tile][e]=Matrix&&r==part*64+c?1.f:0.f;}
  }
 }
 for(int chunk=begin;chunk<end;chunk++){
  long packed=long(head)*nc+chunk_base(left,seq)+chunk,off=packed*8192;
  int token_base=left+chunk*64,valid=min(64,right-token_base);
  if(role==0){
   #pragma unroll
   for(int tile=0;tile<8;tile++){
    #pragma unroll
    for(int e=0;e<8;e++){int r,c;group_coord<64>(tile,e,r,c);s.h[sw(r,c,64)]=bf(accum[tile][e]);}
   }
  }else{
   for(int x=local;x<8192;x+=128)s.k[sw(x/128,x%128,128)]=kp[off+x];
   for(int x=local;x<4096;x+=128){
    int r=x/64,c=x%64;float b=c<valid?fp(beta[long(token_base+c)*hv+head]):0.f;
    s.ab[sw(r,c,64)]=bf(r<valid&&c<=r?fp(ap[packed*4096+x])*b:0.f);
   }
   if(local<64){float g=gp[packed*64+local];s.g[local]=g;s.gamma[local]=expf(g);s.reverse[local]=expf(gp[packed*64+63]-g);}
  }
  __syncthreads(); // 1: state and operands published.
  if(role==0){
   float alpha=s.gamma[63];
   #pragma unroll
   for(int tile=0;tile<8;tile++){
    #pragma unroll
    for(int e=0;e<8;e++)accum[tile][e]*=alpha;
   }
  }else{
   gemm_group<64,64,128,true>(s.k,s.h,accum);
   #pragma unroll
   for(int tile=0;tile<4;tile++){
    #pragma unroll
    for(int e=0;e<8;e++){
     int r,c;group_coord<64>(tile,e,r,c);float value=0.f;
     if(!Matrix&&r<valid)value=fp(v[(long(token_base+r)*hv+head)*128+part*64+c]);
     s.residual[sw(r,c,64)]=bf(r<valid?fmaf(-s.gamma[r],accum[tile][e],value):0.f);
    }
   }
  }
  __syncthreads(); // 2: residual is separate from all H/K readers.
  if(role==1){
   gemm_group<64,64,64,true>(s.ab,s.residual,accum);
   #pragma unroll
   for(int tile=0;tile<4;tile++){
    #pragma unroll
    for(int e=0;e<8;e++){int r,c;group_coord<64>(tile,e,r,c);s.vn[sw(r,c,64)]=bf(accum[tile][e]*s.reverse[r]);}
   }
  }
  __syncthreads(); // 3: independently stored Vn ready.
  if(role==0)gemm_group<128,64,64,false,true>(s.k,s.vn,accum);
  __syncthreads(); // 4: complete state update before reuse.
 }
 if(role==0){
  #pragma unroll
  for(int tile=0;tile<8;tile++){
   #pragma unroll
   for(int e=0;e<8;e++){int r,c;group_coord<64>(tile,e,r,c);result[slot*16384+r*128+part*64+c]=accum[tile][e];}
  }
 }
}

inline void check_launch(){auto e=cudaGetLastError();TORCH_CHECK(e==cudaSuccess,cudaGetErrorString(e));}
template<class Initial> void launch_svo(SvoParams p,const Initial* initial,int ns,cudaStream_t stream){
 auto e=cudaFuncSetAttribute(fused_svo<Initial>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(SvoShared));
 TORCH_CHECK(e==cudaSuccess,cudaGetErrorString(e));
 fused_svo<Initial><<<dim3(p.hv*2,ns,p.use_cp?8:1),384,sizeof(SvoShared),stream>>>(p,initial);
 check_launch();
}
std::vector<torch::Tensor> forward(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor g,torch::Tensor beta,torch::Tensor initial,torch::Tensor bounds){
 TORCH_CHECK(q.is_cuda()&&k.is_cuda()&&v.is_cuda()&&g.is_cuda()&&beta.is_cuda()&&initial.is_cuda()&&bounds.is_cuda(),"CUDA inputs required");
 TORCH_CHECK(k.device()==q.device()&&v.device()==q.device()&&g.device()==q.device()&&beta.device()==q.device()&&initial.device()==q.device()&&bounds.device()==q.device(),"all inputs must use the same CUDA device");
 c10::cuda::CUDAGuard device_guard(q.device());
 TORCH_CHECK(q.is_contiguous()&&k.is_contiguous()&&v.is_contiguous()&&g.is_contiguous()&&beta.is_contiguous()&&initial.is_contiguous()&&bounds.is_contiguous(),"contiguous inputs required");
 TORCH_CHECK(q.dim()==4&&k.sizes()==q.sizes()&&v.dim()==4&&q.size(0)==1&&q.size(3)==128&&v.size(0)==1&&v.size(1)==q.size(1)&&v.size(3)==128,"packed B1,D128 required");
 TORCH_CHECK(q.scalar_type()==torch::kBFloat16&&k.scalar_type()==torch::kBFloat16&&v.scalar_type()==torch::kBFloat16&&beta.scalar_type()==torch::kBFloat16,"BF16 q/k/v/beta required");
 TORCH_CHECK((g.scalar_type()==torch::kFloat32||g.scalar_type()==torch::kBFloat16)&&(initial.scalar_type()==torch::kFloat32||initial.scalar_type()==torch::kBFloat16),"FP32/BF16 gate/state required");
 TORCH_CHECK(bounds.scalar_type()==torch::kInt32&&bounds.dim()==1&&bounds.numel()>=2,"int32 packed bounds required");
 TORCH_CHECK(q.size(1)<=2147483584LL&&q.size(2)>0&&v.size(2)%q.size(2)==0,"supported integer dimensions and grouped heads required");
 int total=q.size(1),hq=q.size(2),hv=v.size(2),ns=bounds.numel()-1,nc=(total+63)/64+ns;
 TORCH_CHECK(initial.dim()==4&&initial.size(0)==ns&&initial.size(1)==hv&&initial.size(2)==128&&initial.size(3)==128,"state shape must be [sequence,Hv,V,K]");
 TORCH_CHECK(g.numel()==int64_t(total)*hv&&beta.numel()==int64_t(total)*hv,"gate/beta shape mismatch");
 auto opts=q.options(),fopts=opts.dtype(torch::kFloat32);
 auto qp=torch::empty({hv,nc,64,128},opts),kp=torch::empty_like(qp);
 auto gp=torch::empty({hv,nc,64},fopts),ap=torch::empty({hv,nc,64,64},opts);
 auto out=torch::empty_like(v),last=torch::empty(initial.sizes(),fopts);
 auto stream=at::cuda::getCurrentCUDAStream();dim3 pg(nc,hv),sg(hv*2,ns);
 #define PREP(GT) prepare<<<pg,128,0,stream>>>((bf16*)q.data_ptr(),(bf16*)k.data_ptr(),(GT*)g.data_ptr(),bounds.data_ptr<int>(),(bf16*)qp.data_ptr(),(bf16*)kp.data_ptr(),gp.data_ptr<float>(),total,hq,hv,ns,nc)
 if(g.scalar_type()==torch::kFloat32){PREP(float);}else{PREP(bf16);}
 #undef PREP
 check_launch();
 inverse<<<pg,512,0,stream>>>((bf16*)kp.data_ptr(),(bf16*)beta.data_ptr(),gp.data_ptr<float>(),bounds.data_ptr<int>(),(bf16*)ap.data_ptr(),hv,ns,nc);
 check_launch();
 const bool use_cp=hv<=16&&int64_t(total)>=int64_t(131072)*ns;
 torch::Tensor meta,summary_b,summary_m,boundary;
 if(use_cp){
  meta=torch::empty({ns,8,hv,4},opts.dtype(torch::kInt32));
  summary_b=torch::empty({ns,8,hv,128,128},fopts);summary_m=torch::empty_like(summary_b);boundary=torch::empty_like(summary_b);
  cp_metadata<<<dim3(8,ns),32,0,stream>>>(gp.data_ptr<float>(),bounds.data_ptr<int>(),meta.data_ptr<int>(),hv,nc);
  check_launch();
  auto e0=cudaFuncSetAttribute(summary_residual<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(SummaryShared));
  TORCH_CHECK(e0==cudaSuccess,cudaGetErrorString(e0));
  summary_residual<false><<<dim3(hv*2,ns,8),256,sizeof(SummaryShared),stream>>>((bf16*)kp.data_ptr(),(bf16*)ap.data_ptr(),(bf16*)v.data_ptr(),(bf16*)beta.data_ptr(),gp.data_ptr<float>(),bounds.data_ptr<int>(),meta.data_ptr<int>(),summary_b.data_ptr<float>(),hv,nc);
  check_launch();
  auto e1=cudaFuncSetAttribute(summary_residual<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(SummaryShared));
  TORCH_CHECK(e1==cudaSuccess,cudaGetErrorString(e1));
  summary_residual<true><<<dim3(hv*2,ns,8),256,sizeof(SummaryShared),stream>>>((bf16*)kp.data_ptr(),(bf16*)ap.data_ptr(),(bf16*)v.data_ptr(),(bf16*)beta.data_ptr(),gp.data_ptr<float>(),bounds.data_ptr<int>(),meta.data_ptr<int>(),summary_m.data_ptr<float>(),hv,nc);
  check_launch();
  #define PREFIX(ST) do{auto e=cudaFuncSetAttribute(cp_prefix<ST>,cudaFuncAttributeMaxDynamicSharedMemorySize,98304);TORCH_CHECK(e==cudaSuccess,cudaGetErrorString(e));cp_prefix<ST><<<sg,512,98304,stream>>>(summary_b.data_ptr<float>(),summary_m.data_ptr<float>(),meta.data_ptr<int>(),(ST*)initial.data_ptr(),bounds.data_ptr<int>(),boundary.data_ptr<float>(),hv);}while(0)
  if(initial.scalar_type()==torch::kFloat32){PREFIX(float);}else{PREFIX(bf16);}
  #undef PREFIX
  check_launch();
 }
 SvoParams p{};
 p.qp=(bf16*)qp.data_ptr();p.kp=(bf16*)kp.data_ptr();p.v=(bf16*)v.data_ptr();p.beta=(bf16*)beta.data_ptr();p.ap=(bf16*)ap.data_ptr();
 p.gp=gp.data_ptr<float>();p.boundary=use_cp?boundary.data_ptr<float>():nullptr;
 p.bounds=bounds.data_ptr<int>();p.meta=use_cp?meta.data_ptr<int>():nullptr;
 p.output=(bf16*)out.data_ptr();p.final=last.data_ptr<float>();p.hv=hv;p.nc=nc;p.use_cp=use_cp;
 if(initial.scalar_type()==torch::kFloat32)launch_svo(p,initial.data_ptr<float>(),ns,stream);
 else launch_svo(p,(bf16*)initial.data_ptr(),ns,stream);
 return {out,last};
}
