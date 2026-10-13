#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(x) do {cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA error at %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
#define BLAS_CHECK(x) do {cublasStatus_t e=(x);if(e!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS error at %s:%d: %d\n",__FILE__,__LINE__,int(e));exit(1);}}while(0)

// Row-major C = A x B. Timing excludes allocations and host transfers.
__global__ void naive_sgemm(const float* A,const float* B,float* C,int N){
 int row=blockIdx.y*blockDim.y+threadIdx.y,col=blockIdx.x*blockDim.x+threadIdx.x;
 if(row>=N||col>=N)return;
 float acc=0.0f;
 for(int k=0;k<N;k++) acc=fmaf(A[row*N+k],B[k*N+col],acc);
 C[row*N+col]=acc;
}

template<int BM,int BK=16>
__global__ void register_tiled_sgemm(const float* __restrict__ A,const float* __restrict__ B,float* __restrict__ C,int N){
 static_assert(BM==32||BM==64,"Supported output tiles: 32 or 64");
 constexpr int THREADS=256, TM=BM/16;
 __shared__ float As[BM][BK+1];
 __shared__ float Bs[BK][BM+1];
 int tid=threadIdx.y*16+threadIdx.x;
 int row0=blockIdx.y*BM+threadIdx.y*TM;
 int col0=blockIdx.x*BM+threadIdx.x*TM;
 float acc[TM][TM]={};
 for(int kb=0;kb<N;kb+=BK){
   for(int idx=tid;idx<BM*BK;idx+=THREADS){
     int r=idx/BK,k=idx%BK;
     int gr=blockIdx.y*BM+r,gk=kb+k;
     As[r][k]=(gr<N&&gk<N)?A[gr*N+gk]:0.0f;
   }
   for(int idx=tid;idx<BK*BM;idx+=THREADS){
     int k=idx/BM,c=idx%BM;
     int gk=kb+k,gc=blockIdx.x*BM+c;
     Bs[k][c]=(gk<N&&gc<N)?B[gk*N+gc]:0.0f;
   }
   __syncthreads();
   #pragma unroll
   for(int k=0;k<BK;k++){
     float a[TM],b[TM];
     #pragma unroll
     for(int i=0;i<TM;i++)a[i]=As[threadIdx.y*TM+i][k];
     #pragma unroll
     for(int j=0;j<TM;j++)b[j]=Bs[k][threadIdx.x*TM+j];
     #pragma unroll
     for(int i=0;i<TM;i++){
       #pragma unroll
       for(int j=0;j<TM;j++)acc[i][j]=fmaf(a[i],b[j],acc[i][j]);
     }
   }
   __syncthreads();
 }
 #pragma unroll
 for(int i=0;i<TM;i++){
   #pragma unroll
   for(int j=0;j<TM;j++){
     int row=row0+i,col=col0+j;
     if(row<N&&col<N)C[row*N+col]=acc[i][j];
   }
 }
}

__global__ void reduce_shared(const float* in,float* out,int n){
 extern __shared__ float s[];
 int t=threadIdx.x,i=blockIdx.x*(blockDim.x*2)+t;
 float x=(i<n?in[i]:0.0f)+(i+blockDim.x<n?in[i+blockDim.x]:0.0f);
 s[t]=x;__syncthreads();
 for(int step=blockDim.x/2;step>0;step>>=1){if(t<step)s[t]+=s[t+step];__syncthreads();}
 if(t==0)out[blockIdx.x]=s[0];
}
__global__ void reduce_shuffle(const float* in,float* out,int n){
 __shared__ float warp[8];
 int t=threadIdx.x,i=blockIdx.x*(blockDim.x*2)+t;
 float x=(i<n?in[i]:0.0f)+(i+blockDim.x<n?in[i+blockDim.x]:0.0f);
 for(int offset=16;offset>0;offset/=2)x+=__shfl_down_sync(0xffffffff,x,offset);
 if((t&31)==0)warp[t/32]=x;
 __syncthreads();
 if(t<32){x=t<8?warp[t]:0.0f;
   for(int offset=16;offset>0;offset/=2)x+=__shfl_down_sync(0xffffffff,x,offset);
   if(t==0)out[blockIdx.x]=x;
 }
}

float median_ms(const std::function<void()>& fn,int repeats=7,int launches=10){
 cudaEvent_t start,stop;CUDA_CHECK(cudaEventCreate(&start));CUDA_CHECK(cudaEventCreate(&stop));
 for(int i=0;i<3;i++)fn();CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
 std::vector<float> timings;
 for(int r=0;r<repeats;r++){
   CUDA_CHECK(cudaEventRecord(start));
   for(int j=0;j<launches;j++)fn();
   CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaEventRecord(stop));CUDA_CHECK(cudaEventSynchronize(stop));
   float ms;CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));timings.push_back(ms/launches);
 }
 std::sort(timings.begin(),timings.end());CUDA_CHECK(cudaEventDestroy(start));CUDA_CHECK(cudaEventDestroy(stop));
 return timings[timings.size()/2];
}

bool validate(const std::vector<float>& got,const std::vector<float>& ref,float& max_abs,float& max_rel){
 max_abs=0;max_rel=0;int errors=0;
 for(size_t i=0;i<ref.size();i++){
   if(!std::isfinite(got[i])){errors++;continue;}
   float diff=std::fabs(got[i]-ref[i]);
   float rel=diff/std::max(1.0f,std::fabs(ref[i]));
   max_abs=std::max(diff,max_abs);max_rel=std::max(rel,max_rel);
   if(diff>0.002f&&rel>0.0005f)errors++;
 }
 return errors==0;
}

int main(){
 cudaDeviceProp prop{};int dev;CUDA_CHECK(cudaGetDevice(&dev));CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
 printf("GPU: %s, compute capability %d.%d, SMs %d\n",prop.name,prop.major,prop.minor,prop.multiProcessorCount);
 if(prop.major<5){fprintf(stderr,"This notebook requires a supported modern CUDA GPU.\n");return 1;}
 std::ofstream csv("/kaggle/working/sgemm_v2_results.csv");if(!csv)csv.open("sgemm_v2_results.csv");
 if(!csv){fprintf(stderr,"Cannot write SGEMM CSV\n");return 1;}
 csv<<"N,kernel,median_ms,gflops,ratio_to_cublas,correct,max_abs,max_rel\n";
 cublasHandle_t handle;BLAS_CHECK(cublasCreate(&handle));
 // Avoid implicit reduced precision for comparisons where this cuBLAS math mode is supported.
 BLAS_CHECK(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));
 bool all_ok=true;
 std::mt19937 rng(2026);std::uniform_real_distribution<float> dist(-0.5f,0.5f);
 for(int N:{127,256,512,1024}){
   const size_t count=size_t(N)*N,bytes=count*sizeof(float);
   std::vector<float> a(count),b(count),ref(count),out(count);
   for(size_t i=0;i<count;i++){a[i]=dist(rng);b[i]=dist(rng);}
   float *da,*db,*dc;CUDA_CHECK(cudaMalloc(&da,bytes));CUDA_CHECK(cudaMalloc(&db,bytes));CUDA_CHECK(cudaMalloc(&dc,bytes));
   CUDA_CHECK(cudaMemcpy(da,a.data(),bytes,cudaMemcpyHostToDevice));
   CUDA_CHECK(cudaMemcpy(db,b.data(),bytes,cudaMemcpyHostToDevice));
   const float alpha=1.f,beta=0.f;
   // cuBLAS uses column-major layout: swap A/B to get row-major A*B.
   auto blas=[&](){BLAS_CHECK(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&alpha,db,N,da,N,&beta,dc,N));};
   float blas_ms=median_ms(blas);
   CUDA_CHECK(cudaMemcpy(ref.data(),dc,bytes,cudaMemcpyDeviceToHost));
   printf("\nN=%d | cuBLAS (pedantic FP32 math): %.5f ms\n",N,blas_ms);
   auto run=[&](const char* name,const std::function<void()>& launch){
     launch();CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
     CUDA_CHECK(cudaMemcpy(out.data(),dc,bytes,cudaMemcpyDeviceToHost));
     float abs_err,rel_err;bool ok=validate(out,ref,abs_err,rel_err);all_ok&=ok;
     float ms=median_ms(launch);
     double gflops=2.0*N*N*N/(ms*1e6),ratio=blas_ms/ms;
     printf(" %-12s %s median %.5f ms %9.1f GFLOPS %5.2fx cuBLAS | abs %.5g rel %.5g\n",name,ok?"PASS":"FAIL",ms,gflops,ratio,abs_err,rel_err);
     csv<<N<<','<<name<<','<<ms<<','<<gflops<<','<<ratio<<','<<(ok?1:0)<<','<<abs_err<<','<<rel_err<<'\n';
   };
   dim3 block(16,16),naive_grid((N+15)/16,(N+15)/16);
   run("naive",[&](){naive_sgemm<<<naive_grid,block>>>(da,db,dc,N);});
   dim3 grid32((N+31)/32,(N+31)/32),grid64((N+63)/64,(N+63)/64);
   run("tile32_reg2",[&](){register_tiled_sgemm<32><<<grid32,block>>>(da,db,dc,N);});
   run("tile64_reg4",[&](){register_tiled_sgemm<64><<<grid64,block>>>(da,db,dc,N);});
   double cugflops=2.0*N*N*N/(blas_ms*1e6);
   csv<<N<<",cublas,"<<blas_ms<<','<<cugflops<<",1,1,0,0\n";
   CUDA_CHECK(cudaFree(da));CUDA_CHECK(cudaFree(db));CUDA_CHECK(cudaFree(dc));
 }
 BLAS_CHECK(cublasDestroy(handle));csv.close();
 std::ofstream rcsv("/kaggle/working/reduction_v2_results.csv");if(!rcsv)rcsv.open("reduction_v2_results.csv");
 if(!rcsv){fprintf(stderr,"Cannot write reduction CSV\n");return 1;}
 rcsv<<"N,kernel,median_ms,result,reference,rel_error,correct\n";
 printf("\nTWO-STAGE REDUCTION: timings include both stages; transfers excluded\n");
 constexpr int THREADS=256;
 for(int N:{(1<<20),(1<<22)}){
   std::vector<float> input(N);double reference=0;
   for(int i=0;i<N;i++){input[i]=(i%13+1)/13.0f;reference+=input[i];}
   int blocks=(N+2*THREADS-1)/(2*THREADS);
   float *di,*p1,*p2;CUDA_CHECK(cudaMalloc(&di,size_t(N)*sizeof(float)));
   CUDA_CHECK(cudaMalloc(&p1,blocks*sizeof(float)));CUDA_CHECK(cudaMalloc(&p2,blocks*sizeof(float)));
   CUDA_CHECK(cudaMemcpy(di,input.data(),size_t(N)*sizeof(float),cudaMemcpyHostToDevice));
   auto run=[&](const char* name,bool use_shuffle){
      // Reduce repeatedly until exactly one value remains.
      auto full=[&](){
        int n=N;const float* src=di;float* dst=p1;
        while(n>1){
          int nb=(n+2*THREADS-1)/(2*THREADS);
          if(use_shuffle)reduce_shuffle<<<nb,THREADS>>>(src,dst,n);
          else reduce_shared<<<nb,THREADS,THREADS*sizeof(float)>>>(src,dst,n);
          n=nb;src=dst;dst=(dst==p1?p2:p1);
        }
      };
      full();CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
      // The final buffer is determined by the number of reduction passes.
      int n=N;float* last=nullptr;float* next=p1;
      while(n>1){int nb=(n+2*THREADS-1)/(2*THREADS);last=next;next=(next==p1?p2:p1);n=nb;}
      float got;CUDA_CHECK(cudaMemcpy(&got,last,sizeof(float),cudaMemcpyDeviceToHost));
      double rel=std::fabs(double(got)-reference)/std::max(1.0,std::fabs(reference));
      bool ok=rel<1e-4;all_ok&=ok;
      float ms=median_ms(full);
      printf(" N=%d %-8s %s median %.5f ms relerr %.3g\n",N,name,ok?"PASS":"FAIL",ms,rel);
      rcsv<<N<<','<<name<<','<<ms<<','<<got<<','<<reference<<','<<rel<<','<<(ok?1:0)<<'\n';
   };
   run("shared",false);run("shuffle",true);
   CUDA_CHECK(cudaFree(di));CUDA_CHECK(cudaFree(p1));CUDA_CHECK(cudaFree(p2));
 }
 printf("\nALL CORRECTNESS CHECKS: %s\n",all_ok?"PASS":"FAIL");
 return all_ok?0:2;
}
