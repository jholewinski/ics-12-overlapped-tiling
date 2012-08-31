
#include <iostream>
#include <cuda.h>


extern "C"
__global__
void kernel(volatile float *A, volatile float *B) {

  unsigned Idx = blockDim.x*blockIdx.x + threadIdx.x;

  float Temp  = A[Idx+1];
  float Temp1 = A[Idx+2];
  float Temp2 = A[Idx+3];
  if (threadIdx.x > 100000) {
    B[Idx+2]  = Temp + Temp1 + Temp2;
  }
}


int main() {

  float *A;
  float *B;


  std::cout << "Block Size X,Block Size Y,Block Size Z,Elements/Thread,Time Tile Size,Dimensions,Register Usage,Num Blocks X,Num Blocks Y,Num Blocks Z,Time Steps,phase2_global_loads,phase2_shared_loads,compute_per_point,phase3_shared_loads,phase4_global_stores,shared_stores,num_fields,data_size,phase_limit,Elapsed Time,EventElapsed,\n";

  for (unsigned Block = 32; Block < 512+1; Block += 32) {
    for (unsigned Grid  = 30; Grid < 1000+1; Grid += 30) {

      cudaMalloc(&A, sizeof(float)*4096*4096);
      cudaMalloc(&B, sizeof(float)*4096*4096);

  
      cudaEvent_t Start, Stop;

      cudaEventCreate(&Start);
      cudaEventCreate(&Stop);

      cudaEventRecord(Start, 0);
      kernel<<<Grid, Block>>>(A, B);
      cudaEventRecord(Stop, 0);

      cudaEventSynchronize(Stop);

      float Elapsed;
      cudaEventElapsedTime(&Elapsed, Start, Stop);
      Elapsed /= 1e3;

      cudaEventDestroy(Start);
      cudaEventDestroy(Stop);


      cudaDeviceProp DeviceProp;
      cudaGetDeviceProperties(&DeviceProp, 0);

      double Clock = DeviceProp.clockRate * 1e3;


 
      cudaFuncAttributes FuncAttrs;
      memset(&FuncAttrs, 0, sizeof(cudaFuncAttributes));
      cudaFuncGetAttributes(&FuncAttrs, "kernel");

 
      cudaFree(A);
      cudaFree(B);

      std::cout << Block << ",1,1,1,1,1," << FuncAttrs.numRegs << "," << Grid << ",1,1,1,1,0,0,0,1,0,1,4,0," << Elapsed << "," << Elapsed << ",\n";

    }
  }
  
  return 0;
}
