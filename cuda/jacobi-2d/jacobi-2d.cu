
#include <iostream>
#include <sys/time.h>
#include <cuda.h>

#include "getopt.h"

// For CUDA printf
#include <stdio.h>

#ifndef TIME_TILE_SIZE
#warning TIME_TILE_SIZE is not set, defaulting to 1
#define TIME_TILE_SIZE 1
#endif

#ifndef FloatType
#define FloatType float
#endif

#if FloatType == float
#define ZERO 0.0f
#define FIVE 5.0f
#else
#define ZERO 0.0
#define FIVE 5.0
#endif

// Command-line options
const option commandLineOptions[] = {
  {"csv", no_argument, 0, 'e'},
  {"print", no_argument, 0, 'p'},
  {"seed", required_argument, 0, 'r'},
  {"size", required_argument, 0, 'n'},
  {"tilex", required_argument, 0, 'x'},
  {"tiley", required_argument, 0, 'y'},
  {"time", required_argument, 0, 't'},
  {"verbose", no_argument, 0, 'v'},
  {"verify", no_argument, 0, 'c'},
  {0, 0, 0, 0}
};


#define CHECK_CALL(err)   checkCudaCall(err, __FILE__, __LINE__)
#define SYNC_DEVICE()     syncDevice(__FILE__, __LINE__)
#define ASSERT_STATE(msg) checkCudaState(msg, __FILE__, __LINE__)

inline void checkCudaCall(cudaError   err,
                          const char *file,
                          const int   line) {
  if(cudaSuccess != err) {
    std::cerr << file << "(" << line << ") :  checkCudaCall failed - " <<
      cudaGetErrorString(err) << std::endl;
    //assert(false && "Cuda error");
    exit(-1);
  }
}

inline void syncDevice(const char *file,
                       const int   line) {
  //cudaError err = cudaDeviceSynchronize();
  //CUresult err  = cuCtxSynchronize();
  //checkCudaCall(err, file, line);

  cudaThreadSynchronize();
}

inline void checkCudaState(const char *errorMessage,
                           const char *file,
                           const int   line) {
  cudaError_t err = cudaGetLastError();
  if(cudaSuccess != err) {
    std::cerr << errorMessage << std::endl;
    checkCudaCall(err, file, line);
  }
}


// Print function
template <typename T>
void printValue(const char* name, T value, bool csv) {
  std::cout << name;
  std::cout << (csv ? "," : ":\t");
  std::cout << value << "\n";
}


// Kernels
__global__
void jacobi_2d_kernel_notimetiling(FloatType* input, FloatType* output,
                                   int32_t problemSize) {
  // Determine our start position
  int offsetI = blockIdx.x * blockDim.x + threadIdx.x;
  int offsetJ = blockIdx.y * blockDim.y + threadIdx.y;

  // For no time tiling, just do everything in global memory
  FloatType c = input[offsetI*problemSize+offsetJ];
  FloatType l = (offsetI > 0) ? input[(offsetI-1)*problemSize+offsetJ] : ZERO;
  FloatType r = (offsetI < (problemSize-1)) ? input[(offsetI+1)*problemSize+offsetJ] : ZERO;
  FloatType t = (offsetJ > 0) ? input[offsetI*problemSize+offsetJ-1] : ZERO;
  FloatType b = (offsetJ < (problemSize-1)) ? input[offsetI*problemSize+offsetJ+1] : ZERO;

  FloatType average = (l + c + r + t + b) / FIVE;

  output[offsetI*problemSize+offsetJ] = average;
}

extern __shared__ FloatType buffer[];

__global__
void jacobi_2d_kernel_overlapped(FloatType* input, FloatType* output,
                                 int32_t problemSize) {
  // Determine out start position
  int baseOffset = blockIdx.x * (blockDim.x-2*(TIME_TILE_SIZE-1)) + threadIdx.x;
  baseOffset -= TIME_TILE_SIZE-1;

  // Load data into shared
  FloatType c = ((baseOffset >= 0) && (baseOffset <= (problemSize-1))) ? input[baseOffset] : ZERO;
  FloatType l = (baseOffset > 0) ? input[baseOffset-1] : ZERO;
  FloatType r = (baseOffset < (problemSize-1)) ? input[baseOffset+1] : ZERO;

  FloatType average = (l + c + r) / FIVE;
  buffer[threadIdx.x] = ((baseOffset >= 0) && (baseOffset <= (problemSize-1))) ? average : buffer[threadIdx.x];

  __syncthreads();

#ifdef DEBUG
  printf("[%d, %d]: Read input at %d (%f)\n", blockIdx.x, threadIdx.x, baseOffset, buffer[threadIdx.x]);
#endif

  // Perform the time iterations
#pragma unroll
  for(int t = 0; t < TIME_TILE_SIZE-1; ++t) {
    FloatType c = buffer[threadIdx.x];
    FloatType l = (threadIdx.x > 0) ? buffer[threadIdx.x-1] : ZERO;
    FloatType r = (threadIdx.x < (blockDim.x-1)) ? buffer[threadIdx.x+1] : ZERO;

    FloatType average = (l + c + r) / FIVE;

#ifdef DEBUG
    if(threadIdx.x == 1 && blockIdx.x == 0) {
      printf("f(%f, %f, %f) = %f\n", l, c, r, average);
    }
#endif

    // Sync before overwriting shared
    __syncthreads();

    buffer[threadIdx.x] = ((baseOffset >= 0) && (baseOffset <= (problemSize-1))) ? average : buffer[threadIdx.x];

    // Sync before re-reading shared
    __syncthreads();
  }

  if(threadIdx.x >= (TIME_TILE_SIZE-1) &&
     threadIdx.x <= (blockDim.x-1-(TIME_TILE_SIZE-1))) {
    output[baseOffset] = buffer[threadIdx.x];
#ifdef DEBUG
    printf("[%d, %d]: Write output at %d\n", blockIdx.x, threadIdx.x, baseOffset);
#endif
  }
}

// Host function
void jacobi_2d_host(FloatType* input, FloatType* output,
                    int32_t problemSize, int32_t timeSteps) {
  FloatType* A = new FloatType[problemSize*problemSize];
  FloatType* B = new FloatType[problemSize*problemSize];

  memcpy(A, input, sizeof(FloatType) * problemSize * problemSize);
  
  for(int t = 0; t < timeSteps; ++t) {
    for(int i = 0; i < problemSize; ++i) {
      for(int j = 0; j < problemSize; ++j) {
        FloatType c = A[i*problemSize+j];
        FloatType l = (i > 0) ? A[(i-1)*problemSize+j] : ZERO;
        FloatType r = (i < (problemSize-1)) ? A[(i+1)*problemSize+j] : ZERO;
        FloatType t = (j > 0) ? A[i*problemSize+j-1] : ZERO;
        FloatType b = (j < (problemSize-1)) ? A[i*problemSize+j+1] : ZERO;

        FloatType average = (l + c + r + t + b) / FIVE;

#ifdef DEBUG
        if(i == 0) {
          printf("host f(%f, %f, %f) = %f\n", l, c, r, average);
        }
#endif
      
        B[i*problemSize+j] = average;
      }
    }
    std::swap(A, B);
  }
  
  for(int i = 0; i < problemSize*problemSize; ++i) {
    output[i] = A[i];
  }

  delete [] A;
  delete [] B;
}

// Comparison function
void compareResults(FloatType* host, FloatType* device, int32_t problemSize,
                    bool csv) {
  double errorNorm, refNorm, diff;
  errorNorm = 0.0;
  refNorm = 0.0;

  for(int i = 0; i < problemSize*problemSize; ++i) {
    diff = host[i] - device[i];
    errorNorm += diff * diff;
    refNorm += host[i] * host[i];
  }

  errorNorm = std::sqrt(errorNorm);
  refNorm = std::sqrt(refNorm);

  printValue("Error Norm", errorNorm, csv);
  printValue("Ref Norm", refNorm, csv);
  
  if(std::abs(refNorm) < 1e-7) {
    printValue("Correctness", "FAILED", csv);
  }
  else if((errorNorm / refNorm) > 1e-2) {
    printValue("Correctness", "FAILED", csv);
  }
  else {
    printValue("Correctness", "PASSED", csv);
  }
}

// Timer function
double rtclock(){
  timeval tp;
  gettimeofday(&tp, NULL);
  return (tp.tv_sec + tp.tv_usec*1.0e-6);
}

// Program entry-point
int main(int argc, char** argv) {

  int32_t problemSize = 512;
  int32_t timeSteps = 64;
  int32_t tileSizeX = 16;
  int32_t tileSizeY = 16;
  int32_t randomSeed = time(NULL);
  bool verbose = false;
  bool verify = false;
  bool print = false;
  bool csv = false;
  
  // Always use device 0
  cudaSetDevice(0);

  // Parse options
  int opt;
  while((opt = getopt_long(argc, argv, "cen:pr:t:vx:y:", commandLineOptions, NULL)) != -1) {
    switch(opt) {
      case 'c':
        verify = true;
        break;
      case 'e':
        csv = true;
        break;
      case 'n':
        problemSize = atoi(optarg);
        break;
      case 'p':
        print = true;
        break;
      case 'r':
        randomSeed = atoi(optarg);
        break;
      case 't':
        timeSteps = atoi(optarg);
        break;
      case 'v':
        verbose = true;
        break;
      case 'x':
        tileSizeX = atoi(optarg);
        break;
      case 'y':
        tileSizeY = atoi(optarg);
        break;
      default:
        std::cerr << "[WARNING] Unknown option: " << opt << "\n";
        break;
    }
  }

  srand(randomSeed);

  // Print out experiment parameters
  if(verbose) {
    printValue("Problem Size", problemSize, csv);
    printValue("Time Steps", timeSteps, csv);
    printValue("Random Seed", randomSeed, csv);
  }

  // Allocate buffers
  FloatType* hostA;
  FloatType* hostB;
  FloatType* deviceA;
  FloatType* deviceB;
  FloatType* refOutput;
  
  // Allocate host buffers
  hostA = new FloatType[problemSize*problemSize];
  hostB = new FloatType[problemSize*problemSize];

  if(verbose) {
    printValue("Buffer Size", (problemSize * problemSize * sizeof(FloatType)), 
               csv);
  }

  // Allocate device buffers
  CHECK_CALL(cudaMalloc((void**)&deviceA, 
             sizeof(FloatType) * problemSize * problemSize));
  CHECK_CALL(cudaMalloc((void**)&deviceB,
             sizeof(FloatType) * problemSize * problemSize));

  // Randomize the input
  for(int i = 0; i < problemSize*problemSize; ++i) {
    hostA[i] = (FloatType)rand() / (FloatType)RAND_MAX;
    hostB[i] = ZERO;
  }

  // Compute reference
  if(verify) {
    refOutput = new FloatType[problemSize*problemSize];
    jacobi_2d_host(hostA, refOutput, problemSize, timeSteps);
  }
  else {
    refOutput = NULL;
  }
  
  // Copy to device
  CHECK_CALL(cudaMemcpy(deviceA, hostA, 
                        sizeof(FloatType) * problemSize * problemSize,
                        cudaMemcpyHostToDevice));
  CHECK_CALL(cudaMemcpy(deviceB, hostB,
                        sizeof(FloatType) * problemSize * problemSize,
                        cudaMemcpyHostToDevice));
  
  // Setup the kernel
  FloatType* input = deviceA;
  FloatType* output = deviceB;
  dim3 grid(problemSize / tileSizeX, problemSize / tileSizeY);
  dim3 block(tileSizeX + 2*(TIME_TILE_SIZE-1), tileSizeY + 2*(TIME_TILE_SIZE-1));

  if(verbose) {
    printValue("Block Size X", block.x, csv);
    printValue("Block Size Y", block.y, csv);
    printValue("Grid Size X", grid.x, csv);
    printValue("Grid Size Y", grid.y, csv);
  }
  
  // Run the kernel
  double startTime = rtclock();
  for(int t = 0; t < timeSteps; t += TIME_TILE_SIZE) {
    const int sharedMemSize = block.x * block.y * sizeof(FloatType);
#if TIME_TILE_SIZE == 1
    jacobi_2d_kernel_notimetiling<<<grid, block>>>(input, output, problemSize);
#else
    jacobi_2d_kernel_overlapped<<<grid, block, sharedMemSize>>>(input, output,
                                                                problemSize);
#endif
    std::swap(input, output);
  }
  SYNC_DEVICE();
  ASSERT_STATE("Kernel");
  double endTime = rtclock();
  double elapsedTime = endTime - startTime;
  
  printValue("Elapsed Time", elapsedTime, csv);

  double flops = problemSize * problemSize * 5.0 * timeSteps;
  double gflops = flops / elapsedTime / 1e9;

  printValue("GFlop/s", gflops, csv);

  CHECK_CALL(cudaMemcpy(hostB, input,
                        sizeof(FloatType) * problemSize * problemSize,
                        cudaMemcpyDeviceToHost));
                          
  if(verify) {
    compareResults(refOutput, hostB, problemSize, csv);
  }

  if(print) {
    std::cout << "GPU -> CPU\n";
    for(int i = 0; i < problemSize*problemSize; ++i) {
      std::cout << hostB[i];
      if(verify) {
        std::cout << " -> " << refOutput[i];
      }
      std::cout << "\n";
    }
  }
  
  // Free buffers
  delete [] hostA;
  delete [] hostB;
  if(refOutput)
    delete [] refOutput;
  CHECK_CALL(cudaFree(deviceA));
  CHECK_CALL(cudaFree(deviceB));
  
  return 0;
}
