
#include "CLCommon.hpp"
#include "CLContext.hpp"
#include "ProgramGenerator.hpp"

#include <cmath>
#include <iomanip>
#include <fstream>

#include <boost/math/common_factor.hpp>
#include <boost/program_options.hpp>

using namespace ot;

namespace po = boost::program_options;

/**
 * Structure to hold generator options.
 */
struct GeneratorParams {
  // Specified
  int32_t     timeTileSize;
  int32_t     timeSteps;
  int32_t     elementsPerThread;
  int32_t     blockSizeX;
  int32_t     blockSizeY;
  int32_t     blockSizeZ;
  int32_t     problemSize;
  std::string dataType;
  
  // Derived
  int32_t     padding;
  int32_t     compsPerBlockX;
  int32_t     compsPerBlockY;
  int32_t     compsPerBlockZ;
  int32_t     sizeLCM;
  int32_t     realSize;
  int32_t     paddedSize;
  int32_t     realPerBlockX;
  int32_t     realPerBlockY;
  int32_t     realPerBlockZ;
  int32_t     sharedSizeX;
  int32_t     sharedSizeY;
  int32_t     sharedSizeZ;
  int32_t     numBlocksX;
  int32_t     numBlocksY;
  int32_t     numBlocksZ;
  std::string fpSuffix;
  
  

  /**
   * Default constructor.
   */
  GeneratorParams(int32_t tts      = 1,
                  int32_t ept      = 1,
                  int32_t bsx      = 8,
                  int32_t bsy      = 8,
                  int32_t bsz      = 8,
                  int32_t ps       = 128,
                  int32_t ts       = 64,
                  std::string type = "float")
    : timeTileSize(tts),
      timeSteps(ts),
      elementsPerThread(ept),
      problemSize(ps),
      dataType(type),
      blockSizeX(bsx),
      blockSizeY(bsy),
      blockSizeZ(bsz) {
  }

  void computeDerived() {
    // Compute derived values
    padding        = timeTileSize + 1;
    compsPerBlockX = blockSizeX;
    compsPerBlockY = blockSizeY*elementsPerThread;
    compsPerBlockZ = blockSizeZ;
    realPerBlockX  = compsPerBlockX - 2*timeTileSize;
    realPerBlockY  = compsPerBlockY - 2*timeTileSize;
    realPerBlockZ  = compsPerBlockZ - 2*timeTileSize;
    sizeLCM        = boost::math::lcm(realPerBlockZ,
                                      boost::math::lcm(realPerBlockX,
                                                       realPerBlockY));
    realSize       = (problemSize / sizeLCM) * sizeLCM;
    numBlocksX     = realSize / realPerBlockX;
    numBlocksY     = realSize / realPerBlockY;
    numBlocksZ     = realSize / realPerBlockZ;
    sharedSizeX    = blockSizeX + 2;
    sharedSizeY    = blockSizeY * elementsPerThread + 2;
    sharedSizeZ    = blockSizeZ + 2;
    paddedSize     = realSize + 2*padding;

    if(dataType == "float") {
      fpSuffix = "f";
    } else {
      fpSuffix = "";
    }

    if(padding < 1 || compsPerBlockX < 1 || compsPerBlockY < 1     ||
       compsPerBlockZ < 1                                          ||
       realPerBlockX < 1 || realPerBlockY < 1 || realPerBlockZ < 1 ||
       sizeLCM < 1 || realSize < 1                                 ||
       numBlocksX < 1 || numBlocksY < 1 || numBlocksZ < 1          ||
       sharedSizeX < 1    || sharedSizeY < 1 || sharedSizeZ < 1    ||
       paddedSize < 1) {
      throw std::runtime_error("Consistency error!");
    }
  }
};

/**
 * Generator for Jacobi 2D.
 */
class Jacobi3DGenerator : public ProgramGenerator {
public:

  Jacobi3DGenerator();

  virtual ~Jacobi3DGenerator();

  std::string generate(GeneratorParams& params);

private:

  void generateHeader(std::ostream&          stream,
                      const GeneratorParams& params);
  
  void generateFooter(std::ostream& stream);
  
  void generateLocals(std::ostream&          stream,
                      const GeneratorParams& params);

  void generateCompute(std::ostream&          stream,
                       const GeneratorParams& params);
};


Jacobi3DGenerator::Jacobi3DGenerator() {
}

Jacobi3DGenerator::~Jacobi3DGenerator() {
}

std::string Jacobi3DGenerator::generate(GeneratorParams& params) {
  std::stringstream program;

  params.computeDerived();
  
  generateHeader(program, params);
  generateLocals(program, params);
  generateCompute(program, params);
  generateFooter(program);

  return program.str();
}

void Jacobi3DGenerator::generateHeader(std::ostream& stream,
                                       const GeneratorParams& params) {
  stream << "/* Auto-generated.  Do not edit by hand. */\n";
  stream << "__kernel\n";
  stream << "void kernel_func(__global " << params.dataType << "* input,\n";
  stream << "                 __global " << params.dataType << "* output) {\n";
}

void Jacobi3DGenerator::generateFooter(std::ostream& stream) {
  stream << "}\n\n";
}

void Jacobi3DGenerator::generateLocals(std::ostream& stream,
                                       const GeneratorParams& params) {
  stream << "  __local " << params.dataType << " buffer[" << params.sharedSizeZ
         << "][" << params.sharedSizeY << "][" << params.sharedSizeZ << "];\n";

  // Compute some pointer values
  stream << "  __global " << params.dataType
         << "* inputPtr = input + ((get_group_id(2)*" << params.realPerBlockZ
         << "+get_local_id(2)+1)*" << params.paddedSize*params.paddedSize << ")"
         << " + ((get_group_id(1)*" << params.realPerBlockY
         << "+get_local_id(1)*" << params.elementsPerThread << "+1)*"
         << params.paddedSize << ") + (get_group_id(0)*" << params.realPerBlockX
         << ") + get_local_id(0) + 1;\n";

  stream << "  __global " << params.dataType
         << "* outputPtr = output + ((get_group_id(2)*" << params.realPerBlockZ
         << "+get_local_id(2)+1)*" << params.paddedSize*params.paddedSize << ")"
         << " + ((get_group_id(1)*" << params.realPerBlockY
         << "+get_local_id(1)*" << params.elementsPerThread << "+1)*"
         << params.paddedSize << ") + (get_group_id(0)*" << params.realPerBlockX
         << ") + get_local_id(0) + 1;\n";

  // Compute some guards
  stream << "  int globalIndexX = (get_group_id(0)*" << params.realPerBlockX
         << ") + get_local_id(0) + 1;\n";
  stream << "  int globalIndexY;\n";
  stream << "  bool validX = globalIndexX >= " << params.padding
         << " && globalIndexX < " << (params.realSize+params.padding) << ";\n";
  stream << "  int globalIndexZ = (get_group_id(2)*" << params.realPerBlockZ
         << ") + get_local_id(2) + 1;\n";
  stream << "  bool validZ = globalIndexZ >= " << params.padding
         << " && globalIndexZ < " << (params.realSize+params.padding) << ";\n";
  
  
  for(int32_t i = 0; i < params.elementsPerThread; ++i) {
    stream << "  globalIndexY = get_group_id(1)*" << params.realPerBlockY
           << " + " << params.elementsPerThread << "*get_local_id(1) + " << i
           << " + 1;\n";
    stream << "  bool valid" << i << " = validX && validZ && globalIndexY >= "
           << params.padding << " && globalIndexY < "
           << (params.realSize+params.padding) << ";\n";
  }

  stream << "  bool writeValidX = get_local_id(0) >= " << params.timeTileSize
         << " && get_local_id(0) < "
         << (params.realPerBlockX+params.timeTileSize) << ";\n";
  stream << "  int effectiveTidY;\n";
  
  for(int32_t i = 0; i < params.elementsPerThread; ++i) {
    stream << "  effectiveTidY = get_local_id(1)*" << params.elementsPerThread
           << " + " << i << ";\n";
    stream << "  bool writeValid" << i << " = effectiveTidY >= "
           << params.timeTileSize << " && effectiveTidY < "
           << (params.realPerBlockY+params.timeTileSize) << ";\n";
  }
  stream << "  bool writeValidZ = get_local_id(2) >= " << params.timeTileSize
         << " && get_local_id(2) < "
         << (params.realPerBlockZ+params.timeTileSize) << ";\n";
  

  // Declare local intermediates
  for(int32_t i = 0; i < params.elementsPerThread; ++i) {
    stream << "  " << params.dataType << " local" << i << ";\n";
    stream << "  " << params.dataType << " new" << i << ";\n";
  }
}

void Jacobi3DGenerator::generateCompute(std::ostream& stream,
                                        const GeneratorParams& params) {

  for(int32_t i = 0; i < params.elementsPerThread; ++i) {
    stream << "  {\n";
    stream << "    " << params.dataType
           << " val0, val1, val2, val3, val4, val5, val6;\n";
    stream << "    // Left\n";
    stream << "    val0 = *(inputPtr+(" << params.paddedSize << "*" << i
           << ")-1);\n";
    stream << "    // Center\n";
    stream << "    val1 = *(inputPtr+(" << params.paddedSize << "*" << i
           << "));\n";
    stream << "    // Right\n";
    stream << "    val2 = *(inputPtr+(" << params.paddedSize << "*" << i
           << ")+1);\n";
    stream << "    // Top\n";
    stream << "    val3 = *(inputPtr+(" << params.paddedSize << "*" << (i-1)
           << "));\n";
    stream << "    // Bottom\n";
    stream << "    val4 = *(inputPtr+(" << params.paddedSize << "*" << (i+1)
           << "));\n";
    stream << "    // Backward\n";
    stream << "    val5 = *(inputPtr+(" << params.paddedSize << "*" << i
           << ")+(" << params.paddedSize*params.paddedSize << "*-1));\n";
    stream << "    // Forward\n";
    stream << "    val6 = *(inputPtr+(" << params.paddedSize << "*" << i
           << ")+(" << params.paddedSize*params.paddedSize << "*1));";
    stream << "    " << params.dataType
           << " result = 0.143" << params.fpSuffix
           << " * (val0+val1+val2+val3+val4+val5+val6);\n";
    stream << "    result = (valid" << i << ") ? result : 0.0"
           << params.fpSuffix << ";\n";
    stream << "    buffer[get_local_id(2)+1][get_local_id(1)*"
           << params.elementsPerThread << "+" << i
           << "+1][get_local_id(0)+1] = result;\n";
    stream << "    local" << i << " = result;\n";
    stream << "  }\n";
  }

  stream << "  barrier(CLK_LOCAL_MEM_FENCE);\n";

  for(int32_t t = 1; t < params.timeTileSize; ++t) {
    stream << "  // Time Step " << t << "\n";
    for(int32_t i = 0; i < params.elementsPerThread; ++i) {
      stream << "  {\n";
      stream << "    " << params.dataType
             << " val0, val1, val2, val3, val4, val5, val6;\n";
      stream << "    // Left\n";
      stream << "    val0 = buffer[get_local_id(2)+1][get_local_id(1)*"
             << params.elementsPerThread
             << "+" << i
             << "+1][get_local_id(0)];\n";
      stream << "    // Center\n";
      stream << "    val1 = local" << i << ";\n";
      stream << "    // Right\n";
      stream << "    val2 = buffer[get_local_id(2)+1][get_local_id(1)*"
             << params.elementsPerThread
             << "+" << i
             << "+1][get_local_id(0)+2];\n";
      stream << "    // Top\n";
      stream << "    val3 = buffer[get_local_id(2)+1][get_local_id(1)*"
             << params.elementsPerThread
             << "+" << i
             << "][get_local_id(0)+1];\n";
      stream << "    // Bottom\n";
      stream << "    val4 = buffer[get_local_id(2)+1][get_local_id(1)*"
             << params.elementsPerThread
             << "+" << i
             << "+2][get_local_id(0)+1];\n";
      stream << "    // Backwards\n";
      stream << "    val5 = buffer[get_local_id(2)][get_local_id(1)*"
             << params.elementsPerThread
             << "+" << i
             << "+1][get_local_id(0)+1];\n";
      stream << "    // Forwards\n";
      stream << "    val4 = buffer[get_local_id(2)+2][get_local_id(1)*"
             << params.elementsPerThread
             << "+" << i
             << "+1][get_local_id(0)+1];\n";
      stream << "    " << params.dataType
             << " result = 0.143" << params.fpSuffix
             << " * (val0+val1+val2+val3+val4);\n";
      stream << "    result = (valid" << i << ") ? result : 0.0"
             << params.fpSuffix << ";\n";
      stream << "    new" << i << " = result;\n";
      stream << "  }\n";
    }
    stream << "  barrier(CLK_LOCAL_MEM_FENCE);\n";
    for(int32_t i = 0; i < params.elementsPerThread; ++i) {
      stream << "  buffer[get_local_id(2)+1][get_local_id(1)*"
             << params.elementsPerThread << "+"
             << i
             << "+1][get_local_id(0)+1] = new" << i << ";\n";
      stream << "  local" << i << " = new" << i << ";\n";
    }
    stream << "  barrier(CLK_LOCAL_MEM_FENCE);\n";
  }
  for(int32_t i = 0; i < params.elementsPerThread; ++i) {
    stream << "  if(writeValid" << i << " && writeValidX) {\n";
    stream << "    *(outputPtr+(" << params.paddedSize << "*" << i
           << ")) = local" << i << ";\n";
    stream << "  }\n";
  }
}


void compareResults(float* host, float* device, const GeneratorParams& params) {
  
  double errorNorm, refNorm, diff;
  errorNorm = 0.0;
  refNorm   = 0.0;

  for(int i = params.padding; i < params.paddedSize-params.padding; ++i) {
    for(int j = params.padding; j < params.paddedSize-params.padding; ++j) {
      
      float h = host[i*params.paddedSize + j];
      float d = device[i*params.paddedSize + j];
      
      diff       = h - d;
      //      std::cout << "h: " << h << "  d: " << d << "  diff: " << diff << "\n";
      errorNorm += diff*diff;
      refNorm   += h*h;
    }
  }
  
  errorNorm = std::sqrt(errorNorm);
  refNorm = std::sqrt(refNorm);

  printValue("Error Norm", errorNorm);
  printValue("Ref Norm", refNorm);
  
  if(std::abs(refNorm) < 1e-7) {
    printValue("Correctness", "FAILED");
  }
  else if((errorNorm / refNorm) > 1e-2) {
    printValue("Correctness", "FAILED");
  }
  else {
    printValue("Correctness", "PASSED");
  }
}

int main(int argc,
         char** argv) {

  cl_int      result;
  std::string kernelFile;
  std::string saveKernelFile;
  
  srand(123456);
 
  Jacobi3DGenerator gen;
  GeneratorParams   params;

  po::options_description desc("Options");
  desc.add_options()
    ("help,h", "Show usage information")
    ("problem-size,n",
     po::value<int32_t>(&params.problemSize)->default_value(128),
     "Set problem size")
    ("time-steps,t",
     po::value<int32_t>(&params.timeSteps)->default_value(64),
     "Set number of time steps")
    ("block-size-x,x",
     po::value<int32_t>(&params.blockSizeX)->default_value(8),
     "Set block size (X)")
    ("block-size-y,y",
     po::value<int32_t>(&params.blockSizeY)->default_value(8),
     "Set block size (Y)")
    ("block-size-z,z",
     po::value<int32_t>(&params.blockSizeZ)->default_value(8),
     "Set block size (Y)")
    ("elements-per-thread,e",
     po::value<int32_t>(&params.elementsPerThread)->default_value(1),
     "Set elements per thread")
    ("time-tile-size,s",
     po::value<int32_t>(&params.timeTileSize)->default_value(1),
     "Set time tile size")
    ("load-kernel,f",
     po::value<std::string>(&kernelFile)->default_value(""),
     "Load kernel from disk")
    ("save-kernel,w",
     po::value<std::string>(&saveKernelFile)->default_value(""),
     "Save kernel to disk")
    ("verify,v", "Verify results")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if(vm.count("help")) {
    std::cerr << desc;
    return 1;
  }

  std::string kernelSource;
  
  if(kernelFile.size() == 0) {
    kernelSource = gen.generate(params);
  } else {
    std::ifstream kernelStream(kernelFile.c_str());
    kernelSource = std::string(std::istreambuf_iterator<char>(kernelStream),
                               (std::istreambuf_iterator<char>()));
    kernelStream.close();
    params.computeDerived();
  }

  if(saveKernelFile.size() != 0) {
    std::ofstream kernelStream(saveKernelFile.c_str());
    kernelStream << kernelSource;
    kernelStream.close();
  }

  printValue("Problem Size", params.problemSize);
  printValue("Time Tile Size", params.timeTileSize);
  printValue("Padded Size", params.paddedSize);
  printValue("Block Size X", params.blockSizeX);
  printValue("Block Size Y", params.blockSizeY);
  printValue("Block Size Z", params.blockSizeZ);
  printValue("Elements/Thread", params.elementsPerThread);
  printValue("Num Blocks X", params.numBlocksX);
  printValue("Num Blocks Y", params.numBlocksY);
  printValue("Num Blocks Z", params.numBlocksZ);
  printValue("Time Steps", params.timeSteps);
  printValue("Padding", params.padding);
  printValue("Real Size", params.realSize);
  
  int arraySize = params.paddedSize * params.paddedSize * params.paddedSize
    * sizeof(float);

  CLContext context;

  // Collect device information.
  size_t globalMemorySize = context.device()
    .getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
  size_t localMemorySize  = context.device()
    .getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
  size_t maxComputeUnits  = context.device()
    .getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  size_t maxWorkGroupSize = context.device()
    .getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

  // Print device information.
  printValue("Global Memory Size", globalMemorySize);
  printValue("Local Memory Size", localMemorySize);
  printValue("Max Compute Units", maxComputeUnits);
  printValue("Max Work-Group Size", maxWorkGroupSize);
  


  // Create a command queue.
  cl::CommandQueue queue(context.context(), context.device(), 0, &result);
  CLContext::throwOnError("cl::CommandQueue", result);
  
  // Build a program from the source
  cl::Program::Sources progSource(1, std::make_pair(kernelSource.c_str(),
                                                    kernelSource.size()));
  cl::Program          program(context.context(), progSource, &result);
  CLContext::throwOnError("cl::Program failed", result);

  std::vector<cl::Device> devices;
  devices.push_back(context.device());
  
  result = program.build(devices);
  if(result != CL_SUCCESS) {
    std::cout << "Source compilation failed.\n";
    std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.device());
    return 1;
  }

  // Extract the kernel
  cl::Kernel kernel(program, "kernel_func", &result);
  CLContext::throwOnError("Failed to extract kernel", result);


  // Allocate host arrays
  float* hostData = new float[arraySize];

  // Fill host arrays
  for(int i = 0; i < params.paddedSize; ++i) {
    for(int j = 0; j < params.paddedSize; ++j) {
      for(int k = 0; k < params.paddedSize; ++k) {
        
        if(i < params.padding || i >= (params.paddedSize-params.padding) ||
           j < params.padding || j >= (params.paddedSize-params.padding) ||
           k < params.padding || k >= (params.paddedSize-params.padding)) {
          hostData[i*params.paddedSize*params.paddedSize+j*params.paddedSize+k]
            = 0.0f;
        }
        else {         
          hostData[i*params.paddedSize*params.paddedSize+j*params.paddedSize+k]
            = (float)rand() / ((float)RAND_MAX + 1.0f);
        }
      }
    }
  }


  // Compute reference

  float* reference = NULL;

  if(vm.count("verify")) {

    reference = new float[arraySize];

    float* refA;
    float* refB;

    refA = new float[arraySize];
    refB = new float[arraySize];

    memcpy(refA, hostData, arraySize);
    memcpy(refB, hostData, arraySize);

#define ARRAY_REF(A, i, j, k)                   \
    (A[i*params.paddedSize*params.paddedSize+j*params.paddedSize+k])

    for(int t = 0; t < params.timeSteps; ++t) {
      for(int i = params.padding; i < params.paddedSize-params.padding; ++i) {
        for(int j = params.padding; j < params.paddedSize-params.padding; ++j) {
          for(int k = params.padding; k < params.paddedSize-params.padding;
              ++k) {
            ARRAY_REF(refB, i, j, k)
              = (0.143f) * (ARRAY_REF(refA, i, j-1, k) +
                            ARRAY_REF(refA, i, j, k) +
                            ARRAY_REF(refA, i, j+1, k) +
                            ARRAY_REF(refA, i, j, k-1) +
                            ARRAY_REF(refA, i, j, k+1) +
                            ARRAY_REF(refA, i+1, j, k) +
                            ARRAY_REF(refA, i-1, j, k));
          }
        }
      }
      memcpy(refA, refB, arraySize);
    }

    memcpy(reference, refA, arraySize);

    delete [] refA;
    delete [] refB;

  }


  // Allocate device arrays
  cl::Buffer deviceInput(context.context(), CL_MEM_READ_WRITE,
                         arraySize, NULL, &result);
  CLContext::throwOnError("Failed to allocate device input", result);

  cl::Buffer deviceOutput(context.context(), CL_MEM_READ_WRITE,
                          arraySize, NULL, &result);
  CLContext::throwOnError("Failed to allocate device output", result);

  // Copy host data to device
  result = queue.enqueueWriteBuffer(deviceInput, CL_TRUE, 0,
                                    arraySize, hostData,
                                    NULL, NULL);
  CLContext::throwOnError("Failed to copy input data to device", result);
  result = queue.enqueueWriteBuffer(deviceOutput, CL_TRUE, 0,
                                    arraySize, hostData,
                                    NULL, NULL);
  CLContext::throwOnError("Failed to copy input data to device", result);

  cl::NDRange globalSize(params.blockSizeX*params.numBlocksX,
                         params.blockSizeY*params.numBlocksY,
                         params.blockSizeZ*params.numBlocksZ);
  cl::NDRange localSize(params.blockSizeX, params.blockSizeY,
                        params.blockSizeZ);


  cl::Buffer* inputBuffer;
  cl::Buffer* outputBuffer;

  inputBuffer  = &deviceInput;
  outputBuffer = &deviceOutput;

  cl::Event waitEvent;

  double startTime = rtclock();

  for(int t = 0; t < params.timeSteps / params.timeTileSize; ++t) {

    // Set kernel arguments
    result = kernel.setArg(0, *inputBuffer);
    CLContext::throwOnError("Failed to set input parameter", result);
    result = kernel.setArg(1, *outputBuffer);
    CLContext::throwOnError("Failed to set output parameter", result);
  
    // Invoke the kernel
    result = queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                        globalSize, localSize,
                                        0, &waitEvent);
    CLContext::throwOnError("Kernel launch failed", result);

    std::swap(inputBuffer, outputBuffer);
  }

  waitEvent.wait();

  double endTime = rtclock();
  double elapsed = endTime - startTime;

  // Copy results back to host
  result = queue.enqueueReadBuffer(*inputBuffer, CL_TRUE, 0,
                                   arraySize, hostData,
                                   NULL, NULL);
  CLContext::throwOnError("Failed to copy result to host", result);

  double gflops   = (double)params.problemSize * (double)params.problemSize
    * (double)params.problemSize
    * 7.0 * (double)params.timeSteps / elapsed / 1e9;
  //double gflops = stencilGen.computeGFlops(elapsed);
  printValue("Actual GFlop/s", gflops);

  gflops = (double)params.blockSizeX * (double)params.blockSizeY
    * (double)params.blockSizeZ
    * (double)params.numBlocksX * (double)params.numBlocksY
    * (double)params.numBlocksZ
    * (double)params.elementsPerThread * 7.0 * (double)params.timeSteps
    / elapsed / 1e9;

  printValue("Device GFlop/s", gflops);
  
  if(vm.count("verify")) {
    compareResults(reference, hostData, params);
  }



  // Clean-up
  delete [] hostData;

  if(vm.count("verify")) {
    delete [] reference;
  }
  
  return 0;
}