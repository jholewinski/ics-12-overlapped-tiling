
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
  int32_t     problemSize;
  std::string dataType;
  
  // Derived
  int32_t     padding;
  int32_t     compsPerBlockX;
  int32_t     sizeLCM;
  int32_t     realSize;
  int32_t     paddedSize;
  int32_t     realPerBlockX;
  int32_t     sharedSizeX;
  int32_t     numBlocksX;
  std::string fpSuffix;
  
  

  /**
   * Default constructor.
   */
  GeneratorParams(int32_t tts      = 1,
                  int32_t ept      = 1,
                  int32_t bsx      = 16,
                  int32_t ps       = 1024,
                  int32_t ts       = 64,
                  std::string type = "float")
    : timeTileSize(tts),
      timeSteps(ts),
      elementsPerThread(ept),
      problemSize(ps),
      dataType(type),
      blockSizeX(bsx) {
  }

  void computeDerived() {
    // Compute derived values
    padding        = timeTileSize*3;
    compsPerBlockX = blockSizeX*elementsPerThread;
    realPerBlockX  = compsPerBlockX - 3*2*(timeTileSize-1);
    sizeLCM        = realPerBlockX;
    realSize       = (problemSize / sizeLCM) * sizeLCM;
    numBlocksX     = realSize / realPerBlockX;
    sharedSizeX    = blockSizeX * elementsPerThread + 3*2;
    paddedSize     = realSize + 2*padding;

    if(dataType == "float") {
      fpSuffix = "f";
    } else {
      fpSuffix = "";
    }

    if(padding < 1 || compsPerBlockX < 1                ||
       realPerBlockX < 1 || sizeLCM < 1 || realSize < 1 ||
       numBlocksX < 1 || sharedSizeX < 1                ||
       paddedSize < 1) {
      throw std::runtime_error("Consistency error!");
    }
  }
};

/**
 * Generator for Jacobi 2D.
 */
class Jacobi1DGenerator : public ProgramGenerator {
public:

  Jacobi1DGenerator();

  virtual ~Jacobi1DGenerator();

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


Jacobi1DGenerator::Jacobi1DGenerator() {
}

Jacobi1DGenerator::~Jacobi1DGenerator() {
}

std::string Jacobi1DGenerator::generate(GeneratorParams& params) {
  std::stringstream program;

  params.computeDerived();
  
  generateHeader(program, params);
  generateLocals(program, params);
  generateCompute(program, params);
  generateFooter(program);

  return program.str();
}

void Jacobi1DGenerator::generateHeader(std::ostream& stream,
                                       const GeneratorParams& params) {
  stream << "/* Auto-generated.  Do not edit by hand. */\n";
  stream << "__kernel\n";
  stream << "void kernel_func(__global " << params.dataType << "* input,\n";
  stream << "                 __global " << params.dataType << "* output) {\n";
}

void Jacobi1DGenerator::generateFooter(std::ostream& stream) {
  stream << "}\n\n";
}

void Jacobi1DGenerator::generateLocals(std::ostream& stream,
                                       const GeneratorParams& params) {
  stream << "  __local " << params.dataType << " buffer["
         << params.sharedSizeX << "];\n";

  // Compute some pointer values
  stream << "  __global " << params.dataType
         << "* inputPtr = input + (get_group_id(0)*"
         << params.realPerBlockX
         << ") + get_local_id(0) + 3;\n";

  stream << "  __global " << params.dataType
         << "* outputPtr = output + (get_group_id(0)*"
         << params.realPerBlockX
         << ") + get_local_id(0) + 3;\n";
 

  // Compute some guards
  stream << "  int globalIndexX;\n";
  for(int32_t i = 0; i < params.elementsPerThread; ++i) {
    stream << "  globalIndexX = (get_group_id(0)*"
           << params.realPerBlockX
           << ") + get_local_id(0) + "
           << (i*params.blockSizeX)
           << " + 3;\n";
    stream << "  bool valid" << i << " = globalIndexX >= " << params.padding
           << " && globalIndexX < " << (params.realSize+params.padding)
           << ";\n";
  }

  stream << "  int effectiveTid;\n";
  
  for(int32_t i = 0; i < params.elementsPerThread; ++i) {
    stream << "  effectiveTid = " << (i*params.blockSizeX)
           << " + get_local_id(0);\n";
    stream << "  bool writeValid" << i << " = effectiveTid >= "
           << params.timeTileSize-1 << " && effectiveTid < "
           << (params.realPerBlockX+params.timeTileSize-1) << ";\n";
  }
  
  // Declare local intermediates
  for(int32_t i = 0; i < params.elementsPerThread; ++i) {
    stream << "  " << params.dataType << " local" << i << ";\n";
    stream << "  " << params.dataType << " new" << i << ";\n";
  }
}

void Jacobi1DGenerator::generateCompute(std::ostream& stream,
                                        const GeneratorParams& params) {

  for(int32_t i = 0; i < params.elementsPerThread; ++i) {
    stream << "  {\n";
    stream << "    " << params.dataType
           << " val0, val1, val2, val3, val4, val5, val6;\n";
    stream << "    val0 = *(inputPtr+(" << params.blockSizeX << "*" << i
           << ")-3);\n";
    stream << "    val1 = *(inputPtr+(" << params.blockSizeX << "*" << i
           << ")-2);\n";
    stream << "    val2 = *(inputPtr+(" << params.blockSizeX << "*" << i
           << ")-1);\n";
    stream << "    val3 = *(inputPtr+(" << params.blockSizeX << "*" << i
           << ")+0);\n";
    stream << "    val4 = *(inputPtr+(" << params.blockSizeX << "*" << i
           << ")+1);\n";
    stream << "    val5 = *(inputPtr+(" << params.blockSizeX << "*" << i
           << ")+2);\n";
    stream << "    val6 = *(inputPtr+(" << params.blockSizeX << "*" << i
           << ")+3);\n";
    stream << "    " << params.dataType
           << " result = 0.2" << params.fpSuffix
           << " * (val0+val1+val2+val3+val4+val5+val6);\n";
    stream << "    result = (valid" << i << ") ? result : 0.0"
           << params.fpSuffix << ";\n";
    stream << "    buffer[get_local_id(0)+" << (i*params.blockSizeX)
           << "+3] = result;\n";
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
      stream << "    val0 = buffer[get_local_id(0)+" << (i*params.blockSizeX)
             << "+0];\n";
      stream << "    val1 = buffer[get_local_id(0)+" << (i*params.blockSizeX)
             << "+1];\n";
      stream << "    val2 = buffer[get_local_id(0)+" << (i*params.blockSizeX)
             << "+2];\n";      
      stream << "    val3 = buffer[get_local_id(0)+" << (i*params.blockSizeX)
             << "+3];\n";      
      stream << "    val4 = buffer[get_local_id(0)+" << (i*params.blockSizeX)
             << "+4];\n";      
      stream << "    val4 = buffer[get_local_id(0)+" << (i*params.blockSizeX)
             << "+5];\n";      
      stream << "    val4 = buffer[get_local_id(0)+" << (i*params.blockSizeX)
             << "+6];\n";      
      stream << "    " << params.dataType
             << " result = 0.2" << params.fpSuffix
             << " * (val0+val1+val2+val3+val4+val5+val6);\n";
      stream << "    result = (valid" << i << ") ? result : 0.0"
             << params.fpSuffix << ";\n";
      stream << "    new" << i << " = result;\n";
      stream << "  }\n";
    }
    stream << "  barrier(CLK_LOCAL_MEM_FENCE);\n";
    for(int32_t i = 0; i < params.elementsPerThread; ++i) {
      stream << "  buffer[get_local_id(0)+" << (i*params.blockSizeX)
             << "+3] = new" << i << ";\n";
      stream << "  local" << i << " = new" << i << ";\n";
    }
    stream << "  barrier(CLK_LOCAL_MEM_FENCE);\n";
  }
  for(int32_t i = 0; i < params.elementsPerThread; ++i) {
    stream << "  if(writeValid" << i << ") {\n";
    stream << "    *(outputPtr+(" << params.blockSizeX << "*" << i
           << ")) = local" << i << ";\n";
    stream << "  }\n";
  }
}


void compareResults(float* host, float* device, const GeneratorParams& params) {
  
  double errorNorm, refNorm, diff;
  errorNorm = 0.0;
  refNorm   = 0.0;

  for(int i = params.padding; i < params.paddedSize-params.padding; ++i) {
    float h = host[i];
    float d = device[i];
      
    diff       = h - d;
    //      std::cout << "h: " << h << "  d: " << d << "  diff: " << diff << "\n";
    errorNorm += diff*diff;
    refNorm   += h*h;
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
 
  Jacobi1DGenerator gen;
  GeneratorParams   params;

  po::options_description desc("Options");
  desc.add_options()
    ("help,h", "Show usage information")
    ("problem-size,n",
     po::value<int32_t>(&params.problemSize)->default_value(1024),
     "Set problem size")
    ("time-steps,t",
     po::value<int32_t>(&params.timeSteps)->default_value(64),
     "Set number of time steps")
    ("block-size-x,x",
     po::value<int32_t>(&params.blockSizeX)->default_value(16),
     "Set block size (X)")
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
  printValue("Elements/Thread", params.elementsPerThread);
  printValue("Num Blocks X", params.numBlocksX);
  printValue("Time Steps", params.timeSteps);
  printValue("Padding", params.padding);
  printValue("Real Size", params.realSize);
  printValue("Real/BlockX", params.realPerBlockX);
  
  int arraySize = params.paddedSize * sizeof(float);

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

  if(params.blockSizeX > maxWorkGroupSize) {
    std::cout << "ERROR: Block dimensions are too large!\n";
    return 1;
  }

  if(params.sharedSizeX > localMemorySize) {
    std::cout << "ERROR: Not enough local memory for even one block!\n";
    return 1;
  }


  // Print some derived statistics
  int32_t sharedSize = params.sharedSizeX * 1 * 4;
  
  int32_t numBlocksFromShared = (int32_t)std::ceil((double)localMemorySize /
                                                   (double)sharedSize);
  
  int64_t totalFPPerBlock = params.blockSizeX * 
    params.elementsPerThread * params.timeSteps * 5;

  int64_t usefulFPPerBlock = 7 * params.realPerBlockX * 
    params.timeSteps;

  double usefulFPRatio = (double)usefulFPPerBlock /
    (double)totalFPPerBlock;

  int32_t globalLoadsPerBlock = params.blockSizeX * 
    params.elementsPerThread * 7;

  int32_t globalStoresPerBlock = params.blockSizeX * 
    params.elementsPerThread * 1;

  int32_t sharedLoadsPerBlock = params.blockSizeX * 
    params.elementsPerThread * 7 * (params.timeTileSize-1);

  int32_t sharedStoresPerBlock = params.blockSizeX * 
    params.elementsPerThread * 1 * (params.timeTileSize-1);

  int32_t arithmeticIntensity = 7.0 / 7.0;

  int32_t maxBlocks = 8;        // TODO: Change based on arch.
  
  printValue("Shared Size", sharedSize);
  printValue("Num Blocks (Shared)", numBlocksFromShared);
  printValue("Total FP", totalFPPerBlock);
  printValue("Useful FP", usefulFPPerBlock);
  printValue("Useful Ratio", usefulFPRatio);
  printValue("Global Loads/Block", globalLoadsPerBlock);
  printValue("Global Stores/Block", globalStoresPerBlock);
  printValue("Shared Loads/Block", sharedLoadsPerBlock);
  printValue("Shared Stores/Block", sharedStoresPerBlock);
  printValue("Arithmetic Intensity", arithmeticIntensity);
  printValue("Max Blocks", maxBlocks);

  

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
      if(i < params.padding || i >= (params.paddedSize-params.padding)) {
        hostData[i] = 0.0f;
      }
      else {         
        hostData[i] = (float)rand() / ((float)RAND_MAX + 1.0f);
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
  
    for(int t = 0; t < params.timeSteps; ++t) {
      for(int i = params.padding; i < params.paddedSize-params.padding; ++i) {
          refB[i] = 0.2f * (refA[i-3] +
                            refA[i-2] +
                            refA[i-1] +
                            refA[i] +
                            refA[i+1] +
                            refA[i+2] +
                            refA[i+3]);
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

  cl::NDRange globalSize(params.blockSizeX*params.numBlocksX);                    
  cl::NDRange localSize(params.blockSizeX);


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

  printValue("Elapsed Time", elapsed);

  double gflops   = (double)params.realSize
    * 7.0 * (double)params.timeSteps / elapsed / 1e9;
  //double gflops = stencilGen.computeGFlops(elapsed);
  printValue("Actual GFlop/s", gflops);

  gflops = (double)params.blockSizeX 
    * (double)params.numBlocksX 
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
