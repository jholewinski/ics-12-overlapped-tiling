
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
  int32_t     problemSize;
  std::string dataType;
  
  // Derived
  int32_t     padding;
  int32_t     compsPerBlockX;
  int32_t     compsPerBlockY;
  int32_t     sizeLCM;
  int32_t     realSize;
  int32_t     paddedSize;
  int32_t     realPerBlockX;
  int32_t     realPerBlockY;
  int32_t     sharedSizeX;
  int32_t     sharedSizeY;
  int32_t     numBlocksX;
  int32_t     numBlocksY;
  std::string fpSuffix;

  bool breakThreads;
  

  /**
   * Default constructor.
   */
  GeneratorParams(int32_t tts      = 1,
                  int32_t ept      = 1,
                  int32_t bsx      = 16,
                  int32_t bsy      = 16,
                  int32_t ps       = 1024,
                  int32_t ts       = 64,
                  bool    bt       = false,
                  std::string type = "float")
    : timeTileSize(tts),
      timeSteps(ts),
      elementsPerThread(ept),
      problemSize(ps),
      dataType(type),
      breakThreads(bt),
      blockSizeX(bsx),
      blockSizeY(bsy) {
  }

  void computeDerived() {
    // Compute derived values
    padding        = timeTileSize;
    compsPerBlockX = blockSizeX;
    compsPerBlockY = blockSizeY*elementsPerThread;
    realPerBlockX  = compsPerBlockX - 2*(timeTileSize-1);
    realPerBlockY  = compsPerBlockY - 2*(timeTileSize-1);
    sizeLCM        = boost::math::gcd(realPerBlockX, realPerBlockY);
    realSize       = (problemSize / sizeLCM) * sizeLCM;
    numBlocksX     = realSize / realPerBlockX;
    numBlocksY     = realSize / realPerBlockY;
    sharedSizeX    = blockSizeX + 2;
    sharedSizeY    = blockSizeY * elementsPerThread + 2;
    paddedSize     = realSize + 2*padding;

    // Make sure padded size is a multiple of 32 floats
    //int32_t rem  = paddedSize % 32;
    //paddedSize  += 32 - rem;

    if(dataType == "float") {
      fpSuffix = "f";
    } else {
      fpSuffix = "";
    }

    /*if(padding < 1 || compsPerBlockX < 1 || compsPerBlockY < 1             ||
       realPerBlockX < 1 || realPerBlockY < 1 || sizeLCM < 1 || realSize < 1 ||
       numBlocksX < 1 || numBlocksY < 1 || sharedSizeX < 1                   ||
       sharedSizeY < 1 || paddedSize < 1) {
      throw std::runtime_error("Consistency error!");
      }*/
    assert(padding > 0);
    assert(compsPerBlockX > 0);
    assert(compsPerBlockY > 0);
    assert(realPerBlockX > 0);
    assert(realPerBlockY > 0);
    assert(realSize > 0);
    assert(numBlocksX > 0);
    assert(numBlocksY > 0);
    assert(paddedSize > 0);
  }
};

/**
 * Generator for Jacobi 2D.
 */
class Jacobi2DGenerator : public ProgramGenerator {
public:

  Jacobi2DGenerator();

  virtual ~Jacobi2DGenerator();

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


Jacobi2DGenerator::Jacobi2DGenerator() {
}

Jacobi2DGenerator::~Jacobi2DGenerator() {
}

std::string Jacobi2DGenerator::generate(GeneratorParams& params) {
  std::stringstream program;

  params.computeDerived();
  
  generateHeader(program, params);
  generateLocals(program, params);
  generateCompute(program, params);
  generateFooter(program);

  return program.str();
}

void Jacobi2DGenerator::generateHeader(std::ostream& stream,
                                       const GeneratorParams& params) {
  stream << "/* Auto-generated.  Do not edit by hand. */\n";
  stream << "__kernel\n";
  stream << "void kernel_func(__read_only image2d_t inputTex,\n";
  stream << "                 sampler_t sampler,\n";
  stream << "                 __global " << params.dataType << "* output) {\n";
}

void Jacobi2DGenerator::generateFooter(std::ostream& stream) {
  stream << "}\n\n";
}

void Jacobi2DGenerator::generateLocals(std::ostream& stream,
                                       const GeneratorParams& params) {
  stream << "  __local " << params.dataType << " buffer[" << params.sharedSizeY
         << "][" << params.sharedSizeX << "];\n";

  // Compute some pointer values
  //stream << "  __global " << params.dataType
  //       << "* inputPtr = input + ((get_group_id(1)*" << params.realPerBlockY
  //       << "+get_local_id(1)*" << params.elementsPerThread << "+1)*"
  //       << params.paddedSize << ") + (get_group_id(0)*" << params.realPerBlockX
  //       << ") + get_local_id(0) + 1;\n";
  stream << "int myY = get_group_id(1)*" << params.realPerBlockY
         << "+get_local_id(1)*" << params.elementsPerThread << "+1;\n";
  stream << "int myX = get_group_id(0)*" << params.realPerBlockX
         << " + get_local_id(0) + 1;\n";

  stream << "  __global " << params.dataType
         << "* outputPtr = output + ((get_group_id(1)*" << params.realPerBlockY
         << "+get_local_id(1)*" << params.elementsPerThread << "+1)*"
         << params.paddedSize << ") + (get_group_id(0)*" << params.realPerBlockX
         << ") + get_local_id(0) + 1;\n";

  // Compute some guards
  //stream << "  int globalIndexX = (get_group_id(0)*" << params.realPerBlockX
  //       << ") + get_local_id(0) + 1;\n";
  //stream << "  int globalIndexY;\n";
  stream << "  bool validX = myX >= " << params.padding
         << " && myX < " << (params.realSize+params.padding) << ";\n";

  for(int32_t i = 0; i < params.elementsPerThread; ++i) {
    //stream << "  globalIndexY = get_group_id(1)*" << params.realPerBlockY
    //       << " + " << params.elementsPerThread << "*get_local_id(1) + " << i
    //       << " + 1;\n";
    stream << "  bool valid" << i << " = validX && myY >= "
           << params.padding << " && myY < "
           << (params.realSize+params.padding) << ";\n";
  }

  if(!params.breakThreads) {
  stream << "  bool writeValidX = get_local_id(0) >= "
         << (params.timeTileSize-1)
         << " && get_local_id(0) < "
         << (params.realPerBlockX+params.timeTileSize-1) << ";\n";
  stream << "  int effectiveTidY;\n";
  
  for(int32_t i = 0; i < params.elementsPerThread; ++i) {
    stream << "  effectiveTidY = get_local_id(1)*" << params.elementsPerThread
           << " + " << i << ";\n";
    stream << "  bool writeValid" << i << " = effectiveTidY >= "
           << params.timeTileSize-1 << " && effectiveTidY < "
           << (params.realPerBlockY+params.timeTileSize-1) << ";\n";
  }
  }

  // Declare local intermediates
  for(int32_t i = 0; i < params.elementsPerThread; ++i) {
    stream << "  " << params.dataType << " local" << i << ";\n";
    stream << "  " << params.dataType << " new" << i << ";\n";
  }
}

void Jacobi2DGenerator::generateCompute(std::ostream& stream,
                                        const GeneratorParams& params) {

  for(int32_t i = 0; i < params.elementsPerThread; ++i) {
    stream << "  {\n";
    stream << "    " << params.dataType << " val0, val1, val2, val3, val4;\n";
    stream << "    float u, v;\n";
    stream << "    // Left\n";
    //stream << "    val0 = *(inputPtr+(" << params.paddedSize << "*" << i
    //       << ")-1);\n";
    stream << "    u = myX-1; v = myY;\n";
    stream << "    u /= " << params.paddedSize << ".0" << params.fpSuffix << ";\n";
    stream << "    v /= " << params.paddedSize << ".0" << params.fpSuffix << ";\n";
    stream << "    val0 = read_imagef(inputTex, sampler, (float2)(u,v)).x;\n";
    stream << "    // Center\n";

    //stream << "    val1 = *(inputPtr+(" << params.paddedSize << "*" << i
    //       << "));\n";
    stream << "    u = myX; v = myY;\n";
    stream << "    u /= " << params.paddedSize << ".0" << params.fpSuffix << ";\n";
    stream << "    v /= " << params.paddedSize << ".0" << params.fpSuffix << ";\n";
    stream << "    val1 = read_imagef(inputTex, sampler, (float2)(u,v)).x;\n";
    stream << "    // Right\n";
    //stream << "    val2 = *(inputPtr+(" << params.paddedSize << "*" << i
    //       << ")+1);\n";
    stream << "    u = myX+1; v = myY;\n";
    stream << "    u /= " << params.paddedSize << ".0" << params.fpSuffix << ";\n";
    stream << "    v /= " << params.paddedSize << ".0" << params.fpSuffix << ";\n";
    stream << "    val2 = read_imagef(inputTex, sampler, (float2)(u,v)).x;\n";
    stream << "    // Top\n";
    //stream << "    val3 = *(inputPtr+(" << params.paddedSize << "*" << (i-1)
    //       << "));\n";
    stream << "    u = myX; v = myY+1;\n";
    stream << "    u /= " << params.paddedSize << ".0" << params.fpSuffix << ";\n";
    stream << "    v /= " << params.paddedSize << ".0" << params.fpSuffix << ";\n";
    stream << "    val3 = read_imagef(inputTex, sampler, (float2)(u,v)).x;\n";
    stream << "    // Bottom\n";
    //stream << "    val4 = *(inputPtr+(" << params.paddedSize << "*" << (i+1)
    //       << "));\n";
    stream << "    u = myX; v = myY-1;\n";
    stream << "    u /= " << params.paddedSize << ".0" << params.fpSuffix << ";\n";
    stream << "    v /= " << params.paddedSize << ".0" << params.fpSuffix << ";\n";
    stream << "    val4 = read_imagef(inputTex, sampler, (float2)(u,v)).x;\n";
    stream << "    " << params.dataType
           << " result = 0.2" << params.fpSuffix
           << " * (val0+val1+val2+val3+val4);\n";
    stream << "    result = (valid" << i << ") ? result : 0.0"
           << params.fpSuffix << ";\n";
    stream << "    buffer[get_local_id(1)*" << params.elementsPerThread << "+"
           << i
           << "+1][get_local_id(0)+1] = result;\n";
    stream << "    local" << i << " = result;\n";
    stream << "  }\n";
  }

  stream << "  barrier(CLK_LOCAL_MEM_FENCE);\n";

  for(int32_t t = 1; t < params.timeTileSize; ++t) {
    stream << "  // Time Step " << t << "\n";
    if(params.breakThreads) {
      stream << "  if(get_local_id(0) >= " << t
             << " && get_local_id(0) < get_local_size(0)-"
             << t << ")\n";
    }
    stream << "  {\n";
    for(int32_t i = 0; i < params.elementsPerThread; ++i) {
      if(params.breakThreads) {
        stream << "  if(get_local_id(1)*" << params.elementsPerThread << "+" << i
               << " >= " << t << " && get_local_id(1)*"
               << params.elementsPerThread << "+" << i << " < get_local_size(1)*"
               << params.elementsPerThread << "-" << t << ")\n";
      }
      stream << "  {\n";
      stream << "    " << params.dataType << " val0, val1, val2, val3, val4;\n";
      stream << "    // Left\n";
      stream << "    val0 = buffer[get_local_id(1)*" << params.elementsPerThread
             << "+" << i
             << "+1][get_local_id(0)];\n";
      stream << "    // Center\n";
      stream << "    val1 = local" << i << ";\n";
      stream << "    // Right\n";
      stream << "    val2 = buffer[get_local_id(1)*" << params.elementsPerThread
             << "+" << i
             << "+1][get_local_id(0)+2];\n";
      stream << "    // Top\n";
      stream << "    val3 = buffer[get_local_id(1)*" << params.elementsPerThread
             << "+" << i
             << "][get_local_id(0)+1];\n";
      stream << "    // Bottom\n";
      stream << "    val4 = buffer[get_local_id(1)*" << params.elementsPerThread
             << "+" << i
             << "+2][get_local_id(0)+1];\n";
      stream << "    " << params.dataType
             << " result = 0.2" << params.fpSuffix
             << " * (val0+val1+val2+val3+val4);\n";
      stream << "    result = (valid" << i << ") ? result : 0.0"
             << params.fpSuffix << ";\n";
      stream << "    new" << i << " = result;\n";
      stream << "  }\n";
      if(params.breakThreads) {
        //stream << "  else { return; }\n";
        stream << "  barrier(CLK_LOCAL_MEM_FENCE);\n";
      }
    }
    stream << "  barrier(CLK_LOCAL_MEM_FENCE);\n";
    for(int32_t i = 0; i < params.elementsPerThread; ++i) {
      stream << "  buffer[get_local_id(1)*" << params.elementsPerThread << "+"
             << i
             << "+1][get_local_id(0)+1] = new" << i << ";\n";
      stream << "  local" << i << " = new" << i << ";\n";
    }
    stream << "  barrier(CLK_LOCAL_MEM_FENCE);\n";
    stream << "  }\n";
    if(params.breakThreads) {
      stream << "else { return; }\n";
    }
  }
  for(int32_t i = 0; i < params.elementsPerThread; ++i) {
    if(!params.breakThreads) {
      stream << "  if(writeValid" << i << " && writeValidX)\n";
    }
    stream << "  {\n";
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
 
  Jacobi2DGenerator gen;
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
    ("block-size-y,y",
     po::value<int32_t>(&params.blockSizeY)->default_value(16),
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
    ("break-threads,b", "Break threads")
    ("output-results,o", "Output results")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if(vm.count("help")) {
    std::cerr << desc;
    return 1;
  }

  if(vm.count("break-threads")) {
    params.breakThreads = true;
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
  printValue("Elements/Thread", params.elementsPerThread);
  printValue("Num Blocks X", params.numBlocksX);
  printValue("Num Blocks Y", params.numBlocksY);
  printValue("Time Steps", params.timeSteps);
  printValue("Padding", params.padding);
  printValue("Real Size", params.realSize);
  
  int arraySize = params.paddedSize * params.paddedSize * sizeof(float);

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

  if(params.blockSizeX*params.blockSizeY > maxWorkGroupSize) {
    std::cout << "ERROR: Block dimensions are too large!\n";
    return 1;
  }

  if(params.sharedSizeX*params.sharedSizeY > localMemorySize) {
    std::cout << "ERROR: Not enough local memory for even one block!\n";
    return 1;
  }


  // Print some derived statistics
  int32_t sharedSize = params.sharedSizeX * params.sharedSizeY * 1 * 4;
  
  int32_t numBlocksFromShared = (int32_t)std::ceil((double)localMemorySize /
                                                   (double)sharedSize);
  
  int64_t totalFPPerBlock = params.blockSizeX * params.blockSizeY *
    params.elementsPerThread * params.timeSteps * 5;

  int64_t usefulFPPerBlock = 5 * params.realPerBlockX * params.realPerBlockY*
    params.timeSteps;

  double usefulFPRatio = (double)usefulFPPerBlock /
    (double)totalFPPerBlock;

  int32_t globalLoadsPerBlock = params.blockSizeX * params.blockSizeY *
    params.elementsPerThread * 5;

  int32_t globalStoresPerBlock = params.blockSizeX * params.blockSizeY *
    params.elementsPerThread * 1;

  int32_t sharedLoadsPerBlock = params.blockSizeX * params.blockSizeY *
    params.elementsPerThread * 5 * (params.timeTileSize-1);

  int32_t sharedStoresPerBlock = params.blockSizeX * params.blockSizeY *
    params.elementsPerThread * 1 * (params.timeTileSize-1);

  int32_t arithmeticIntensity = 5.0 / 5.0;

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
      if(i < params.padding || i >= (params.paddedSize-params.padding) ||
         j < params.padding || j         >= (params.paddedSize-params.padding)) {
        hostData[i*params.paddedSize +j]  = 0.0f;
      }
      else {         
        hostData[i*params.paddedSize + j] = (float)rand() / ((float)RAND_MAX + 1.0f);
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
        for(int j = params.padding; j < params.paddedSize-params.padding; ++j) {
          refB[i*params.paddedSize + j] = (1.0f / 5.0f) * (refA[i*params.paddedSize + (j-1)]
                                                    + refA[i*params.paddedSize + (j)]
                                                    + refA[i*params.paddedSize + (j+1)]
                                                    + refA[(i-1)*params.paddedSize + (j)]
                                                    + refA[(i+1)*params.paddedSize + (j)]);
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


  // Create image object
  cl::ImageFormat format(CL_R, CL_FLOAT);
  cl::Image2D     deviceTex(context.context(), CL_MEM_READ_ONLY, format,
                            params.paddedSize, params.paddedSize, 0, NULL,
                            &result);
  CLContext::throwOnError("Failed to create texture", result);

  cl::Sampler sampler(context.context(), CL_TRUE, CL_ADDRESS_CLAMP,
                      CL_FILTER_NEAREST, &result);
  CLContext::throwOnError("Failed to create sampler", result);
  
  
  
  cl::NDRange globalSize(params.blockSizeX*params.numBlocksX,
                         params.blockSizeY*params.numBlocksY);
  cl::NDRange localSize(params.blockSizeX, params.blockSizeY);


  cl::Buffer* inputBuffer;
  cl::Buffer* outputBuffer;

  inputBuffer  = &deviceInput;
  outputBuffer = &deviceOutput;

  cl::Event waitEvent;

  double startTime = rtclock();

  for(int t = 0; t < params.timeSteps / params.timeTileSize; ++t) {

    // Copy input buffer to image
    cl::size_t<3> dest;
    dest[0]   = 0;
    dest[1]   = 0;
    dest[2]   = 0;
    cl::size_t<3> region;
    region[0] = params.paddedSize;
    region[1] = params.paddedSize;
    region[2] = 1;
    result    = queue.enqueueCopyBufferToImage(*inputBuffer, deviceTex, 0, dest, region);
    CLContext::throwOnError("Failed to copy input data to texture", result);
    
    // Set kernel arguments
    result = kernel.setArg(0, deviceTex);
    CLContext::throwOnError("Failed to set input parameter", result);
    result = kernel.setArg(1, sampler);
    CLContext::throwOnError("Failed to set input parameter", result);
    result = kernel.setArg(2, *outputBuffer);
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

  double gflops   = (double)params.realSize * (double)params.realSize
    * 5.0 * (double)params.timeSteps / elapsed / 1e9;
  //double gflops = stencilGen.computeGFlops(elapsed);
  printValue("Actual GFlop/s", gflops);

  gflops = (double)params.blockSizeX * (double)params.blockSizeY
    * (double)params.numBlocksX * (double)params.numBlocksY
    * (double)params.elementsPerThread * 5.0 * (double)params.timeSteps
    / elapsed / 1e9;

  printValue("Device GFlop/s", gflops);
  
  if(vm.count("verify")) {
    compareResults(reference, hostData, params);
  }


  if(vm.count("output-results")) {
    std::cerr << "---- GPU Output ----\n";
    for(int32_t i = 0; i < params.paddedSize; ++i) {
      for(int32_t j = 0; j < params.paddedSize; ++j) {
        std::cerr << hostData[i*params.paddedSize+j] << " ";
      }
      std::cerr << "\n";
    }
    if(vm.count("verify")) {
      std::cerr << "---- Reference ----\n";
      for(int32_t i = 0; i < params.paddedSize; ++i) {
        for(int32_t j = 0; j < params.paddedSize; ++j) {
          std::cerr << reference[i*params.paddedSize+j] << " ";
        }
        std::cerr << "\n";
      }
    }
  }


  // Clean-up
  delete [] hostData;

  if(vm.count("verify")) {
    delete [] reference;
  }
  
  return 0;
}
