
#include "CLCommon.hpp"
#include "CLContext.hpp"
#include "ProgramGenerator.hpp"

#include <cmath>
#include <iomanip>
#include <fstream>

#include <boost/math/common_factor.hpp>
#include <boost/program_options.hpp>
#include <boost/regex.hpp>

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

  int32_t phaseLimit;

  bool dumpClocks;

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
      blockSizeZ(bsz),
      phaseLimit(0),
      dumpClocks(false) {
  }

  void computeDerived() {
    // Compute derived values
    padding        = timeTileSize;
    compsPerBlockX = blockSizeX;
    compsPerBlockY = blockSizeY*elementsPerThread;
    compsPerBlockZ = blockSizeZ;
    realPerBlockX  = compsPerBlockX - 2*(timeTileSize-1);
    realPerBlockY  = compsPerBlockY - 2*(timeTileSize-1);
    realPerBlockZ  = compsPerBlockZ - 2*(timeTileSize-1);
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
class Poisson3DGenerator : public ProgramGenerator {
public:

  Poisson3DGenerator();

  virtual ~Poisson3DGenerator();

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


Poisson3DGenerator::Poisson3DGenerator() {
}

Poisson3DGenerator::~Poisson3DGenerator() {
}

std::string Poisson3DGenerator::generate(GeneratorParams& params) {
  std::stringstream program;

  params.computeDerived();
  
  generateHeader(program, params);
  generateLocals(program, params);
  generateCompute(program, params);
  generateFooter(program);

  return program.str();
}

void Poisson3DGenerator::generateHeader(std::ostream& stream,
                                       const GeneratorParams& params) {
  stream << "/* Auto-generated.  Do not edit by hand. */\n";
  if (params.dumpClocks) {
    stream << "struct ClockData {\n";
    stream << "  unsigned int SMId;\n";
    stream << "  unsigned Clock[5];\n";
    stream << "};\n";
  }

  stream << "#define SQR(x) ((x)*(x))\n";

  stream << "__kernel\n";
  stream << "void kernel_func(__global " << params.dataType << "* input,\n";
  stream << "                 __global " << params.dataType << "* output,\n";
  if (params.dumpClocks) {
    stream << "                 __global struct ClockData* clockData,\n";
  }

  stream << "                 unsigned baseTime) {\n";
}

void Poisson3DGenerator::generateFooter(std::ostream& stream) {
  stream << "}\n\n";
}

void Poisson3DGenerator::generateLocals(std::ostream& stream,
                                       const GeneratorParams& params) {
  stream << "  __local " << params.dataType << " buffer[" << params.sharedSizeZ
         << "][" << params.sharedSizeY << "][" << params.sharedSizeZ << "];\n";

  if (params.dumpClocks) {
    stream << "  unsigned int clockP0;\n";
    stream << "  asm(\"mov.u32 %0, %%clock;\" : \"=r\"(clockP0));\n";
  }

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
  stream << "  bool writeValidZ = get_local_id(2) >= "
         << (params.timeTileSize-1)
         << " && get_local_id(2) < "
         << (params.realPerBlockZ+params.timeTileSize-1) << ";\n";
  

  // Declare local intermediates
  for(int32_t i = 0; i < params.elementsPerThread; ++i) {
    stream << "  " << params.dataType << " local" << i << ";\n";
    stream << "  " << params.dataType << " new" << i << ";\n";
  }

  if(params.phaseLimit == 1) {
    stream << "  if(get_local_id(0) != (unsigned)(-1)) { return; }\n";
  }

  if (params.dumpClocks) {
    stream << "  unsigned int clockP1;\n";
    stream << "  asm(\"mov.u32 %0, %%clock;\" : \"=r\"(clockP1));\n";
  }

}

void Poisson3DGenerator::generateCompute(std::ostream& stream,
                                        const GeneratorParams& params) {
  if (params.phaseLimit == 3) {
    // We only want phase 3, so completely skip phase 2
    stream << "  if(get_local_id(0) == 100000) {\n";
  }

  for(int32_t i = 0; i < params.elementsPerThread; ++i) {
    stream << "  {\n";
    for (int32_t x = -1; x < 2; ++x) {
      for (int32_t y = -1; y < 2; ++y) {
        for (int32_t z = -1; z < 2; ++z) {
          stream << "    float val" << (x+1) << (y+1) << (z+1) << ";\n";
      
          stream << "    val" << (x+1) << (y+1) << (z+1) << " = *(inputPtr+(" << params.paddedSize << "*" << (i+y)
                 << ")+(" << params.paddedSize*params.paddedSize << "*" << z << ")+" << x << ");\n";
        }
      }
    }
    stream << "    " << params.dataType
           << " result = 26.0f*val111 - (val000+val001+val002+val010+val011+val012+val020+val021+val022+val100+val101+val102+val110+val112+val120+val121+val122+val200+val201+val202+val210+val211+val212+val220+val221+val222);\n";
    stream << "    result = (valid" << i << ") ? result : 0.0"
           << params.fpSuffix << ";\n";
    stream << "    buffer[get_local_id(2)+1][get_local_id(1)*"
           << params.elementsPerThread << "+" << i
           << "+1][get_local_id(0)+1] = result;\n";
    stream << "    local" << i << " = result;\n";
    stream << "  }\n";
  }

  stream << "  barrier(CLK_LOCAL_MEM_FENCE);\n";

  if (params.dumpClocks) {
    stream << "  unsigned int clockP2;\n";
    stream << "  asm(\"mov.u32 %0, %%clock;\" : \"=r\"(clockP2));\n";
  }


  if(params.phaseLimit == 2) {
    stream << "  if(get_local_id(0) != (unsigned)(-1)) { return; }\n";
  }

  if (params.phaseLimit == 3) {
    stream << "  }\n";
  }

  stream << "  #pragma unroll\n";
  stream << "  for(int t = 1; t < " << params.timeTileSize << "; ++t) {\n";
    stream << "  if (baseTime + t >= " << params.timeSteps << ") break;\n";
    for(int32_t i = 0; i < params.elementsPerThread; ++i) {
      stream << "  {\n";
      for (int32_t x = -1; x < 2; ++x) {
        for (int32_t y = -1; y < 2; ++y) {
          for (int32_t z = -1; z < 2; ++z) {
            stream << "    float val" << (x+1) << (y+1) << (z+1) << ";\n";
            stream << "    val" << (x+1) << (y+1) << (z+1)
                   << " = buffer[get_local_id(2)+" << (z+1)
                   << "][get_local_id(1)*" << params.elementsPerThread
                   << "+" << (y+i+1)
                   << "][get_local_id(0)+" << (x+1) << "];\n";
          }
        }
      }

      stream << "    " << params.dataType
             << " result = 26.0f*val111 - (val000+val001+val002+val010+val011+val012+val020+val021+val022+val100+val101+val102+val110+val112+val120+val121+val122+val200+val201+val202+val210+val211+val212+val220+val221+val222);\n";
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
    stream << "  }\n";

    if(params.phaseLimit == 3) {
    stream << "  if(get_local_id(0) != (unsigned)(-1)) { return; }\n";
  }

  if (params.dumpClocks) {
    stream << "  unsigned int clockP3;\n";
    stream << "  asm(\"mov.u32 %0, %%clock;\" : \"=r\"(clockP3));\n";
  }
  

        
  for(int32_t i = 0; i < params.elementsPerThread; ++i) {
    stream << "  if(writeValid" << i << " && writeValidX) {\n";
    stream << "    *(outputPtr+(" << params.paddedSize << "*" << i
           << ")) = local" << i << ";\n";
    stream << "  }\n";
  }


  if (params.dumpClocks) {
    stream << "  unsigned int clockP4;\n";
    stream << "  asm(\"mov.u32 %0, %%clock;\" : \"=r\"(clockP4));\n";

    stream << "  unsigned int sm;\n";
    stream << "  asm(\"mov.u32 %0, %%smid;\" : \"=r\"(sm));\n";

    stream << "  if (sm == 1) {\n";
    stream << "    struct ClockData CD;\n";
    stream << "    CD.SMId = sm;\n";
    stream << "    CD.Clock[0] = clockP0;\n";
    stream << "    CD.Clock[1] = clockP1;\n";
    stream << "    CD.Clock[2] = clockP2;\n";
    stream << "    CD.Clock[3] = clockP3;\n";
    stream << "    CD.Clock[4] = clockP4;\n";

    stream << "    unsigned BlockId = get_group_id(2)*get_num_groups(1)*get_num_groups(0) + get_group_id(1)*get_num_groups(0) + get_group_id(0);\n";
    stream << "    unsigned ThreadsPerBlock = get_local_size(0) * get_local_size(1) * get_local_size(2);\n";
    stream << "    unsigned ThreadId = get_local_id(2)*get_local_size(1)*get_local_size(0) + get_local_id(1)*get_local_size(0) + get_local_id(0);\n";
    stream << "    unsigned GlobalId = BlockId*ThreadsPerBlock + ThreadId;\n";
    stream << "    clockData[GlobalId] = CD;\n";
    stream << "  }\n";
  }

}


void compareResults(float* host, float* device, const GeneratorParams& params) {
  
  double errorNorm, refNorm, diff;
  errorNorm = 0.0;
  refNorm   = 0.0;

  for(int i = params.padding; i < params.paddedSize-params.padding; ++i) {
    for(int j = params.padding; j < params.paddedSize-params.padding; ++j) {
      for(int k = params.padding; k < params.paddedSize-params.padding; ++k) {
      
        float h
        = host[i*params.paddedSize*params.paddedSize+j*params.paddedSize+k];
        float d
        = device[i*params.paddedSize*params.paddedSize+j*params.paddedSize+k];
      
        diff       = h - d;
        errorNorm += diff*diff;
        refNorm   += h*h;
      }
    }
  }
  
  errorNorm = std::sqrt(errorNorm);
  refNorm   = std::sqrt(refNorm);

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
 
  Poisson3DGenerator gen;
  GeneratorParams   params;

  po::options_description desc("Options");
  desc.add_options()
    ("help,h", "Show usage information")
    ("dump-clocks,c",
     po::value<bool>(&params.dumpClocks)->default_value(false),
     "Dump clock values")
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
     "Set block size (Z)")
    ("elements-per-thread,e",
     po::value<int32_t>(&params.elementsPerThread)->default_value(1),
     "Set elements per thread")
    ("time-tile-size,s",
     po::value<int32_t>(&params.timeTileSize)->default_value(1),
     "Set time tile size")
    ("phase-limit,p",
     po::value<int32_t>(&params.phaseLimit)->default_value(0),
     "Stop after a certain kernel phase")
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
    kernelSource        = gen.generate(params);
  } else {
    std::ifstream kernelStream(kernelFile.c_str());
    kernelSource        = std::string(std::istreambuf_iterator<char>(kernelStream),
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

  if(params.blockSizeX*params.blockSizeY*params.blockSizeZ > maxWorkGroupSize) {
    std::cout << "ERROR: Block dimensions are too large!\n";
    return 1;
  }

  if(params.sharedSizeX*params.sharedSizeY*params.sharedSizeZ >
     localMemorySize) {
    std::cout << "ERROR: Not enough local memory for even one block!\n";
    return 1;
  }

  // Print some derived statistics
  int32_t sharedSize = params.sharedSizeX * params.sharedSizeY * params.sharedSizeZ * 1 * 4;
  
  int32_t numBlocksFromShared = (int32_t)std::ceil((double)localMemorySize /
                                                   (double)sharedSize);
  
  int64_t totalFPPerBlock = params.blockSizeX * params.blockSizeY * params.blockSizeZ *
    params.elementsPerThread * params.timeSteps * 7;

  int64_t usefulFPPerBlock = 7 * params.realPerBlockX * params.realPerBlockY *
    params.realPerBlockZ * params.timeSteps;

  double usefulFPRatio = (double)usefulFPPerBlock /
    (double)totalFPPerBlock;

  int32_t globalLoadsPerBlock = params.blockSizeX * params.blockSizeY *
    params.blockSizeZ * params.elementsPerThread * 7;

  int32_t globalStoresPerBlock = params.blockSizeX * params.blockSizeY *
    params.blockSizeZ * params.elementsPerThread * 1;

  int32_t sharedLoadsPerBlock = params.blockSizeX * params.blockSizeY *
    params.blockSizeZ * params.elementsPerThread * 7 * (params.timeTileSize-1);

  int32_t sharedStoresPerBlock = params.blockSizeX * params.blockSizeY *
    params.blockSizeZ * params.elementsPerThread * 1 * (params.timeTileSize-1);

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
  cl::CommandQueue queue(context.context(), context.device(), CL_QUEUE_PROFILING_ENABLE, &result);
  CLContext::throwOnError("cl::CommandQueue", result);
  
  // Build a program from the source
  cl::Program::Sources progSource(1, std::make_pair(kernelSource.c_str(),
                                                    kernelSource.size()));
  cl::Program          program(context.context(), progSource, &result);
  CLContext::throwOnError("cl::Program failed", result);

  std::vector<cl::Device> devices;
  devices.push_back(context.device());
  
  result = program.build(devices, "-cl-nv-verbose");
  if(result != CL_SUCCESS) {
    std::cout << "Source compilation failed.\n";
    std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.device());
    return 1;
  }

    // Extract out the register usage
#ifndef SIM_BUILD
  std::string log = 
    program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.device());
  boost::regex                regExpr("Used ([0-9]+) registers");
  boost::smatch               match;
  std::string::const_iterator start, end;
  start           = log.begin();
  end             = log.end();
  if(boost::regex_search(start, end, match, regExpr,
                         boost::match_default)) {
    printValue("Register Usage", match[1]);
  } else {
    printValue("Register Usage", 0);
  }
#endif

  // Extract the kernel
  cl::Kernel kernel(program, "kernel_func", &result);
  CLContext::throwOnError("Failed to extract kernel", result);


  // Allocate host arrays
  float* hostData = new float[arraySize];

  // Fill host arrays
#if 0
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
#endif
  

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

#define ARRAY_REF(A, i, j, k)                                         \
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


  cl::Buffer *deviceClock;

  struct ClockData {
    unsigned int SMId;
    unsigned int Clock[5];
  };
  
  unsigned ClockDataSize = sizeof(ClockData)*params.numBlocksX*params.numBlocksY*params.numBlocksZ*params.blockSizeX*params.blockSizeY*params.blockSizeZ;
  
  if (params.dumpClocks) {
    ClockData *hostClock = new ClockData[params.blockSizeX*params.blockSizeY*params.blockSizeZ*params.numBlocksX*params.numBlocksY*params.numBlocksZ];

    memset(hostClock, 0, ClockDataSize);
    
    deviceClock = new cl::Buffer(context.context(), CL_MEM_READ_WRITE,
                                 ClockDataSize, NULL, &result);
    CLContext::throwOnError("Failed to allocate device output", result);

    result = queue.enqueueWriteBuffer(*deviceClock, CL_TRUE, 0,
                                      ClockDataSize, hostClock,
                                      NULL, NULL);
    CLContext::throwOnError("Failed to copy input data to device", result);

    delete hostClock;
  }


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

  std::vector<cl::Event> AllEvents;
  
  double startTime = rtclock();

  for(int t = 0; t < params.timeSteps; t += params.timeTileSize) {

    // Set kernel arguments
    result   = kernel.setArg(0, *inputBuffer);
    CLContext::throwOnError("Failed to set input parameter", result);
    result   = kernel.setArg(1, *outputBuffer);
    CLContext::throwOnError("Failed to set output parameter", result);
    if (params.dumpClocks) {
      result = kernel.setArg(2, *deviceClock);
      CLContext::throwOnError("Failed to set output parameter", result);
      result = kernel.setArg(3, t);
      CLContext::throwOnError("Failed to set output parameter", result);
    } else {
      result = kernel.setArg(2, t);
      CLContext::throwOnError("Failed to set output parameter", result);
    }
  
    // Invoke the kernel
    result = queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                        globalSize, localSize,
                                        0, &waitEvent);
    CLContext::throwOnError("Kernel launch failed", result);

    AllEvents.push_back(waitEvent);
    
    std::swap(inputBuffer, outputBuffer);
  }

  waitEvent.wait();

  double endTime = rtclock();
  double elapsed = endTime - startTime;

  cl_ulong EventStart;
  cl_ulong EventEnd;
  
  CLContext::throwOnError("Profile error", AllEvents[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &EventStart));
  CLContext::throwOnError("Profile error", AllEvents[AllEvents.size()-1].getProfilingInfo(CL_PROFILING_COMMAND_END, &EventEnd));

  size_t ProfileTimerResolution = context.device()
    .getInfo<CL_DEVICE_PROFILING_TIMER_RESOLUTION>();

  printValue("EventElapsed", (EventEnd-EventStart)*1e-9);
  printValue("ProfileTimerResolution", ProfileTimerResolution);

  // Copy results back to host
  result = queue.enqueueReadBuffer(*inputBuffer, CL_TRUE, 0,
                                   arraySize, hostData,
                                   NULL, NULL);
  CLContext::throwOnError("Failed to copy result to host", result);

  if (params.dumpClocks) {
    ClockData *hostClock = new ClockData[params.blockSizeX*params.blockSizeY*params.blockSizeZ*params.numBlocksX*params.numBlocksY*params.numBlocksZ];
      
    result = queue.enqueueReadBuffer(*deviceClock, CL_TRUE, 0,
                                     ClockDataSize, hostClock,
                                     NULL, NULL);
    CLContext::throwOnError("Failed to copy result to host", result);

    std::cerr << "bx,by,bz,warp,sm,clock0,clock1,clock2,clock3,clock4,\n";

    for (unsigned bz = 0; bz < params.numBlocksZ; ++bz) {
      for (unsigned by = 0; by < params.numBlocksY; ++by) {
        for (unsigned bx = 0; bx < params.numBlocksX; ++bx) {
          for (unsigned w = 0; w < (params.blockSizeX*params.blockSizeY*params.blockSizeZ/32); ++w) {
            unsigned BlockId         = bz*params.numBlocksY*params.numBlocksX + by*params.numBlocksX + bx;
            unsigned ThreadsPerBlock = params.blockSizeX*params.blockSizeY*params.blockSizeZ;
            unsigned ThreadId        = w * 32;
            unsigned GlobalId        = BlockId*ThreadsPerBlock + ThreadId;
            if (hostClock[GlobalId].SMId == 1) {
              std::cerr << bx
                        << "," << by
                        << "," << bz
                        << "," << w
                        << "," << hostClock[GlobalId].SMId
                        << "," << hostClock[GlobalId].Clock[0]
                        << "," << hostClock[GlobalId].Clock[1]
                        << "," << hostClock[GlobalId].Clock[2]
                        << "," << hostClock[GlobalId].Clock[3]
                        << "," << hostClock[GlobalId].Clock[4]
                        << ",\n";
            }
          }
        }
      }
    }

    delete hostClock;
  }


  printValue("Elapsed Time", elapsed);

  double gflops   = (double)params.realSize * (double)params.realSize
    * (double)params.realSize
    * 27.0 * (double)params.timeSteps / elapsed / 1e9;
  //double gflops = stencilGen.computeGFlops(elapsed);
  printValue("Actual GFlop/s", gflops);

  gflops = (double)params.blockSizeX * (double)params.blockSizeY
    * (double)params.blockSizeZ
    * (double)params.numBlocksX * (double)params.numBlocksY
    * (double)params.numBlocksZ
    * (double)params.elementsPerThread * 27.0 * (double)params.timeSteps
    / elapsed / 1e9;

  printValue("phase2_global_loads", 9.0);
  printValue("phase2_shared_loads", 0.0);
  printValue("compute_per_point", 27.0);
  printValue("phase3_shared_loads", 9.0);
  printValue("phase4_global_stores", 1.0);
  printValue("shared_stores", 1.0);
  printValue("num_fields", 1.0);
  printValue("data_size", 4.0);

  printValue("phase_limit", params.phaseLimit);
  
  printValue("Device GFlop/s", gflops);
  printValue("Dimensions", 3);
  
  if(vm.count("verify")) {
    compareResults(reference, hostData, params);
  }

  if (params.dumpClocks) {
    delete deviceClock;
  }
  


  // Clean-up
  delete [] hostData;

  if(vm.count("verify")) {
    delete [] reference;
  }
  
  return 0;
}
