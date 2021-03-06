
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
  bool timeOnlyGMem;
  bool timeOnlyInstr;
  bool timeOnlySMem;
  int32_t phaseLimit;

  bool dumpClocks;

  int32_t                 numBlocksZ;
  
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
      blockSizeY(bsy),
      blockSizeZ(1),
      timeOnlyGMem(false),
      timeOnlyInstr(false),
      timeOnlySMem(false),
      phaseLimit(0),
      dumpClocks(false) {
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

    numBlocksZ = 1;
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
  if (params.dumpClocks) {
    stream << "struct ClockData {\n";
    stream << "  unsigned int SMId;\n";
    stream << "  unsigned Clock[5];\n";
    stream << "};\n";
  }
  stream << "__kernel\n";
  stream << "void kernel_func(__global " << params.dataType << "* input,\n";
  stream << "                 __global " << params.dataType << "* output,\n";
  if (params.dumpClocks) {
    stream << "                 __global struct ClockData* clockData,\n";
  }
  stream << "                 unsigned baseTime) {\n";
}

void Jacobi2DGenerator::generateFooter(std::ostream& stream) {
  stream << "}\n\n";
}

void Jacobi2DGenerator::generateLocals(std::ostream& stream,
                                       const GeneratorParams& params) {
  stream << "  __local " << params.dataType << " buffer[" << params.sharedSizeY
         << "][" << params.sharedSizeX << "];\n";

  if (params.dumpClocks) {
    stream << "  unsigned int clockP0;\n";
    stream << "  asm(\"mov.u32 %0, %%clock;\" : \"=r\"(clockP0));\n";
  }

  // Compute some pointer values
  stream << "  __global " << params.dataType
         << "* inputPtr = input + ((get_group_id(1)*" << params.realPerBlockY
         << "+get_local_id(1)*" << params.elementsPerThread << "+1)*"
         << params.paddedSize << ") + (get_group_id(0)*" << params.realPerBlockX
         << ") + get_local_id(0) + 1;\n";

  stream << "  __global " << params.dataType
         << "* outputPtr = output + ((get_group_id(1)*" << params.realPerBlockY
         << "+get_local_id(1)*" << params.elementsPerThread << "+1)*"
         << params.paddedSize << ") + (get_group_id(0)*" << params.realPerBlockX
         << ") + get_local_id(0) + 1;\n";

  // Compute some guards
  stream << "  int globalIndexX = (get_group_id(0)*" << params.realPerBlockX
         << ") + get_local_id(0) + 1;\n";
  stream << "  int globalIndexY;\n";
  stream << "  bool validX = globalIndexX >= " << params.padding
         << " && globalIndexX < " << (params.realSize+params.padding) << ";\n";

  for(int32_t i = 0; i < params.elementsPerThread; ++i) {
    stream << "  globalIndexY = get_group_id(1)*" << params.realPerBlockY
           << " + " << params.elementsPerThread << "*get_local_id(1) + " << i
           << " + 1;\n";
    stream << "  bool valid" << i << " = validX && globalIndexY >= "
           << params.padding << " && globalIndexY < "
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

  if(params.timeOnlyGMem) {
    for(int32_t i = 0; i < params.elementsPerThread; ++i) {
      stream << "  " << params.dataType << " test0_" << i << ";\n";
      stream << "  " << params.dataType << " test1_" << i << ";\n";
      stream << "  " << params.dataType << " test2_" << i << ";\n";
      stream << "  " << params.dataType << " test3_" << i << ";\n";
      stream << "  " << params.dataType << " test4_" << i << ";\n";
    }
  }

  if(params.phaseLimit == 1) {
    stream << "  if(get_local_id(0) != (unsigned)(-1)) { return; }\n";
  }

  if (params.dumpClocks) {
    stream << "  unsigned int clockP1;\n";
    stream << "  asm(\"mov.u32 %0, %%clock;\" : \"=r\"(clockP1));\n";
  }

  //stream << "  asm(\"mov.u32 %0, %%clock;\" : \"=r\"(clockStop));\n";
  //stream << "  if(get_global_id(0) == 0) {\n";
  //stream << "    clockData[0] = (clockStop - clockStart);\n";
  //stream << "  }\n";
}

void Jacobi2DGenerator::generateCompute(std::ostream& stream,
                                        const GeneratorParams& params) {

  //stream << "  asm(\"mov.u32 %0, %%clock;\" : \"=r\"(clockStart));\n";


  if (params.phaseLimit == 3) {
    // We only want phase 3, so completely skip phase 2
    stream << "  if(get_local_id(0) == 100000) {\n";
  }
  
  for(int32_t i = 0; i < params.elementsPerThread; ++i) {
    stream << "  {\n";
    if(!params.timeOnlySMem) {
      stream << "    " << params.dataType << " val0, val1, val2, val3, val4;\n";
      stream << "    // Left\n";
      stream << "    val0 = *(inputPtr+(" << params.paddedSize << "*" << i
             << ")-1);\n";
      if(!params.timeOnlyInstr) {
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
      } else {
        stream << "  val1 = val2 = val3 = val4 = val0;\n";
      }
    }
    if(params.timeOnlyGMem) {
      stream << "  test0_" << i << " = val0;\n";
      stream << "  test1_" << i << " = val1;\n";
      stream << "  test2_" << i << " = val2;\n";
      stream << "  test3_" << i << " = val3;\n";
      stream << "  test4_" << i << " = val4;\n";
    } else {
      if(params.timeOnlySMem) {
        stream << "  float result = (float)get_local_id(0);\n";
      } else {
        stream << "    " << params.dataType
               << " result = 0.2" << params.fpSuffix
               << " * (val0+val1+val2+val3+val4);\n";
        stream << "    result = (valid" << i << ") ? result : 0.0"
               << params.fpSuffix << ";\n";
      }
      if(!params.timeOnlyInstr) {
        stream << "    buffer[get_local_id(1)*" << params.elementsPerThread << "+"
               << i
               << "+1][get_local_id(0)+1] = result;\n";
      }
      stream << "    local" << i << " = result;\n";
    }
    stream << "  }\n";
  }

  if(!params.timeOnlyGMem && !params.timeOnlyInstr && !params.timeOnlySMem) {
    stream << "  barrier(CLK_LOCAL_MEM_FENCE);\n";
  }

  if(params.phaseLimit == 2) {
    stream << "  if(get_local_id(0) != (unsigned)(-1)) { return; }\n";
  }

  if (params.phaseLimit == 3) {
    stream << "  }\n";
  }

  if (params.dumpClocks) {
    stream << "  unsigned int clockP2;\n";
    stream << "  asm(\"mov.u32 %0, %%clock;\" : \"=r\"(clockP2));\n";
  }

  //stream << "  asm(\"mov.u32 %0, %%clock;\" : \"=r\"(clockStop));\n";
  //stream << "  if(get_global_id(0) == 0) {\n";
  //stream << "    clockData[1] = (clockStop - clockStart);\n";
  //stream << "  }\n";

  //stream << "  asm(\"mov.u32 %0, %%clock;\" : \"=r\"(clockStart));\n";

  if(!params.timeOnlyGMem) {
    stream << "  #pragma unroll\n";
    stream << "  for (int t = 1; t < " << params.timeTileSize << "; ++t) {\n";
    //stream << "  // Time Step " << t << "\n";
    if(params.breakThreads) {
      stream << "  if(get_local_id(0) >= t"
             << " && get_local_id(0) < get_local_size(0)-"
             << "t)\n";
    }
    stream << "  if (baseTime + t >= " << params.timeSteps << ") break;\n";
    stream << "  {\n";
    for(int32_t i = 0; i < params.elementsPerThread; ++i) {
      if(params.breakThreads) {
        stream << "  if(get_local_id(1)*" << params.elementsPerThread << "+" << i
               << " >= " << "t && get_local_id(1)*"
               << params.elementsPerThread << "+" << i << " < get_local_size(1)*"
               << params.elementsPerThread << "-" << "t)\n";
      }
      stream << "  {\n";
      stream << "    " << params.dataType << " val0, val1, val2, val3, val4;\n";
      if(!params.timeOnlyInstr) {
        stream << "    // Left\n";
        stream << "    val0 = buffer[get_local_id(1)*" << params.elementsPerThread
               << "+" << i
               << "+1][get_local_id(0)];\n";
      }
      stream << "    // Center\n";
      stream << "    val1 = local" << i << ";\n";
      if(!params.timeOnlyInstr) {
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
      } else {
        stream << "val0 = val2 = val3 = val4 = val1;\n";
      }
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
    if(!params.timeOnlyInstr) {
      stream << "  barrier(CLK_LOCAL_MEM_FENCE);\n";
      for(int32_t i = 0; i < params.elementsPerThread; ++i) {
        stream << "  buffer[get_local_id(1)*" << params.elementsPerThread << "+"
               << i
               << "+1][get_local_id(0)+1] = new" << i << ";\n";
        stream << "  local" << i << " = new" << i << ";\n";
      }
      stream << "  barrier(CLK_LOCAL_MEM_FENCE);\n";
    } else {
      for(int32_t i = 0; i < params.elementsPerThread; ++i) {
        stream << "  local" << i << " = new" << i << ";\n";
      }
    }

    stream << "  }\n";
    if(params.breakThreads) {
      stream << "else { return; }\n";
    }
  }
  stream << "  }\n";

  //stream << "  asm(\"mov.u32 %0, %%clock;\" : \"=r\"(clockStop));\n";
  //stream << "  if(get_global_id(0) == 0) {\n";
  //stream << "    clockData[2] = (clockStop - clockStart);\n";
  //stream << "  }\n";

  //stream << "  asm(\"mov.u32 %0, %%clock;\" : \"=r\"(clockStart));\n";

  if (params.dumpClocks) {
    stream << "  unsigned int clockP3;\n";
    stream << "  asm(\"mov.u32 %0, %%clock;\" : \"=r\"(clockP3));\n";
  }
  
  if(params.phaseLimit == 3) {
    stream << "  if(get_local_id(0) != (unsigned)(-1)) { goto dump_stats; }\n";
  }
  
  if(params.timeOnlyInstr || params.timeOnlySMem) {
    stream << "  if(get_local_id(0) == 1000000) {\n";
  }
  for(int32_t i = 0; i < params.elementsPerThread; ++i) {
    if(!params.breakThreads) {
      stream << "  if(writeValid" << i << " && writeValidX)\n";
    }
    stream << "  {\n";
    if(params.timeOnlyGMem) {
      stream << "    *(outputPtr+(" << params.paddedSize << "*" << i
             << ")) = test0_" << i << " + test1_" << i << " + test2_" << i
             << " + test3_" << i << " + test4_" << i << ";\n";
    } else {
      stream << "    *(outputPtr+(" << params.paddedSize << "*" << i
             << ")) = local" << i << ";\n";
    }
    stream << "  }\n";
  }
  if(params.timeOnlyInstr || params.timeOnlySMem) {
    stream << "}\n";
  }

  stream << "  unsigned int clockP4;\n";

  stream << "dump_stats: (void)0;\n\n";
  
  if (params.dumpClocks) {
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

    stream << "    unsigned BlockId = get_group_id(1)*get_num_groups(0) + get_group_id(0);\n";
    stream << "    unsigned ThreadsPerBlock = get_local_size(0) * get_local_size(1);\n";
    stream << "    unsigned ThreadId = get_local_id(1)*get_local_size(0) + get_local_id(0);\n";
    stream << "    unsigned GlobalId = BlockId*ThreadsPerBlock + ThreadId;\n";
    stream << "    clockData[GlobalId] = CD;\n";
    stream << "  }\n";
  }
  //stream << "  asm(\"mov.u32 %0, %%clock;\" : \"=r\"(clockStop));\n";
  //stream << "  if(get_global_id(0) == 0) {\n";
  //stream << "    clockData[3] = (clockStop - clockStart);\n";
  //stream << "  }\n";
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
  std::string timeOnly;

  srand(123456);

  Jacobi2DGenerator gen;
  GeneratorParams   params;

  po::options_description desc("Options");
  desc.add_options()
    ("help,h", "Show usage information")
    ("problem-size,n",
     po::value<int32_t>(&params.problemSize)->default_value(1024),
     "Set problem size")
    ("dump-clocks,c",
     po::value<bool>(&params.dumpClocks)->default_value(false),
     "Dump clock values")
    ("time-steps,t",
     po::value<int32_t>(&params.timeSteps)->default_value(64),
     "Set number of time steps")
    ("block-size-x,x",
     po::value<int32_t>(&params.blockSizeX)->default_value(16),
     "Set block size (X)")
    ("block-size-y,y",
     po::value<int32_t>(&params.blockSizeY)->default_value(16),
     "Set block size (Y)")
    ("block-size-z,z",
     po::value<int32_t>(&params.blockSizeZ)->default_value(1),
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
    ("break-threads,b", "Break threads")
    ("time-only",
     po::value<std::string>(&timeOnly)->default_value(""),
     "Time only the specified kernel work (gmem, instr)")
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

  if(timeOnly == "gmem") {
    params.timeOnlyGMem     = true;
  } else if(timeOnly == "instr") {
    params.timeOnlyInstr    = true;
  } else if(timeOnly == "smem") {
    params.timeOnlySMem     = true;
  } else {
    assert(timeOnly.size() == 0);
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

  //printValue("Problem Size", params.problemSize);
  //printValue("Time Tile Size", params.timeTileSize);
  //printValue("Padded Size", params.paddedSize);
  //printValue("Block Size X", params.blockSizeX);
  //printValue("Block Size Y", params.blockSizeY);
  //printValue("Elements/Thread", params.elementsPerThread);
  //printValue("Num Blocks X", params.numBlocksX);
  //printValue("Num Blocks Y", params.numBlocksY);
  //printValue("Time Steps", params.timeSteps);
  //printValue("Padding", params.padding);
  //printValue("Real Size", params.realSize);

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
  //printValue("Global Memory Size", globalMemorySize);
  //printValue("Local Memory Size", localMemorySize);
  //printValue("Max Compute Units", maxComputeUnits);
  //printValue("Max Work-Group Size", maxWorkGroupSize);

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
    params.elementsPerThread * params.timeTileSize * 5;

  int64_t usefulFPPerBlock = 5 * params.realPerBlockX * params.realPerBlockY*
    params.timeTileSize;

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

  //printValue("Shared Size", sharedSize);
  //printValue("Num Blocks (Shared)", numBlocksFromShared);
  //printValue("Total FP/Block", totalFPPerBlock);
  //printValue("Useful FP/Block", usefulFPPerBlock);
  //printValue("Useful Ratio", usefulFPRatio);
  //printValue("Global Loads/Block", globalLoadsPerBlock);
  //printValue("Global Stores/Block", globalStoresPerBlock);
  //printValue("Shared Loads/Block", sharedLoadsPerBlock);
  //printValue("Shared Stores/Block", sharedStoresPerBlock);
  //printValue("Arithmetic Intensity", arithmeticIntensity);
  //printValue("Max Blocks", maxBlocks);


  ProgramGenerator::printProgramParameters(params, 1, 5, 1, 5);


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
#ifndef SIM_BUILD
    std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.device());
#endif
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
#ifndef SIM_BUILD
  CLContext::throwOnError("Failed to extract kernel", result);
#endif


  // Allocate host arrays
  float* hostData = new float[arraySize];

  // Fill host arrays
#if 0
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

  cl::Buffer *deviceClock;

  struct ClockData {
    unsigned int SMId;
    unsigned int Clock[5];
  };
  
  unsigned ClockDataSize = sizeof(ClockData)*params.numBlocksX*params.numBlocksY*params.blockSizeX*params.blockSizeY;
  
  if (params.dumpClocks) {
    ClockData *hostClock = new ClockData[params.blockSizeX*params.blockSizeY*params.numBlocksX*params.numBlocksY];

    memset(hostClock, 0, ClockDataSize);
    
    deviceClock = new cl::Buffer(context.context(), CL_MEM_WRITE_ONLY,
                                 ClockDataSize, NULL, &result);
    CLContext::throwOnError("Failed to allocate device output", result);

    result = queue.enqueueWriteBuffer(*deviceClock, CL_TRUE, 0,
                                      ClockDataSize, hostClock,
                                      NULL, NULL);
    CLContext::throwOnError("Failed to copy input data to device", result);

    delete hostClock;
  }

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
                         params.blockSizeY*params.numBlocksY);
  cl::NDRange localSize(params.blockSizeX, params.blockSizeY);


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
    ClockData *hostClock = new ClockData[params.blockSizeX*params.blockSizeY*params.numBlocksX*params.numBlocksY];
      
    result = queue.enqueueReadBuffer(*deviceClock, CL_TRUE, 0,
                                     ClockDataSize, hostClock,
                                     NULL, NULL);
    CLContext::throwOnError("Failed to copy result to host", result);

    unsigned NumThreads = params.blockSizeX*params.blockSizeY*params.numBlocksX*params.numBlocksY;

    std::cerr << "bx,by,warp,sm,clock0,clock1,clock2,clock3,clock4,\n";

    for (unsigned by = 0; by < params.numBlocksY; ++by) {
      for (unsigned bx = 0; bx < params.numBlocksX; ++bx) {
        for (unsigned w = 0; w < (params.blockSizeX*params.blockSizeY/32); ++w) {
          unsigned BlockId         = by*params.numBlocksX + bx;
          unsigned ThreadsPerBlock = params.blockSizeX*params.blockSizeY;
          unsigned ThreadId        = w * 32;
          unsigned GlobalId        = BlockId*ThreadsPerBlock + ThreadId;
          if (hostClock[GlobalId].SMId == 1) {
            std::cerr << bx
                      << "," << by
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

    for (unsigned i = 0; i < NumThreads; i += 32) {
      
    }
    
    delete hostClock;
  }

  printValue("Elapsed Time", elapsed);

  double gflops   = (double)params.realSize * (double)params.realSize
    * 5.0 * (double)params.timeSteps / elapsed / 1e9;
  //double gflops = stencilGen.computeGFlops(elapsed);
  printValue("Actual GFlop/s", gflops);

  gflops = (double)params.blockSizeX * (double)params.blockSizeY
    * (double)params.numBlocksX * (double)params.numBlocksY
    * (double)params.elementsPerThread * 5.0 * (double)params.timeSteps
    / elapsed / 1e9;

  printValue("Real Size", params.realSize);

  printValue("Device GFlop/s", gflops);

  if(vm.count("verify")) {
    compareResults(reference, hostData, params);
  }

  //printValue("Clock (Init)", hostClock[0]);
  //printValue("Clock (TS 0)", hostClock[1]);
  //printValue("Clock (TS 1+)", hostClock[2]);
  //printValue("Clock (Write)", hostClock[3]);
  //printValue("Clock (Total)", (hostClock[0] + hostClock[1] + hostClock[2] + hostClock[3]));

  if (params.dumpClocks) {
    delete deviceClock;
  }

  printValue("phase2_global_loads", 5.0);
  printValue("phase2_shared_loads", 0.0);
  printValue("compute_per_point", 5.0);
  printValue("phase3_shared_loads", 5.0);
  printValue("phase4_global_stores", 1.0);
  printValue("shared_stores", 1.0);
  printValue("num_fields", 1.0);
  printValue("data_size", 4.0);
  printValue("Dimensions", 2);

  
  // Clean-up
  delete [] hostData;

  if(vm.count("verify")) {
    delete [] reference;
  }

  return 0;
}
