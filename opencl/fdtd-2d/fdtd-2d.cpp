
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

  int32_t phaseLimit;


  /**
   * Default constructor.
   */
  GeneratorParams(int32_t tts      = 1,
                  int32_t ept      = 1,
                  int32_t bsx      = 16,
                  int32_t bsy      = 16,
                  int32_t ps       = 1024,
                  int32_t ts       = 64,
                  std::string type = "float")
    : timeTileSize(tts),
      timeSteps(ts),
      elementsPerThread(ept),
      problemSize(ps),
      dataType(type),
      blockSizeX(bsx),
      blockSizeY(bsy),
      phaseLimit(0) {
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

    if(dataType == "float") {
      fpSuffix = "f";
    } else {
      fpSuffix = "";
    }

    if(padding < 1 || compsPerBlockX < 1 || compsPerBlockY < 1               ||
       realPerBlockX < 1 || realPerBlockY < 1 || sizeLCM < 1 || realSize < 1 ||
       numBlocksX < 1 || numBlocksY < 1 || sharedSizeX < 1                   ||
       sharedSizeY < 1 || paddedSize < 1) {
      throw std::runtime_error("Consistency error!");
    }
  }
};

/**
 * Generator for Jacobi 2D.
 */
class FDTD2DGenerator : public ProgramGenerator {
public:

  FDTD2DGenerator();

  virtual ~FDTD2DGenerator();

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


FDTD2DGenerator::FDTD2DGenerator() {
}

FDTD2DGenerator::~FDTD2DGenerator() {
}

std::string FDTD2DGenerator::generate(GeneratorParams& params) {
  std::stringstream program;

  params.computeDerived();

  generateHeader(program, params);
  generateLocals(program, params);
  generateCompute(program, params);
  generateFooter(program);

  return program.str();
}

void FDTD2DGenerator::generateHeader(std::ostream& stream,
                                     const GeneratorParams& params) {
  stream << "/* Auto-generated.  Do not edit by hand. */\n";
  stream << "__kernel\n";
  stream << "void kernel_func(__global " << params.dataType << "* inputEX,\n";
  stream << "                 __global " << params.dataType << "* inputEY,\n";
  stream << "                 __global " << params.dataType << "* inputHZ,\n";
  stream << "                 __global " << params.dataType << "* outputEX,\n";
  stream << "                 __global " << params.dataType << "* outputEY,\n";
  stream << "                 __global " << params.dataType << "* outputHZ) {\n";
}

void FDTD2DGenerator::generateFooter(std::ostream& stream) {
  stream << "}\n\n";
}

void FDTD2DGenerator::generateLocals(std::ostream& stream,
                                     const GeneratorParams& params) {
  stream << "  __local " << params.dataType << " bufferEX[" << params.sharedSizeY
         << "][" << params.sharedSizeX << "];\n";
  stream << "  __local " << params.dataType << " bufferEY[" << params.sharedSizeY
         << "][" << params.sharedSizeX << "];\n";
  stream << "  __local " << params.dataType << " bufferHZ[" << params.sharedSizeY
         << "][" << params.sharedSizeX << "];\n";


  // Compute some pointer values
  stream << "  __global " << params.dataType
         << "* inputEXPtr = inputEX + ((get_group_id(1)*" << params.realPerBlockY
         << "+get_local_id(1)*" << params.elementsPerThread << "+1)*"
         << params.paddedSize << ") + (get_group_id(0)*" << params.realPerBlockX
         << ") + get_local_id(0) + 1;\n";

  stream << "  __global " << params.dataType
         << "* inputEYPtr = inputEY + ((get_group_id(1)*" << params.realPerBlockY
         << "+get_local_id(1)*" << params.elementsPerThread << "+1)*"
         << params.paddedSize << ") + (get_group_id(0)*" << params.realPerBlockX
         << ") + get_local_id(0) + 1;\n";

  stream << "  __global " << params.dataType
         << "* inputHZPtr = inputHZ + ((get_group_id(1)*" << params.realPerBlockY
         << "+get_local_id(1)*" << params.elementsPerThread << "+1)*"
         << params.paddedSize << ") + (get_group_id(0)*" << params.realPerBlockX
         << ") + get_local_id(0) + 1;\n";

  stream << "  __global " << params.dataType
         << "* outputEXPtr = outputEX + ((get_group_id(1)*" << params.realPerBlockY
         << "+get_local_id(1)*" << params.elementsPerThread << "+1)*"
         << params.paddedSize << ") + (get_group_id(0)*" << params.realPerBlockX
         << ") + get_local_id(0) + 1;\n";

  stream << "  __global " << params.dataType
         << "* outputEYPtr = outputEY + ((get_group_id(1)*" << params.realPerBlockY
         << "+get_local_id(1)*" << params.elementsPerThread << "+1)*"
         << params.paddedSize << ") + (get_group_id(0)*" << params.realPerBlockX
         << ") + get_local_id(0) + 1;\n";

  stream << "  __global " << params.dataType
         << "* outputHZPtr = outputHZ + ((get_group_id(1)*" << params.realPerBlockY
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

  // Declare local intermediates
  for(int32_t i = 0; i < params.elementsPerThread; ++i) {
    stream << "  " << params.dataType << " localEX" << i << ";\n";
    stream << "  " << params.dataType << " newEX" << i << ";\n";
    stream << "  " << params.dataType << " localEY" << i << ";\n";
    stream << "  " << params.dataType << " newEY" << i << ";\n";
    stream << "  " << params.dataType << " localHZ" << i << ";\n";
    stream << "  " << params.dataType << " newHZ" << i << ";\n";
  }
}

void FDTD2DGenerator::generateCompute(std::ostream& stream,
                                      const GeneratorParams& params) {

  for(int32_t i = 0; i < params.elementsPerThread; ++i) {
    stream << "  {\n";
    stream << "    " << params.dataType
           << " eyCenter, hzCenter, hzTop, hzLeft, exCenter;\n";
    stream << "    eyCenter = *(inputEYPtr+(" << params.paddedSize << "*" << i
           << "));\n";
    stream << "    exCenter = *(inputEXPtr+(" << params.paddedSize << "*" << i
           << "));\n";
    stream << "    hzCenter = *(inputHZPtr+(" << params.paddedSize << "*" << i
           << "));\n";
    stream << "    hzTop = *(inputHZPtr+(" << params.paddedSize << "*" << (i-1)
           << "));\n";
    stream << "    hzLeft = *(inputHZPtr+(" << params.paddedSize << "*" << i
           << ")-1);\n";
    stream << "    " << params.dataType
           << " resultEY = eyCenter - 0.5" << params.fpSuffix
           << "*(hzCenter-hzTop);\n";
    stream << "    resultEY = (valid" << i << ") ? resultEY : 0.0"
           << params.fpSuffix << ";\n";
    stream << "    bufferEY[get_local_id(1)*" << params.elementsPerThread << "+"
           << i
           << "+1][get_local_id(0)+1] = resultEY;\n";
    stream << "    localEY" << i << " = resultEY;\n";
        stream << "    " << params.dataType
           << " resultEX = exCenter - 0.5" << params.fpSuffix
           << "*(hzCenter-hzLeft);\n";
    stream << "    resultEX = (valid" << i << ") ? resultEX : 0.0"
           << params.fpSuffix << ";\n";
    stream << "    bufferEX[get_local_id(1)*" << params.elementsPerThread << "+"
           << i
           << "+1][get_local_id(0)+1] = resultEX;\n";
    stream << "    localEX" << i << " = resultEX;\n";
    stream << "  }\n";
  }
  stream << "  barrier(CLK_LOCAL_MEM_FENCE);\n";
  for(int32_t i = 0; i < params.elementsPerThread; ++i) {
    stream << "  {\n";
    stream << "    " << params.dataType
           << " hzCenter, exCenter, exRight, eyBottom, eyCenter;\n";
    stream << "    hzCenter = *(inputHZPtr+(" << params.paddedSize << "*" << i
           << "));\n";
    stream << "    exCenter = bufferEX[get_local_id(1)*"
           << params.elementsPerThread << "+" << i
           << "+1][get_local_id(0)+1];\n";
    stream << "    exRight = bufferEX[get_local_id(1)*"
           << params.elementsPerThread << "+" << i
           << "+1][get_local_id(0)+2];\n";
    stream << "    eyCenter = bufferEY[get_local_id(1)*"
           << params.elementsPerThread << "+" << i
           << "+1][get_local_id(0)+1];\n";
    stream << "    eyBottom = bufferEY[get_local_id(1)*"
           << params.elementsPerThread << "+" << i
           << "+2][get_local_id(0)+1];\n";
    stream << "    " << params.dataType
           << " resultHZ = hzCenter - 0.7" << params.fpSuffix
           << "*(exRight-exCenter+eyBottom+eyCenter);\n";
    stream << "    resultHZ = (valid" << i << ") ? resultHZ : 0.0"
           << params.fpSuffix << ";\n";
    stream << "    bufferHZ[get_local_id(1)*" << params.elementsPerThread << "+"
           << i
           << "+1][get_local_id(0)+1] = resultHZ;\n";
    stream << "    localHZ" << i << " = resultHZ;\n";
    stream << "  }\n";
  }

  stream << "  barrier(CLK_LOCAL_MEM_FENCE);\n";

  if(params.phaseLimit == 2) {
    stream << "  if(get_local_id(0) != (unsigned)(-1)) { return; }\n";
  }
  
  for(int32_t t = 1; t < params.timeTileSize; ++t) {
    stream << "  // Time Step " << t << "\n";
    for(int32_t i = 0; i < params.elementsPerThread; ++i) {
      stream << "  {\n";
      stream << "    " << params.dataType
             << " eyCenter, hzCenter, hzTop, hzLeft, exCenter;\n";
      stream << "    eyCenter = bufferEY[get_local_id(1)*"
             << params.elementsPerThread << "+" << i
             << "+1][get_local_id(0)+1];\n";
      stream << "    hzCenter = bufferHZ[get_local_id(1)*"
             << params.elementsPerThread << "+" << i
             << "+1][get_local_id(0)+1];\n";
      stream << "    hzTop = bufferHZ[get_local_id(1)*"
             << params.elementsPerThread << "+" << i
             << "][get_local_id(0)+1];\n";
      stream << "    hzLeft = bufferHZ[get_local_id(1)*"
             << params.elementsPerThread << "+" << i
             << "+1][get_local_id(0)];\n";
      stream << "    exCenter = bufferEX[get_local_id(1)*"
             << params.elementsPerThread << "+" << i
             << "+1][get_local_id(0)+1];\n";
      stream << "    " << params.dataType
             << " resultEY = eyCenter - 0.5" << params.fpSuffix
             << "*(hzCenter-hzTop);\n";
      stream << "    resultEY = (valid" << i << ") ? resultEY : 0.0"
             << params.fpSuffix << ";\n";
      stream << "    newEY" << i << " = resultEY;\n";
      stream << "    " << params.dataType
             << " resultEX = exCenter - 0.5" << params.fpSuffix
             << "*(hzCenter-hzLeft);\n";
      stream << "    resultEX = (valid" << i << ") ? resultEX : 0.0"
             << params.fpSuffix << ";\n";
      stream << "    newEX" << i << " = resultEX;\n";
      stream << "  }\n";
    }
    stream << "  barrier(CLK_LOCAL_MEM_FENCE);\n";
    for(int32_t i = 0; i < params.elementsPerThread; ++i) {
      stream << "  bufferEX[get_local_id(1)*" << params.elementsPerThread << "+"
             << i
             << "+1][get_local_id(0)+1] = newEX" << i << ";\n";
      stream << "  localEX" << i << " = newEX" << i << ";\n";
      stream << "  bufferEY[get_local_id(1)*" << params.elementsPerThread << "+"
             << i
             << "+1][get_local_id(0)+1] = newEY" << i << ";\n";
      stream << "  localEY" << i << " = newEY" << i << ";\n";
    }
    stream << "  barrier(CLK_LOCAL_MEM_FENCE);\n";
    for(int32_t i = 0; i < params.elementsPerThread; ++i) {
      stream << "  {\n";
      stream << "    " << params.dataType
             << " hzCenter, exCenter, exRight, eyBottom, eyCenter;\n";
      stream << "    hzCenter = *(inputHZPtr+(" << params.paddedSize << "*" << i
             << "));\n";
      stream << "    exCenter = bufferEX[get_local_id(1)*"
             << params.elementsPerThread << "+" << i
             << "+1][get_local_id(0)+1];\n";
      stream << "    exRight = bufferEX[get_local_id(1)*"
             << params.elementsPerThread << "+" << i
             << "+1][get_local_id(0)+2];\n";
      stream << "    eyCenter = bufferEY[get_local_id(1)*"
             << params.elementsPerThread << "+" << i
             << "+1][get_local_id(0)+1];\n";
      stream << "    eyBottom = bufferEY[get_local_id(1)*"
             << params.elementsPerThread << "+" << i
             << "+2][get_local_id(0)+1];\n";
      stream << "    " << params.dataType
             << " resultHZ = hzCenter - 0.7" << params.fpSuffix
             << "*(exRight-exCenter+eyBottom+eyCenter);\n";
      stream << "    resultHZ = (valid" << i << ") ? resultHZ : 0.0"
             << params.fpSuffix << ";\n";
      stream << "  }\n";
    }
    stream << "  barrier(CLK_LOCAL_MEM_FENCE);\n";
    for(int32_t i = 0; i < params.elementsPerThread; ++i) {
      stream << "  bufferHZ[get_local_id(1)*" << params.elementsPerThread << "+"
             << i
             << "+1][get_local_id(0)+1] = newHZ" << i << ";\n";
      stream << "  localHZ" << i << " = newHZ" << i << ";\n";
    }
    stream << "  barrier(CLK_LOCAL_MEM_FENCE);\n";
  }

  if(params.phaseLimit == 3) {
    stream << "  if(get_local_id(0) != (unsigned)(-1)) { return; }\n";
  }
  
  for(int32_t i = 0; i < params.elementsPerThread; ++i) {
    stream << "  if(writeValid" << i << " && writeValidX) {\n";
    stream << "    *(outputHZPtr+(" << params.paddedSize << "*" << i
           << ")) = localHZ" << i << ";\n";
    stream << "  }\n";
    stream << "  if(writeValid" << i << " && writeValidX) {\n";
    stream << "    *(outputEXPtr+(" << params.paddedSize << "*" << i
           << ")) = localEX" << i << ";\n";
    stream << "  }\n";
    stream << "  if(writeValid" << i << " && writeValidX) {\n";
    stream << "    *(outputEYPtr+(" << params.paddedSize << "*" << i
           << ")) = localEY" << i << ";\n";
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

  FDTD2DGenerator gen;
  GeneratorParams params;

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

  // printValue("Problem Size", params.problemSize);
  // printValue("Time Tile Size", params.timeTileSize);
  // printValue("Padded Size", params.paddedSize);
  // printValue("Block Size X", params.blockSizeX);
  // printValue("Block Size Y", params.blockSizeY);
  // printValue("Elements/Thread", params.elementsPerThread);
  // printValue("Num Blocks X", params.numBlocksX);
  // printValue("Num Blocks Y", params.numBlocksY);
  // printValue("Time Steps", params.timeSteps);
  // printValue("Padding", params.padding);
  // printValue("Real Size", params.realSize);

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
  // printValue("Global Memory Size", globalMemorySize);
  // printValue("Local Memory Size", localMemorySize);
  // printValue("Max Compute Units", maxComputeUnits);
  // printValue("Max Work-Group Size", maxWorkGroupSize);


  if(params.blockSizeX*params.blockSizeY > maxWorkGroupSize) {
    std::cout << "ERROR: Block dimensions are too large!\n";
    return 1;
  }

  if(3*params.sharedSizeX*params.sharedSizeY > localMemorySize) {
    std::cout << "ERROR: Not enough local memory for even one block!\n";
    return 1;
  }

// Print some derived statistics
  int32_t sharedSize = params.sharedSizeX * params.sharedSizeY * 3 * 4;

  int32_t numBlocksFromShared = (int32_t)std::ceil((double)localMemorySize /
                                                   (double)sharedSize);

  int64_t totalFPPerBlock = params.blockSizeX * params.blockSizeY *
    params.elementsPerThread * params.timeSteps * 11;

  int64_t usefulFPPerBlock = 11 * params.realPerBlockX * params.realPerBlockY*
    params.timeSteps;

  double usefulFPRatio = (double)usefulFPPerBlock /
    (double)totalFPPerBlock;

  int32_t globalLoadsPerBlock = params.blockSizeX * params.blockSizeY *
    params.elementsPerThread * 6;

  int32_t globalStoresPerBlock = params.blockSizeX * params.blockSizeY *
    params.elementsPerThread * 3;

  int32_t sharedLoadsPerBlock = params.blockSizeX * params.blockSizeY *
    params.elementsPerThread * (4 + 9 * (params.timeTileSize-1));

  int32_t sharedStoresPerBlock = params.blockSizeX * params.blockSizeY *
    params.elementsPerThread * (2 + 3 * (params.timeTileSize-1));

  int32_t arithmeticIntensity = 11.0 / 6.0;

  int32_t maxBlocks = 8;        // TODO: Change based on arch.

  // printValue("Shared Size", sharedSize);
  // printValue("Num Blocks (Shared)", numBlocksFromShared);
  // printValue("Total FP", totalFPPerBlock);
  // printValue("Useful FP", usefulFPPerBlock);
  // printValue("Useful Ratio", usefulFPRatio);
  // printValue("Global Loads/Block", globalLoadsPerBlock);
  // printValue("Global Stores/Block", globalStoresPerBlock);
  // printValue("Shared Loads/Block", sharedLoadsPerBlock);
  // printValue("Shared Stores/Block", sharedStoresPerBlock);
  // printValue("Arithmetic Intensity", arithmeticIntensity);
  // printValue("Max Blocks", maxBlocks);

  ProgramGenerator::printProgramParameters(params, 3, 8, 3, 11);

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

  result = program.build(devices, "-cl-nv-verbose");
  if(result != CL_SUCCESS) {
    std::cout << "Source compilation failed.\n";
    std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.device());
    return 1;
  }

  // Extract out the register usage
  std::string log =
    program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.device());
  boost::regex regExpr("Used ([0-9]+) registers");
  boost::smatch match;
  std::string::const_iterator start, end;
  start = log.begin();
  end = log.end();
  if(boost::regex_search(start, end, match, regExpr,
                         boost::match_default)) {
    printValue("Register Usage", match[1]);
  } else {
    printValue("Register Usage", 0);
  }

  // Extract the kernel
  cl::Kernel kernel(program, "kernel_func", &result);
  CLContext::throwOnError("Failed to extract kernel", result);


  // Allocate host arrays
  float* hostDataEX = new float[arraySize];
  float* hostDataEY = new float[arraySize];
  float* hostDataHZ = new float[arraySize];

  // Fill host arrays
  for(int i = 0; i < params.paddedSize; ++i) {
    for(int j = 0; j < params.paddedSize; ++j) {
      if(i < params.padding || i >= (params.paddedSize-params.padding) ||
         j < params.padding || j         >= (params.paddedSize-params.padding)) {
        hostDataEX[i*params.paddedSize +j]  = 0.0f;
        hostDataEY[i*params.paddedSize +j]  = 0.0f;
        hostDataHZ[i*params.paddedSize +j]  = 0.0f;
      }
      else {
        hostDataEX[i*params.paddedSize + j] = (float)rand() / ((float)RAND_MAX + 1.0f);
        hostDataEY[i*params.paddedSize + j] = (float)rand() / ((float)RAND_MAX + 1.0f);
        hostDataHZ[i*params.paddedSize + j] = (float)rand() / ((float)RAND_MAX + 1.0f);
      }
    }
  }


  // Compute reference

  float* reference = NULL;

  if(vm.count("verify")) {

    /*reference = new float[arraySize];

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
    delete [] refB;*/

  }


  // Allocate device arrays
  cl::Buffer deviceInputEX(context.context(), CL_MEM_READ_WRITE,
                           arraySize, NULL, &result);
  CLContext::throwOnError("Failed to allocate device input", result);
  cl::Buffer deviceInputEY(context.context(), CL_MEM_READ_WRITE,
                           arraySize, NULL, &result);
  CLContext::throwOnError("Failed to allocate device input", result);
  cl::Buffer deviceInputHZ(context.context(), CL_MEM_READ_WRITE,
                           arraySize, NULL, &result);
  CLContext::throwOnError("Failed to allocate device input", result);

  cl::Buffer deviceOutputEX(context.context(), CL_MEM_READ_WRITE,
                            arraySize, NULL, &result);
  CLContext::throwOnError("Failed to allocate device output", result);
  cl::Buffer deviceOutputEY(context.context(), CL_MEM_READ_WRITE,
                            arraySize, NULL, &result);
  CLContext::throwOnError("Failed to allocate device output", result);
  cl::Buffer deviceOutputHZ(context.context(), CL_MEM_READ_WRITE,
                            arraySize, NULL, &result);
  CLContext::throwOnError("Failed to allocate device output", result);


  // Copy host data to device
  result = queue.enqueueWriteBuffer(deviceInputEX, CL_TRUE, 0,
                                    arraySize, hostDataEX,
                                    NULL, NULL);
  CLContext::throwOnError("Failed to copy input data to device", result);
  result = queue.enqueueWriteBuffer(deviceInputEY, CL_TRUE, 0,
                                    arraySize, hostDataEY,
                                    NULL, NULL);
  CLContext::throwOnError("Failed to copy input data to device", result);
  result = queue.enqueueWriteBuffer(deviceInputHZ, CL_TRUE, 0,
                                    arraySize, hostDataHZ,
                                    NULL, NULL);
  CLContext::throwOnError("Failed to copy input data to device", result);

  result = queue.enqueueWriteBuffer(deviceOutputEX, CL_TRUE, 0,
                                    arraySize, hostDataEX,
                                    NULL, NULL);
  CLContext::throwOnError("Failed to copy input data to device", result);
  result = queue.enqueueWriteBuffer(deviceOutputEY, CL_TRUE, 0,
                                    arraySize, hostDataEY,
                                    NULL, NULL);
  CLContext::throwOnError("Failed to copy input data to device", result);
  result = queue.enqueueWriteBuffer(deviceOutputHZ, CL_TRUE, 0,
                                    arraySize, hostDataHZ,
                                    NULL, NULL);
  CLContext::throwOnError("Failed to copy input data to device", result);

  cl::NDRange globalSize(params.blockSizeX*params.numBlocksX,
                         params.blockSizeY*params.numBlocksY);
  cl::NDRange localSize(params.blockSizeX, params.blockSizeY);


  cl::Buffer* inputBufferEX;
  cl::Buffer* inputBufferEY;
  cl::Buffer* inputBufferHZ;
  cl::Buffer* outputBufferEX;
  cl::Buffer* outputBufferEY;
  cl::Buffer* outputBufferHZ;

  inputBufferEX  = &deviceInputEX;
  inputBufferEY  = &deviceInputEY;
  inputBufferHZ  = &deviceInputHZ;
  outputBufferEX = &deviceOutputEX;
  outputBufferEY = &deviceOutputEY;
  outputBufferHZ = &deviceOutputHZ;

  cl::Event waitEvent;

  double startTime = rtclock();

  for(int t = 0; t < params.timeSteps / params.timeTileSize; ++t) {

    // Set kernel arguments
    result = kernel.setArg(0, *inputBufferEX);
    CLContext::throwOnError("Failed to set input parameter", result);
    result = kernel.setArg(1, *inputBufferEY);
    CLContext::throwOnError("Failed to set input parameter", result);
    result = kernel.setArg(2, *inputBufferHZ);
    CLContext::throwOnError("Failed to set input parameter", result);
    result = kernel.setArg(3, *outputBufferEX);
    CLContext::throwOnError("Failed to set output parameter", result);
    result = kernel.setArg(4, *outputBufferEY);
    CLContext::throwOnError("Failed to set output parameter", result);
    result = kernel.setArg(5, *outputBufferHZ);
    CLContext::throwOnError("Failed to set output parameter", result);

    // Invoke the kernel
    result = queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                        globalSize, localSize,
                                        0, &waitEvent);
    CLContext::throwOnError("Kernel launch failed", result);

    std::swap(inputBufferEX, outputBufferEX);
    std::swap(inputBufferEY, outputBufferEY);
    std::swap(inputBufferHZ, outputBufferHZ);
  }

  waitEvent.wait();

  double endTime = rtclock();
  double elapsed = endTime - startTime;

  // Copy results back to host
  result = queue.enqueueReadBuffer(*inputBufferEX, CL_TRUE, 0,
                                   arraySize, hostDataEX,
                                   NULL, NULL);
  CLContext::throwOnError("Failed to copy result to host", result);
  result = queue.enqueueReadBuffer(*inputBufferEY, CL_TRUE, 0,
                                   arraySize, hostDataEY,
                                   NULL, NULL);
  CLContext::throwOnError("Failed to copy result to host", result);
  result = queue.enqueueReadBuffer(*inputBufferHZ, CL_TRUE, 0,
                                   arraySize, hostDataHZ,
                                   NULL, NULL);
  CLContext::throwOnError("Failed to copy result to host", result);

  printValue("Elapsed Time", elapsed);

  double gflops   = (double)params.realSize * (double)params.realSize
    * 11.0 * (double)params.timeSteps / elapsed / 1e9;
  //double gflops = stencilGen.computeGFlops(elapsed);
  printValue("Actual GFlop/s", gflops);

  gflops = (double)params.blockSizeX * (double)params.blockSizeY
    * (double)params.numBlocksX * (double)params.numBlocksY
    * (double)params.elementsPerThread * 11.0 * (double)params.timeSteps
    / elapsed / 1e9;

  printValue("Device GFlop/s", gflops);

  if(vm.count("verify")) {
    //compareResults(reference, hostData, params);
  }



  // Clean-up
  delete [] hostDataEX;
  delete [] hostDataEY;
  delete [] hostDataHZ;

  if(vm.count("verify")) {
    delete [] reference;
  }

  return 0;
}
