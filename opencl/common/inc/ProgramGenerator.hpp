
#ifndef PROGRAM_GENERATOR_HPP_INC
#define PROGRAM_GENERATOR_HPP_INC 1

namespace ot {

/**
 * Base class for all generators.
 */
class ProgramGenerator {
public:

  ProgramGenerator();

  virtual ~ProgramGenerator();

  std::string getProgram();

  template<typename T>
  static void printProgramParameters(const T& params, int32_t numArrays,
                              int32_t numLoadsPerPt, int32_t numStoresPerPt,
                              int32_t flopsPerPt) {
    int32_t sharedSize = params.sharedSizeX * params.sharedSizeY *
                         numArrays * 4;

    int64_t totalFPPerBlock = params.blockSizeX * params.blockSizeY *
      params.elementsPerThread * params.timeTileSize * flopsPerPt;

    int64_t usefulFPPerBlock = params.realPerBlockX * params.realPerBlockY *
      params.timeTileSize * flopsPerPt;

    double usefulFPRatio = (double)usefulFPPerBlock / (double)totalFPPerBlock;

    int32_t globalLoadsPerBlock = params.blockSizeX * params.blockSizeY *
      params.elementsPerThread * numLoadsPerPt;

    int32_t globalStoresPerBlock = params.blockSizeX * params.blockSizeY *
      params.elementsPerThread * numStoresPerPt;

    int32_t sharedLoadsPerBlock = params.blockSizeX * params.blockSizeY *
      params.elementsPerThread * numLoadsPerPt * (params.timeTileSize-1);

    int32_t sharedStoresPerBlock = params.blockSizeX * params.blockSizeY *
      params.elementsPerThread * numStoresPerPt * (params.timeTileSize-1);

    printValue("Time Tile Size", params.timeTileSize);
    printValue("Block Size X", params.blockSizeX);
    printValue("Block Size Y", params.blockSizeY);
    printValue("Num Blocks X", params.blockSizeX);
    printValue("Num Blocks Y", params.blockSizeY);
    printValue("Elements/Thread", params.elementsPerThread);
    printValue("Shared Size", sharedSize);
    printValue("Total FP/Block", totalFPPerBlock);
    printValue("Useful FP/Block", usefulFPPerBlock);
    printValue("Useful Ratio", usefulFPRatio);
    printValue("Global Loads/Block", globalLoadsPerBlock);
    printValue("Global Stores/Block", globalStoresPerBlock);
    printValue("Shared Loads/Block", sharedLoadsPerBlock);
    printValue("Shared Stores/Block", sharedStoresPerBlock);
  }

};

}

#endif
