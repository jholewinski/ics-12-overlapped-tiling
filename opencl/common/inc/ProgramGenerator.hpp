
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

};

}

#endif
