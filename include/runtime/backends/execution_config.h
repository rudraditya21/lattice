#ifndef LATTICE_RUNTIME_BACKENDS_EXECUTION_CONFIG_H_
#define LATTICE_RUNTIME_BACKENDS_EXECUTION_CONFIG_H_

#include <string>

#include "runtime/backend.h"

namespace lattice::runtime {

ExecutionConfig LoadExecutionConfig(const std::string& prefix,
                                    ExecutionConfig base);

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_BACKENDS_EXECUTION_CONFIG_H_
