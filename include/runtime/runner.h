#ifndef LATTICE_RUNTIME_RUNNER_H_
#define LATTICE_RUNTIME_RUNNER_H_

#include <memory>
#include <string>

#include "runtime/ops.h"

namespace lattice::runtime {

/// Executes a source string as a program (sequence of statements) and returns the resulting signal.
ExecResult RunSource(const std::string& source, std::shared_ptr<Environment> env);

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_RUNNER_H_
