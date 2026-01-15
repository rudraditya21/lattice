#ifndef LATTICE_RUNTIME_BACKENDS_MEMORY_UTILS_H_
#define LATTICE_RUNTIME_BACKENDS_MEMORY_UTILS_H_

#include <cstddef>

namespace lattice::runtime {

void SecureZero(void* ptr, size_t bytes);

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_BACKENDS_MEMORY_UTILS_H_
