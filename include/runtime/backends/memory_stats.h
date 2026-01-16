#ifndef LATTICE_RUNTIME_BACKENDS_MEMORY_STATS_H_
#define LATTICE_RUNTIME_BACKENDS_MEMORY_STATS_H_

#include <sstream>
#include <string>

#include "runtime/backend.h"

namespace lattice::runtime {

inline bool HasOutstandingAllocs(const MemoryPoolStats& stats) {
    return stats.in_use_blocks > 0 || stats.in_use_bytes > 0;
}

inline std::string FormatPoolStats(const MemoryPoolStats& stats) {
    std::ostringstream ss;
    ss << stats.in_use_blocks << " blocks, " << stats.in_use_bytes << " bytes";
    return ss.str();
}

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_BACKENDS_MEMORY_STATS_H_
