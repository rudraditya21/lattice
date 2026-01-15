#include "runtime/backends/memory_utils.h"

#include <cstring>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__) || defined(__FreeBSD__)
#include <strings.h>
#endif

namespace lattice::runtime {

void SecureZero(void* ptr, size_t bytes) {
    if (!ptr || bytes == 0) {
        return;
    }
#if defined(_WIN32)
    SecureZeroMemory(ptr, bytes);
#elif defined(__STDC_LIB_EXT1__)
    memset_s(ptr, bytes, 0, bytes);
#elif defined(__GLIBC__) || defined(__FreeBSD__)
    explicit_bzero(ptr, bytes);
#else
    volatile unsigned char* p = static_cast<volatile unsigned char*>(ptr);
    while (bytes--) {
        *p++ = 0;
    }
#endif
}

}  // namespace lattice::runtime
