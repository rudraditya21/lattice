# Backend Guarantees (CPU)

- Alignment: allocations are 64-byte aligned by default; request a different alignment via `Allocate(bytes, alignment)`. CPU allocations are zero-initialized and scrubbed before free (best-effort).
- Thread-safety: `Backend` creation and singleton access are thread-safe. Streams can be created from multiple threads. Submitting work to a stream is thread-safe; each stream synchronizes its own tasks.
- Determinism: CPU backend executes tasks in the order submitted per stream. RNG builtins (`philox`, `threefry`, distribution samplers) are deterministic given the same seed/stream/counter. Parallel kernels may reorder floating-point operations; use a single-threaded stream for strict determinism.
- Unsupported/not implemented: no device memory pools, no NUMA pinning, no events with external waiting, no advanced scheduling/priorities, no device/host transfers beyond host allocations. Capability flags reflect current support (dense/sparse/ragged tensors, FFT/BLAS/conv are routed through CPU implementations only).

## Backend Selection
- `LATTICE_BACKEND=auto|cpu|opencl|cuda|hip|metal` selects a preferred backend; if unavailable, Lattice falls back to CPU. `auto` probes CUDA → HIP → Metal → OpenCL → CPU.
- `LATTICE_KERNEL_DIR=<path>` overrides the kernel search directory (defaults to `./OpenCL`, `./CUDA`, `./HIP`, or `./Metal` depending on backend).
- `LATTICE_CACHE_DIR=<path>` overrides the kernel cache directory (defaults to `./.lattice_cache`).
- Kernel cache is on by default (per-device, per-build-options). Metal uses an in-memory pipeline cache per run.
- `LATTICE_GPU_SMOKE_TEST=1` runs the vector-add smoke test for the selected GPU backend during `backend_tests`.
- All detected devices for the active backend are initialized and kept active.

### OpenCL
- `LATTICE_OPENCL_BUILD_OPTIONS=<opts>` appends OpenCL compiler options.
- `LATTICE_OPENCL_VERBOSE=1` logs OpenCL device info and build options.

### CUDA
- `LATTICE_CUDA_BUILD_OPTIONS=<opts>` appends NVRTC compiler options.
- `LATTICE_CUDA_VERBOSE=1` logs CUDA device info and build options.

### HIP
- `LATTICE_HIP_BUILD_OPTIONS=<opts>` appends HIPRTC compiler options.
- `LATTICE_HIP_ARCH=<gfx>` sets `--gpu-architecture` for HIPRTC (e.g. `gfx1030`).
- `LATTICE_HIP_VERBOSE=1` logs HIP device info and build options.

### Metal
- `LATTICE_METAL_VERBOSE=1` logs Metal device info.

## Kernel ABI
- OpenCL: `OpenCL/lattice_abi.h` and `include/runtime/backends/opencl_abi.h`.
- CUDA: `CUDA/lattice_abi.h` and `include/runtime/backends/cuda_abi.h`.
- HIP: `HIP/lattice_abi.h` and `include/runtime/backends/hip_abi.h`.
- Metal: `Metal/lattice_abi.h` and `include/runtime/backends/metal_abi.h`.
- Kernels should include `lattice_abi.h` and follow the fixed argument order expected by the host launcher.
