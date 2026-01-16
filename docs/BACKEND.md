# Backend Guarantees (CPU)

- Alignment: allocations are 64-byte aligned by default; request a different alignment via `Allocate(bytes, alignment)`. CPU allocations are zero-initialized and scrubbed before free (best-effort).
- Thread-safety: `Backend` creation and singleton access are thread-safe. Streams can be created from multiple threads. Submitting work to a stream is thread-safe; each stream synchronizes its own tasks.
- Determinism: CPU backend executes tasks in the order submitted per stream. RNG builtins (`philox`, `threefry`, distribution samplers) are deterministic given the same seed/stream/counter. Parallel kernels may reorder floating-point operations; use a single-threaded stream for strict determinism.
- Unsupported/not implemented: no device memory pools, no NUMA pinning, no events with external waiting, no advanced scheduling/priorities, no device/host transfers beyond host allocations. Capability flags reflect current support (dense/sparse/ragged tensors, FFT/BLAS/conv are routed through CPU implementations only).

## Stream/Queue Semantics
- Ordering: each `Stream` is FIFO. Submissions to the same stream are executed in order. There is no implicit ordering between different streams.
- Async behavior: `Submit` enqueues work and returns immediately. Work may run asynchronously relative to the host. `Stream::Synchronize()` waits for all previously submitted work on that stream to complete.
- Events: use `Stream::CreateEvent()` and `Stream::RecordEvent()` to capture completion of prior work on that stream. `AddDependency` inserts a wait so subsequent work in the stream happens after the event.
- GPU backends: CUDA/HIP/Metal/OpenCL enqueue kernels and copies asynchronously; `ExecutionConfig::sync_on_launch` forces a stream sync after each launch for debugging.
- Stream instances: `CreateStream()` creates a backend stream/queue when supported; on failure it falls back to the device default stream/queue.

## Profiling Hooks and Counters
- Enable timing and counters with `ExecutionConfig::enable_profiling` (per backend or via env config). When disabled, counters stay idle unless a hook is registered.
- `Backend::ProfilingStats()` returns aggregate counters for kernel launches, host↔device copy calls/bytes, stream syncs, and event record/wait calls with host-side elapsed time accumulation.
- `Backend::ResetProfilingStats()` clears counters; `Backend::SetProfilingHook()` registers a callback that receives `ProfilingEvent` records.
- Profiling durations are host wall-clock time for the backend API call (not GPU kernel execution time); use `ElapsedNs(start, end)` with GPU events for device-side timing.

## Backend Selection
- `LATTICE_BACKEND=auto|cpu|opencl|cuda|hip|metal` selects a preferred backend; if unavailable, Lattice falls back to CPU. `auto` probes CUDA → HIP → Metal → OpenCL → CPU.
- `LATTICE_KERNEL_DIR=<path>` overrides the kernel search directory (defaults to `./OpenCL`, `./CUDA`, `./HIP`, or `./Metal` depending on backend).
- `LATTICE_CACHE_DIR=<path>` overrides the kernel cache directory (defaults to `./.lattice_cache`).
- `LATTICE_CACHE_DISABLE=1` disables the persistent kernel cache (in-memory caches remain).
- `LATTICE_CACHE_MAX_BYTES=<bytes|K|M|G>` caps total cache size (default 512M).
- `LATTICE_CACHE_MAX_ENTRIES=<count>` caps number of cached binaries (default 4096).
- `LATTICE_CACHE_MAX_AGE_DAYS=<days>` evicts entries older than the age threshold (default 30).
- `LATTICE_CACHE_UPDATE_ATIME=0` disables access-time updates on cache hits.
- `LATTICE_REBUILD_ON_FAILURE=0` disables rebuild attempts after cached binaries fail to load/build (per-backend overrides: `LATTICE_OPENCL_REBUILD_ON_FAILURE`, `LATTICE_CUDA_REBUILD_ON_FAILURE`, `LATTICE_HIP_REBUILD_ON_FAILURE`).
- `LATTICE_BUILD_DEBUG=1` appends backend debug flags to kernel builds (OpenCL `-g -cl-opt-disable`, CUDA `-G`, HIP `-g -O0`).
- `LATTICE_SYNC_ON_LAUNCH=1` forces a stream/queue sync after each kernel launch.
- `LATTICE_ASYNC_LAUNCH=1` disables per-launch syncs (per-backend overrides: `LATTICE_OPENCL_SYNC_ON_LAUNCH`, `LATTICE_CUDA_SYNC_ON_LAUNCH`, `LATTICE_HIP_SYNC_ON_LAUNCH`, `LATTICE_METAL_SYNC_ON_LAUNCH`, and matching `_ASYNC_LAUNCH`).
- Kernel cache is on by default (per-device, per-build-options). Metal uses an in-memory pipeline cache per run.
- Device metadata is persisted under `LATTICE_CACHE_DIR/devices` as key/value text files keyed by a device fingerprint.
- `LATTICE_GPU_SMOKE_TEST=1` runs the vector-add smoke test for the selected GPU backend during `backend_tests`.
- All detected devices for the active backend are initialized and kept active.
- `LATTICE_DEVICE_BLACKLIST=<pattern[,pattern...]>` skips devices whose name/vendor/driver contains any pattern (case-insensitive).
- `LATTICE_DISABLE_SOFTWARE_DEVICES=1` disables devices tagged as software emulation in the quirk table.
- `LATTICE_IGNORE_DEVICE_QUIRKS=1` ignores the built-in quirk table (still applies `LATTICE_DEVICE_BLACKLIST`).
- Device selection envs are supported per backend (prefix with `LATTICE_OPENCL`, `LATTICE_CUDA`, `LATTICE_HIP`, `LATTICE_METAL`) or globally (prefix with `LATTICE`):
  - `_DEVICE_TYPE=cpu|gpu|accel|any` filters by device type (OpenCL only honors CPU/accel).
  - `_DEVICE_INCLUDE=<pattern[,pattern...]>` keeps devices matching any pattern (name/vendor/driver).
  - `_DEVICE_EXCLUDE=<pattern[,pattern...]>` drops devices matching any pattern.
  - `_DEVICE_INDICES=<list>` selects indices (e.g. `0,2-4`).
  - `_DEVICE_MASK=<bitmask>` selects indices by bitmask order (e.g. `1010`).
  - `_DEVICE_ORDER=<list>` reorders selected indices (e.g. `2,0`).
- Memory pool controls (device + pinned pools):
  - `LATTICE_DEVICE_POOL_DISABLE=1` disables device memory pooling (default enabled).
  - `LATTICE_DEVICE_POOL_MAX_BYTES=<bytes|K|M|G>` caps cached device bytes (default 256M).
  - `LATTICE_DEVICE_POOL_MAX_ENTRIES=<count>` caps cached device blocks (default 4096).
  - `LATTICE_DEVICE_POOL_MAX_ENTRY_BYTES=<bytes|K|M|G>` skips pooling for larger buffers (default 64M).
  - `LATTICE_DEVICE_POOL_BUCKET_BYTES=<bytes|K|M|G>` rounds device buffers to bucket size (default 256B).
  - `LATTICE_DEVICE_POOL_SCRUB=1` zeros device buffers on release.
  - `LATTICE_DEVICE_POOL_SECURE_SCRUB=1` enforces scrub-on-release (alias for secure scrub policy).
  - `LATTICE_DEVICE_POOL_ZERO_ON_ALLOC=1` zeros device buffers on allocation (fresh or reused).
  - `LATTICE_DEVICE_POOL_SCRUB_ON_ALLOC=1` zeros pooled device buffers on reuse.
  - `LATTICE_PINNED_POOL_DISABLE=1` disables pinned host pooling (default enabled).
  - `LATTICE_PINNED_POOL_MAX_BYTES=<bytes|K|M|G>` caps cached pinned bytes (default 128M).
  - `LATTICE_PINNED_POOL_MAX_ENTRIES=<count>` caps cached pinned blocks (default 2048).
  - `LATTICE_PINNED_POOL_MAX_ENTRY_BYTES=<bytes|K|M|G>` skips pooling for larger pinned buffers (default 32M).
  - `LATTICE_PINNED_POOL_BUCKET_BYTES=<bytes|K|M|G>` rounds pinned buffers to bucket size (default 256B).
  - `LATTICE_PINNED_POOL_SCRUB=1` zeros pinned buffers on release (default enabled).
  - `LATTICE_PINNED_POOL_SECURE_SCRUB=1` zeros pinned buffers on release using secure wiping.
  - `LATTICE_PINNED_POOL_ZERO_ON_ALLOC=1` zeros pinned buffers on allocation (fresh or reused).
  - `LATTICE_PINNED_POOL_SCRUB_ON_ALLOC=1` zeros pooled pinned buffers on reuse.
  - `LATTICE_PINNED_POOL_UNIFIED=1` attempts device-mapped unified pinned memory when supported.
  - Per-backend overrides: `LATTICE_OPENCL_DEVICE_POOL_*`, `LATTICE_CUDA_DEVICE_POOL_*`,
    `LATTICE_HIP_DEVICE_POOL_*`, `LATTICE_METAL_DEVICE_POOL_*` and `LATTICE_OPENCL_PINNED_POOL_*`,
    `LATTICE_CUDA_PINNED_POOL_*`, `LATTICE_HIP_PINNED_POOL_*`, `LATTICE_METAL_PINNED_POOL_*`,
    `LATTICE_CUDA_PINNED_POOL_UNIFIED`, `LATTICE_HIP_PINNED_POOL_UNIFIED`.
- Structured logging is available via:
  - `LATTICE_LOG_LEVEL=error|warn|info|debug|trace` (enables structured logs).
  - `LATTICE_LOG_FORMAT=text|json` (default `text`).
  - `LATTICE_LOG=1` enables info-level logs with default format.
- Kernel source tracing is available via:
  - `LATTICE_TRACE_KERNELS=1` to emit kernel sources/build options into a trace directory.
  - `LATTICE_TRACE_DIR=<path>` to override the trace directory (defaults to `LATTICE_CACHE_DIR/trace`).

### OpenCL
- `LATTICE_OPENCL_BUILD_OPTIONS=<opts>` appends OpenCL compiler options.
- `LATTICE_OPENCL_BUILD_DEBUG=1` appends OpenCL debug flags.
- `LATTICE_OPENCL_VERBOSE=1` logs OpenCL device info and build options.

### CUDA
- `LATTICE_CUDA_BUILD_OPTIONS=<opts>` appends NVRTC compiler options.
- `LATTICE_CUDA_BUILD_DEBUG=1` appends CUDA debug flags.
- `LATTICE_CUDA_VERBOSE=1` logs CUDA device info and build options.

### HIP
- `LATTICE_HIP_BUILD_OPTIONS=<opts>` appends HIPRTC compiler options.
- `LATTICE_HIP_ARCH=<gfx>` sets `--gpu-architecture` for HIPRTC (e.g. `gfx1030`).
- `LATTICE_HIP_BUILD_DEBUG=1` appends HIP debug flags.
- `LATTICE_HIP_VERBOSE=1` logs HIP device info and build options.

### Metal
- `LATTICE_METAL_VERBOSE=1` logs Metal device info.
- `LATTICE_METAL_BUILD_DEBUG=1` appends Metal debug flags (no compiler flags today, but affects cache keys).

## Kernel ABI
- OpenCL: `OpenCL/lattice_abi.h` and `include/runtime/backends/opencl_abi.h`.
- CUDA: `CUDA/lattice_abi.h` and `include/runtime/backends/cuda_abi.h`.
- HIP: `HIP/lattice_abi.h` and `include/runtime/backends/hip_abi.h`.
- Metal: `Metal/lattice_abi.h` and `include/runtime/backends/metal_abi.h`.
- Kernels should include `lattice_abi.h` and follow the fixed argument order expected by the host launcher.
- ABI versioning: `LATTICE_ABI_VERSION` is encoded as `0xMMMMmmmm` (major/minor). Compatibility rule: major must match and kernel version must be >= `LATTICE_ABI_VERSION_MIN`.
- Host build options inject `LATTICE_ABI_VERSION` and `LATTICE_ABI_VERSION_MIN`; kernel headers validate this at compile time.

### Kernel Argument Order & Alignment
- Order is fixed across backends: all input/output buffers first, then a single params struct passed by value (or by `constant` reference in Metal).
- Params structs are 8-byte aligned and have fixed sizes:
  - `ElemwiseParams`=280 bytes (includes broadcast metadata with `kMaxTensorDims=8`).
  - `ReduceParams`=24 bytes.
  - `MatmulParams`=56 bytes.
  - `TransposeParams`=16 bytes.
  - `Conv2dParams`=48 bytes.
  - `Pool2dParams`=48 bytes.
  - `FftParams`=8 bytes.
  - `SolveParams`=16 bytes.
  - `LuParams`=8 bytes.
  - `QrParams`=16 bytes.
  - `SvdParams`=16 bytes.
  - `QuantileParams`=16 bytes.
  - `CorrelationParams`=8 bytes.
  - `RegressionParams`=8 bytes.
- OpenCL: `__kernel void op(__global T* in0, ..., lattice_*_params_t params)`.
- CUDA/HIP: `extern "C" __global__ void op(const T* in0, ..., lattice_*_params_t params)`.
- Metal: `kernel void op(const device T* in0 [[buffer(0)]], ..., constant lattice_*_params_t& params [[buffer(N)]])`.
- Feature macros: `LATTICE_HAS_FP16` / `LATTICE_HAS_FP64` are defined for CUDA/HIP/Metal when supported (OpenCL uses the same macros via extensions).
