# Backend Guarantees (CPU)

- Alignment: allocations are 64-byte aligned by default; request a different alignment via `Allocate(bytes, alignment)`. CPU allocations are zero-initialized and scrubbed before free (best-effort).
- Thread-safety: `Backend` creation and singleton access are thread-safe. Streams can be created from multiple threads. Submitting work to a stream is thread-safe; each stream synchronizes its own tasks.
- Determinism: CPU backend executes tasks in the order submitted per stream. RNG builtins (`philox`, `threefry`, distribution samplers) are deterministic given the same seed/stream/counter. Parallel kernels may reorder floating-point operations; use a single-threaded stream for strict determinism.
- Unsupported/not implemented: no device memory pools, no NUMA pinning, no events with external waiting, no advanced scheduling/priorities, no device/host transfers beyond host allocations. Capability flags reflect current support (dense/sparse/ragged tensors, FFT/BLAS/conv are routed through CPU implementations only).
