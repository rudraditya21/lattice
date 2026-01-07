# Lattice Language Overview

## Goals
- Scientific computing and statistics with reproducibility and determinism as defaults.
- Numeric rigor: full scalar tower (ints, floats, complex, decimal, rational), tensors/arrays, and typed builtins.
- Gradual typing: optional annotations enforced at boundaries; dynamic when omitted.
- Multi-backend ready: CPU reference backend today with alloc/stream/event scaffolding and capability
  flags. GPU backends (OpenCL/CUDA/HIP/Metal) use in-house kernels rather than external BLAS/FFT libraries.

## Syntax (C-like, expression-oriented)
- Statements end by newline or `;`. Blocks with `{ ... }`.
- Functions: `func name(a: i32, b) -> f64 { return a + b; }`.
- Variables: `x = 3`, annotated `x: i32 = 3`.
- Control flow: `if (cond) { ... } else { ... }`, `while`, `for (init; cond; step)`, `break`, `continue`, `return`.
- Tuples: `(a, b)`; singleton `(a,)`; indexing `t[0]`. Tuples are the only literal collection.
- Records: `{x: 1, y: 2}`; access `r["x"]`. Square-bracket list literals are not supported.
- Comments: `// comment`.

## Types
- Scalars: `i8/i16/i32/i64`, `u8/u16/u32/u64`, `f16/bfloat16/f32/f64`, `complex64/complex128`, `decimal`, `rational`, `bool`, `string`.
- Aggregates: tuples, records, tensors (dense, sparse CSR/COO, ragged).
- Optional annotations on params/returns/lets; enforced when present; implicit promotions follow numeric tower (complex > float > int; decimal/rational isolated).

## Tensors
- Creation: `tensor(d0, d1, fill)`, `tensor_values((1,2,3))`, nested tuples for n-D, sparse via `tensor_sparse_csr/coo`, ragged via `tensor_ragged`.
- Ops: elementwise `+ - * /` with broadcasting; reductions `sum/mean/var/std`; matmul/transpose; conv2d/max_pool2d; fft1d (dense).
- Types carry shape/dtype and tensor kind; type checker enforces compatible mixes; runtime errors include source context.

## Builtins (selected)
- Math: `pow, gcd, lcm, abs, sign, mod, floor, ceil, round, clamp, min, max`.
- Special functions: `gamma, beta, erf, erfc, igamma`.
- Distributions: `normal_pdf/cdf/sample`, `uniform_pdf/cdf/sample`, `exponential_pdf/cdf/sample`, `poisson_pmf/sample`, `binomial_pmf/sample`.
- Stats helpers: `quantile, correlation, regression`.
- Constructors/casts: `int, float, complex, decimal, rational, tensor*` variants, `to_dense`, `to_sparse_*`.
- RNG: deterministic `philox(seed, stream, ctr)`, `threefry(...)`; samplers derive from Philox.

## Error Model
- `util::Error` with source line/col; parser/lexer/runtime/type errors halt execution.
- Determinism: non-finite results rejected; division by zero and domain violations throw.

## Backend Architecture (CPU today)
- Interface: alloc/free, streams, events, capability flags, tensor/FFT/BLAS/conv descriptors; status/error codes.
- CPU backend: aligned allocs with canaries, zero-init, scrub-on-free, pooling for small blocks, NUMA hint (`LATTICE_NUMA_NODE`), leak tracking.
- Streams: fixed thread pool with task fusion and work stealing; dependencies via events; priorities tracked; deterministic mode for ordered execution.
- Docs: see `docs/BACKEND.md` for guarantees/unsupported items.

## Testing
- Comprehensive unit tests for lexer/parser/runtime/type checker, tensors (dense/sparse/ragged), annotations, builtins, RNG, and backend alloc/streams.
- Deterministic RNG and deterministic mode aid reproducible test baselines.
