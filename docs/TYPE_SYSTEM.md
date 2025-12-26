# Type System Plan

## Goals
- Serve scientific/statistical workloads with predictable promotion rules and explicitness for lossy conversions.
- Allow gradual typing: optional annotations with inference where obvious, but runnable in a mostly dynamic mode.
- Keep numerics first-class, with tensors/arrays as core aggregates.

## Numeric Tower
- Integers: `i8/i16/i32/i64`, `u8/u16/u32/u64`.
- Floating point: `f16`, `bfloat16`, `f32`, `f64`.
- Complex: `complex64` (f32 real/imag), `complex128` (f64 real/imag).
- Exact numbers: `decimal` (software, configurable precision), `rational` (normalized numerator/denominator, signed).
- Strings: `string` (immutable).
- Promotions (enforced):
  - complex > float > int; signed/unsigned width-aware.
  - `decimal`↔`decimal` and `rational`↔`rational` only; cross with other numerics requires explicit cast.
  - bfloat16/f16 promote to f32+ when mixing with ≥ f32 to avoid loss.
- Finite-only policy: NaN/inf are rejected at runtime (division by zero or non-finite results raise
  errors with source locations); deterministic printing is preserved.

## Aggregates
- Scalars above plus tensors with shape/dtype metadata, tuples, and records.
- Row-major default; elementwise ops support NumPy-style broadcasting (trailing dimensions align; a
  dimension may be `1` or equal to the other operand; scalar `()` broadcasts to any shape). Shape
  mismatch raises an error. Sum/mean builtins reduce all elements; dtype promotion applied to
  elementwise operations.
- Tuples: fixed-length, immutable positional collections; element dtypes tracked when known; structural equality and assignment require matching length and element dtypes.
- Records: ordered immutable field sets `{name: value}`; keys are strings/identifiers; field dtypes tracked when known; structural equality and assignment require matching names/order and dtypes.
- Future: sparse/ragged variants gated behind flags (design below).

## Type Hints and Inference
- Optional annotations on function parameters/returns: `func add(a: f64, b: f64) -> f64 { ... }`.
- Let-binding/assignment can carry annotations: `x: i32 = 3`.
- Local inference: propagate from literals, operations, and annotations; shape/dtype carried with tensors.
- Gradual typing: unannotated values are dynamically typed; annotated values enforce checks at boundaries.
- Type errors surface with source ranges; implicit narrowing disallowed without explicit cast.

## Runtime Representation (direction)
- Tagged value supporting the numeric tower plus function/boolean and tensor aggregate.
- Tensor stores shape/row-major strides/element dtype and flattened storage. Small-vector
  optimization (SVO) exists for tiny tensors (inline buffer of 8 elements); benchmark and tune the
  threshold per backend once perf harness is in place. Consider extending SVO to tuple/record
  metadata if profiles show allocator churn.
- Decimal/rational backed by dedicated structs (decimal via software decimal; rational via
  normalized ints).
- Benchmark plan: microbenchmarks for tensor elementwise ops and reductions across sizes 1–64 to
  validate SVO thresholds; track allocations and cache misses. If tuples/records are hot, add inline
  storage for up to N elements/fields guarded by perf data.

## Standard Library Considerations
- Math functions overload across the numeric tower where meaningful (dtype-aware `abs`, `pow`, `gcd`, `lcm`, `mod`, `clamp`, `min`, `max`, `sum`, `mean`); exact types use exact algorithms when possible.
- Conversions: `int()`, `float()`, `complex()`, `decimal()`, `rational()`, `tensor()` with explicit semantics; `cast(type, expr)` available.
- Random features (e.g., typed RNG) are deferred until later; modules/imports are out of scope for now.

## Sparse and Ragged Tensors (Plan)
- Goals: memory efficiency for structured sparsity; basic ragged (jagged) list-of-lists semantics for
  uneven dimensions; opt-in via constructors/flags to keep dense fast paths unchanged.
- Storage formats:
  - Sparse: start with CSR/COO for 2D; extend to block-sparse; store `indices`, `indptr` (CSR),
    `values`, and `shape`, plus `dtype` and a `format` enum. Support row-major iteration and
    elementwise ops where both operands share format/shape. Provide dense<->sparse conversion.
  - Ragged: represent with a `values` buffer and a `row_splits`/`offsets` vector per ragged
    dimension; carry `dtype` and outer shape (number of rows). Indexing validates bounds; no
    broadcasting across ragged dims.
- API surface (to be added):
  - `tensor_dense(...)` (existing) remains default.
  - `tensor_sparse_csr(shape, indptr, indices, values, dtype=...)`, `tensor_sparse_coo(shape,
    coords, values, dtype=...)`.
  - `tensor_ragged(row_splits, values, dtype=...)`.
  - `to_dense(tensor)`, `to_sparse(tensor, format=...)` for conversions.
- Semantics and checks:
  - Shape/dtype validated at construction; sparse indices must be in-bounds and sorted (or flagged
    unsorted with a normalize step).
  - Elementwise ops: dense ⊕ sparse (auto-densify or error based on a flag); sparse ⊕ sparse only
    when formats match; ragged supports only elementwise ops within aligned ragged structure; no
    cross with dense unless densified explicitly.
  - Reductions: sparse uses sparse-aware reductions; ragged reduces along innermost dense values;
    reduction over ragged axes requires explicit semantics (e.g., pad/error) and should be gated by
    flags.
  - Alignment/padding: allow optional alignment hints for dense buffers; sparse values/indices kept
    contiguous; document padding policy for packed buffers (currently none).
- Typing:
  - Extend `Type` to carry a tensor flavor: `dense`, `sparse(format)`, `ragged` plus element dtype
    and shape metadata. Default remains dense.
  - Type checker enforces operations only between compatible flavors unless explicit conversions are
    called.
- Performance/benchmark plan:
  - Add micro/macro benchmarks for sparse matvec/matmul vs. dense, varying sparsity.
  - Measure ragged indexing and simple reductions.
  - Tune SVO thresholds separately for dense and for small sparse metadata buffers.
