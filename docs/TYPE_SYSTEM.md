# Type System Plan

## Goals
- Serve scientific/statistical workloads with predictable promotion rules and explicitness for lossy conversions.
- Allow gradual typing: optional annotations with inference where obvious, but runnable in a mostly dynamic mode.
- Keep numerics first-class, with tensors/arrays as core aggregates.

## Numeric Tower
- Integers: `i8/i16/i32/i64`, `u8/u16/u32/u64`.
- Floating point: `f16`, `f32`, `f64`, `bfloat16`.
- Complex: `complex64` (f32 real/imag), `complex128` (f64 real/imag).
- Exact numbers: decimals (configurable precision), rationals (normalized numerator/denominator, signed).
- Promotions (sketch):
  - integer → widest integer in operation; mixed int/float → float; float/complex → complex.
  - decimals/rationals do not auto-mix with floats unless explicitly converted.
  - bfloat16/f16 promote to f32 when mixing with >= f32 to avoid loss.

## Aggregates
- Scalars above plus arrays/tensors with shape/dtype metadata.
- Row-major default; broadcasting semantics explicit and consistent (NumPy-style).
- Future: sparse/ragged variants gated behind flags.

## Type Hints and Inference
- Optional annotations on function parameters/returns: `func add(a: f64, b: f64) -> f64 { ... }`.
- Let-binding/assignment can carry annotations: `x: i32 = 3`.
- Local inference: propagate from literals, operations, and annotations; shape/dtype inference for tensors.
- Gradual typing: unannotated values are dynamically typed; annotated values enforce checks at boundaries.
- Type errors surface with source ranges; implicit narrowing disallowed without explicit cast.

## Runtime Representation (direction)
- Tagged value supporting the numeric tower plus function/boolean and aggregates.
- Separate heap arena for tensors with dtype/shape headers; scalars stored inline.
- Decimal/rational backed by dedicated structs (decimal via software decimal; rational via normalized ints).

## Standard Library Considerations
- Math functions overload across numeric tower where meaningful; exact types use exact algorithms when possible.
- Conversions: `int()`, `float()`, `complex()`, `decimal(precision=…)`, `rational()` with explicit semantics.
- Random module returns typed tensors/scalars respecting dtype parameters.
