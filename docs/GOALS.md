# Lattice Language: Goals and Non-Goals

## Goals
- First-class numerical computing and statistics: dense tensors/arrays, linear algebra, special functions, distributions, and deterministic RNGs.
- Reproducibility by default: explicit seeds, controlled NaN/inf behavior, stable promotion rules, and deterministic execution.
- Performance with predictability: clear semantics for evaluation order, lexical scoping, and well-defined mutability/shadowing; optimizations must not change observable results.
- Gradual typing for numerics: optional static type hints with inference, strict promotions (int → float → complex; decimal/rational isolated), shape-aware tensors with broadcasting rules.
- Portability: CPU SIMD-first, with planned GPU/accelerator backends; consistent behavior across platforms and builds.

## Non-Goals
- General-purpose systems programming (no raw pointers/unsafe memory by default, no OS-level APIs).
- Implicit nondeterminism (hidden randomness, unordered execution without opt-in).
- Dynamic ad-hoc meta-programming that undermines static/gradual typing guarantees.
- Deeply coupled UI/web tooling; focus remains on scientific/numeric kernels and reproducible computation.
- Silent coercions between exact and inexact types (e.g., decimal/rational ↔ float/complex) without explicit casts.
