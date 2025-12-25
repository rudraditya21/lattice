# lattice

Lattice is a small scientific computing language aimed at reproducible numerics and statistical workflows. It ships a REPL and embeddable library with:
- Arithmetic expressions with identifiers, unary minus, calls, assignments, and blocks.
- Control flow via `if/else` statements, nested blocks, `while` loops, `for` loops, and `break`/`continue`.
- Strict typing with optional annotations and a numeric tower (ints, floats, complex, decimals, rationals) plus tensors.
- Built-in constants `pi`, `e`, `gamma`, `inf` and math helpers `pow`, `gcd`, `lcm`, `abs`, `sign`, `mod`, `floor`, `ceil`, `round`, `clamp`, `min`, `max`, `sum`, `mean`.
- Typed constructors/casts: `int()`, `float()`, `complex()`, `decimal()`, `rational()`, `tensor()`.
The project is organized with headers in `include/` and sources in `src/`.

## Layout
- `include/` – public headers (`lexer/`, `parser/`, `runtime/`, `repl/`, `builtin/`, `util/`)
- `src/` – implementation (`lexer/`, `parser/`, `runtime/`, `repl/`, `builtin/`, `main.cpp`)
- `tests/` – modular tests under `tests/{lexer,parser,runtime,builtin,util,repl}` with shared helpers at `tests/test_util.*` and entry in `tests/main.cpp`
- `CMakeLists.txt` – build configuration
- `.clang-format` – formatting rules
- `.gitignore` – repository hygiene

## Build
```bash
cmake -S . -B build
cmake --build build
```

## Run REPL
```bash
./build/lattice
```
Example session (typed values, complex/tensors):
```
lattice> x = 3
3
lattice> x + 2
5
lattice> complex(1, -2)
1+-2i
lattice> tensor(2, 2, 1)
tensor[2x2]<12>
lattice> sum(tensor(2, 2, 1))
4
lattice> exit
```

## Control Flow
- `if/else`: `if (condition) { expr_or_stmt } else { other_stmt }`. Conditions must be `bool`.
- Comparisons: `==`, `!=`, `>`, `<`, `>=`, `<=` yield booleans (`true`/`false`).
- Boolean literals: `true`, `false`; arithmetic on bools is allowed (1/0) but control-flow requires `bool`.
- `while`: `while (condition) body`. Condition re-evaluated each iteration.
- `for`: `for (init; condition; increment) body`. Any of the three clauses may be empty (e.g., `for (; cond; )`).
- `break` / `continue`: only valid inside loops; `continue` skips to the next iteration; `break` exits the nearest loop. At the REPL top level they print an error.
- Blocks: `{ stmt1; stmt2; ... }` with optional semicolons after statements.
- Functions: `func name(param1, param2) { ... }` defines a function; `return expr;` exits with a value. Without `return`, the last statement value is used if present, otherwise `0`. Functions may carry type annotations on params/returns; annotated boundaries are enforced, unannotated code remains dynamic.

## Types and Promotion
- Numeric tower: `i8/i16/i32/i64/u8/u16/u32/u64`, `f16/bfloat16/f32/f64`, `complex64/complex128`, `decimal`, `rational`, `bool`.
- Aggregates: `tensor` with shape/row-major strides and element dtype (default `f64`).
- Constructors/casts: `int`, `float`, `complex`, `decimal`, `rational`, `tensor`.
- Promotion: complex > float > int; signed/unsigned width-aware; decimal↔decimal and rational↔rational only; tensor ops promote element dtype; implicit narrowing is rejected (use `cast`/constructors).
- Math builtins enforce dtype expectations (e.g., `gcd/lcm` require integers, `abs` handles complex/decimal/rational, `pow` supports complex).

## Tests
```bash
cmake --build build
ctest --test-dir build
```
Tests exercise parsing and evaluation (arithmetic, identifiers, constants, builtins, and error paths).

## Formatting
Use clang-format with the provided configuration:
```bash
find include src \( -name '*.h' -o -name '*.cpp' \) -print0 | xargs -0 clang-format -i
```
