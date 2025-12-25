# lattice

Lattice is a small scientific computing language aimed at reproducible numerics and statistical workflows. It ships a REPL and embeddable library with:
- Arithmetic expressions with identifiers, unary minus, calls, assignments, and blocks.
- Control flow via `if/else` statements, nested blocks, `while` loops, `for` loops, and `break`/`continue`.
- Built-in constants `pi`, `e`, `gamma`, `inf` and math helpers `pow`, `gcd`, `lcm`, `abs`, `sign`, `mod`, `floor`, `ceil`, `round`, `clamp`, `min`, `max`.
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
Example session:
```
lattice> x = 3
3.000000
lattice> x + 2
5.000000
lattice> pi * 2
6.283185
lattice> pow(2, 3)
8.000000
lattice> gcd(12, 8)
4.000000
lattice> lcm(3, 5)
15.000000
lattice> exit
```

## Control Flow
- `if/else`: `if (condition) { expr_or_stmt } else { other_stmt }`. Any non-zero/`true` value is truthy.
- Comparisons: `==`, `!=`, `>`, `<`, `>=`, `<=` yield booleans (`true`/`false`, internally `1`/`0`).
- Boolean literals: `true`, `false`; they participate in expressions and control flow like C++ (converted to `1`/`0` in arithmetic).
- `while`: `while (condition) body`. Condition re-evaluated each iteration.
- `for`: `for (init; condition; increment) body`. Any of the three clauses may be empty (e.g., `for (; cond; )`).
- `break` / `continue`: only valid inside loops; `continue` skips to the next iteration; `break` exits the nearest loop. At the REPL top level they print an error.
- Blocks: `{ stmt1; stmt2; ... }` with optional semicolons after statements.
- Functions: `func name(param1, param2) { ... }` defines a function; `return expr;` exits with a value. Without `return`, the last statement value is used if present, otherwise `0`. User functions share the C++-style semantics for truthiness and comparisons and can call other functions or builtins.

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
