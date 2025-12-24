# lattice

Lattice is a small scientific computing language aimed at reproducible numerics and statistical workflows. It ships a REPL and embeddable library with:
- Arithmetic expressions with identifiers, unary minus, calls, assignments, and blocks.
- Control flow via `if/else` statements and nested blocks.
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
