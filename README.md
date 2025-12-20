# lattice

Minimal lattice language with a REPL supporting numeric literals, identifiers, unary minus, and `+/-/*//` operations plus assignment. Built-in constants `pi` and `e` are available, and builtin functions `pow(x, y)`, `gcd(a, b)`, `lcm(a, b)`, `abs(x)`, `sign(x)`, `mod(a, b)`, `floor(x)`, `ceil(x)`, `round(x)`, `clamp(x, lo, hi)`, `min(a, b)`, and `max(a, b)` are provided. The project is organized with headers in `include/` and sources in `src/`.

## Layout
- `include/` – public headers (`lexer/`, `parser/`, `runtime/`, `repl/`, `builtin/`, `util/`)
- `src/` – implementation (`lexer/`, `parser/`, `runtime/`, `repl/`, `builtin/`, `main.cpp`)
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

## Formatting
Use clang-format with the provided configuration:
```bash
find include src \( -name '*.h' -o -name '*.cpp' \) -print0 | xargs -0 clang-format -i
```
