# lattice

Minimal lattice language with a REPL supporting numeric literals, identifiers, unary minus, and `+/-/*//` operations plus assignment. Built-in constants `pi` and `e` are available. The project is organized with headers in `include/` and sources in `src/`.

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
lattice> exit
```

## Formatting
Use clang-format with the provided configuration:
```bash
find include src \( -name '*.h' -o -name '*.cpp' \) -print0 | xargs -0 clang-format -i
```
