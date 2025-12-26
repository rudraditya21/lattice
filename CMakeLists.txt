cmake_minimum_required(VERSION 3.15)

project(lattice LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(lattice_lib
    src/lexer/lexer.cpp
    src/parser/parser.cpp
    src/repl/repl.cpp
    src/runtime/ops.cpp
    src/runtime/decimal.cpp
    src/runtime/runner.cpp
    src/runtime/type_checker.cpp
    src/builtin/builtins.cpp
)

target_include_directories(lattice_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(lattice main.cpp)

target_link_libraries(lattice PRIVATE lattice_lib)

enable_testing()
add_executable(lattice_tests
    tests/main.cpp
    tests/test_util.cpp
    tests/lexer/lexer_tests.cpp
    tests/parser/parser_tests.cpp
    tests/runtime/runtime_tests.cpp
    tests/runtime/decimal_rational_tests.cpp
    tests/runtime/complex_tests.cpp
    tests/runtime/tensor_tests.cpp
    tests/runtime/tuple_record_tests.cpp
    tests/runtime/destructuring_tests.cpp
    tests/runtime/type_checker_tests.cpp
    tests/builtin/builtin_tests.cpp
    tests/builtin/print_tests.cpp
    tests/util/util_tests.cpp
    tests/repl/repl_tests.cpp
    tests/interpreter/interpreter_tests.cpp
    tests/runtime/promotion_tests.cpp
)
target_include_directories(lattice_tests PRIVATE tests)
target_link_libraries(lattice_tests PRIVATE lattice_lib)
add_test(NAME lattice_tests COMMAND lattice_tests)
