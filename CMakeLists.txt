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
    src/builtin/builtins.cpp
)

target_include_directories(lattice_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(lattice main.cpp)

target_link_libraries(lattice PRIVATE lattice_lib)
