cmake_minimum_required(VERSION 3.15)

project(lattice LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LATTICE_SOURCES
    src/lexer/lexer.cpp
    src/parser/parser.cpp
    src/repl/repl.cpp
    src/runtime/ops.cpp
    src/runtime/backends/cpu/backend.cpp
    src/runtime/backends/gpu/dynloader.cpp
    src/runtime/backends/gpu/opencl_loader.cpp
    src/runtime/backends/gpu/opencl_backend.cpp
    src/runtime/backends/device_quirks.cpp
    src/runtime/backends/device_selector.cpp
    src/runtime/backends/cache_store.cpp
    src/runtime/backends/gpu/cuda_loader.cpp
    src/runtime/backends/gpu/cuda_backend.cpp
    src/runtime/backends/gpu/hip_loader.cpp
    src/runtime/backends/gpu/hip_backend.cpp
    src/runtime/decimal.cpp
    src/runtime/runner.cpp
    src/runtime/type_checker.cpp
    src/builtin/builtins.cpp
)

if(APPLE)
    list(APPEND LATTICE_SOURCES src/runtime/backends/gpu/metal_backend.mm)
endif()

add_library(lattice_lib ${LATTICE_SOURCES})

target_include_directories(lattice_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/OpenCL-Headers
)

if(UNIX AND NOT APPLE)
    target_link_libraries(lattice_lib PRIVATE dl)
endif()

if(APPLE)
    set_source_files_properties(src/runtime/backends/gpu/metal_backend.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(QUARTZCORE_FRAMEWORK QuartzCore)
    if(METAL_FRAMEWORK)
        target_link_libraries(lattice_lib PRIVATE ${METAL_FRAMEWORK})
    endif()
    if(FOUNDATION_FRAMEWORK)
        target_link_libraries(lattice_lib PRIVATE ${FOUNDATION_FRAMEWORK})
    endif()
    if(QUARTZCORE_FRAMEWORK)
        target_link_libraries(lattice_lib PRIVATE ${QUARTZCORE_FRAMEWORK})
    endif()
endif()

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
    tests/runtime/backend_tests.cpp
    tests/runtime/backend_edge_tests.cpp
    tests/runtime/abi_tests.cpp
    tests/runtime/device_selector_tests.cpp
    tests/runtime/device_quirks_tests.cpp
    tests/runtime/cache_store_tests.cpp
    tests/runtime/destructuring_tests.cpp
    tests/runtime/error_location_tests.cpp
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
