cmake_minimum_required(VERSION 3.26)
project(phys-autodiff VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
# Default to a broad set of modern architectures; override with -DCMAKE_CUDA_ARCHITECTURES=XX if needed
set(CMAKE_CUDA_ARCHITECTURES 75;86;89;90 CACHE STRING "CUDA archs" FORCE)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Core library
add_library(phys_core STATIC
        src/mlp_cpu.cpp
        src/mlp_cuda.cu
)
target_include_directories(phys_core PUBLIC include)

# Tests (each as a separate executable)
add_executable(test_mlp_compare
        test/test_mlp_compare.cpp
)
target_link_libraries(test_mlp_compare PRIVATE phys_core)

add_executable(test_phys_cpu_ref
        test/test_phys_cpu_ref.cpp
)
target_link_libraries(test_phys_cpu_ref PRIVATE phys_core)

add_executable(test_phys_cuda_nonfused_vs_cpu
        test/test_phys_cuda_nonfused_vs_cpu.cpp
)
target_link_libraries(test_phys_cuda_nonfused_vs_cpu PRIVATE phys_core)

add_executable(test_phys_cuda_fused_vs_nonfused
        test/test_phys_cuda_fused_vs_nonfused.cpp
)
target_link_libraries(test_phys_cuda_fused_vs_nonfused PRIVATE phys_core)

add_executable(test_phys_perf
        test/test_phys_perf.cpp
)
target_link_libraries(test_phys_perf PRIVATE phys_core)
