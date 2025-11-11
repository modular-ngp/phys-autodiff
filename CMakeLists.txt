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

add_executable(main
        main.cpp
        src/mlp_cpu.cpp
        src/mlp_cuda.cu
)
target_include_directories(main PRIVATE include)
