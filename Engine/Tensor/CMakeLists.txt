cmake_minimum_required(VERSION 3.18)

project(Phase1_Tensor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TENSOR_SOURCES
    tensor.cpp
    tensor_ops.cpp
    tensor_print.cpp
    backend.cpp
    cpu_backend.cpp
    matmul.cpp
)

if(ENABLE_CUDA)

    enable_language(CUDA)

    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    set(CMAKE_CUDA_ARCHITECTURES 75)

    find_package(CUDAToolkit REQUIRED)

    list(APPEND TENSOR_SOURCES
        cuda_backend.cu
    )

endif()


add_library(tensor_lib
    ${TENSOR_SOURCES}
)

target_include_directories(tensor_lib PUBLIC
    .
)

if(ENABLE_CUDA)

    target_include_directories(tensor_lib PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(tensor_lib PUBLIC
        CUDA::cudart
        CUDA::cublas
    )

endif()


if(APPLE)
    target_compile_definitions(tensor_lib PUBLIC
        ACCELERATE_NEW_LAPACK
    )

    target_link_libraries(tensor_lib PUBLIC
        "-framework Accelerate"
    )
else()
    find_package(BLAS REQUIRED)
    target_link_libraries(tensor_lib PUBLIC
        BLAS::BLAS
    )
endif()