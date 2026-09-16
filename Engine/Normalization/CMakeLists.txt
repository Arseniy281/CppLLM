cmake_minimum_required(VERSION 3.10)
project(Phase9_Normalization)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(normalization_lib
    rms_norm.cpp
)

target_link_libraries(normalization_lib PUBLIC
    autograd_lib
    tensor_lib
)