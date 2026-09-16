cmake_minimum_required(VERSION 3.10)
project(Phase3_AutoGrad)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(autograd_lib 
    add_op.cpp
    gelu.cpp
    mul_op.cpp
    relu.cpp
    sigmoid.cpp
    square_op.cpp
    sub_op.cpp
    sum_op.cpp
    tanh_op.cpp
    embedding_op.cpp
    rope_op.cpp
    rms_norm_op.cpp
    concatenate_op.cpp
    mul_scalar_op.cpp
    softmax_op.cpp
    transpose_op.cpp
    reshape_op.cpp
)

target_include_directories(autograd_lib PUBLIC .)