cmake_minimum_required(VERSION 3.10)
project(Phase8_Attention)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(attention_lib
    rope.cpp
    attention.cpp
    softmax.cpp
)

target_link_libraries(attention_lib PUBLIC
    layers_lib
    autograd_lib
    tensor_lib
)