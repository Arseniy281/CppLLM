cmake_minimum_required(VERSION 3.10)
project(Phase10_Transformer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(transformer_lib
    transformer_block.cpp
    transformer.cpp
)

target_link_libraries(transformer_lib PUBLIC
    attention_lib
    layers_lib
    normalization_lib
    autograd_lib
    tensor_lib
)