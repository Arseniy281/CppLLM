cmake_minimum_required(VERSION 3.10)

project(CppLLM)

option(
    ENABLE_CUDA
    "Enable CUDA backend"
    OFF
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(ENABLE_CUDA)

    enable_language(CUDA)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    add_compile_definitions(ENABLE_CUDA=1)

else()

    add_compile_definitions(ENABLE_CUDA=0)

endif()


add_subdirectory(Engine/Tensor)
add_subdirectory(Engine/Attention)
add_subdirectory(Engine/Autograd)
add_subdirectory(Engine/Layers)
add_subdirectory(Engine/Transformer)
add_subdirectory(Engine/Normalization)
add_subdirectory(Engine/Tokenizer)


add_executable(tokenize
    Models/ToyModel/tokenize.cpp
)

target_link_libraries(tokenize
    PRIVATE
    tokenizer_lib
)

add_executable(train
    Models/ToyModel/train.cpp
)

target_link_libraries(train
    PRIVATE
    transformer_lib
    tokenizer_lib
)

add_executable(finetune
    Models/ToyModel/finetune.cpp
)

target_link_libraries(finetune
    PRIVATE
    transformer_lib
    tokenizer_lib
)

add_executable(generate
    Models/ToyModel/generate.cpp
)

target_link_libraries(generate
    PRIVATE
    transformer_lib
    tokenizer_lib
)

add_executable(data_generator
    Data/data_generator.cpp
)