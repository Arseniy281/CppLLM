cmake_minimum_required(VERSION 3.10)

project(Phase4_Layers)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# Layers
# ============================================================

add_library(layers_lib
    ce_loss.cpp
    mse_loss.cpp
    linear_layer.cpp
    feed_forward.cpp
    language_model.cpp
    embedding_layer.cpp
)

# ============================================================
# Includes
# ============================================================

target_include_directories(layers_lib PUBLIC
    .
)