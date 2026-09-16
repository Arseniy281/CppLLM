#pragma once

#include "../../Engine/Tensor/device.h"
#include <iostream>

#if ENABLE_CUDA
constexpr Device TRAINING_DEVICE = Device::CUDA;
#else
constexpr Device TRAINING_DEVICE = Device::CPU;
#endif

const char* DeviceName(Device d) {
    switch (d) {
        case Device::CPU:  return "CPU";
        case Device::CUDA: return "CUDA";
    }
    return "Unknown";
}

#include <filesystem>

inline std::filesystem::path GetProjectRoot() {
    namespace fs = std::filesystem;

    fs::path current = fs::current_path();

    if (fs::exists(current / "Data")) {
        return current;
    }

    if (fs::exists(current.parent_path() / "Data")) {
        return current.parent_path();
    }

    throw std::runtime_error("Cannot find project root");
}

constexpr size_t VOCAB_SIZE = 1000;
constexpr size_t EMBED_DIM = 32;
constexpr size_t BLOCKS = 2;
constexpr size_t HEADS = 2;
constexpr size_t HIDDEN = 64;
constexpr size_t CONTEXT = 32;
constexpr size_t BATCH_SIZE = 16;
constexpr size_t STEPS = 100;

constexpr float LEARNING_RATE = 0.001f;
const float WEIGHT_DECAY = 0.0f;
constexpr unsigned int SEED = 42;

const float BETA1 = 0.9f;
const float BETA2 = 0.999f;
const float EPS = 1e-8f;

const size_t VALIDATION_EVERY = 100;
const size_t VALIDATION_BATCHES = 10;

inline const std::filesystem::path PROJECT_ROOT = GetProjectRoot();

inline const std::filesystem::path DATA_PATH = PROJECT_ROOT / "Data/data.txt";
inline const std::filesystem::path TOKENIZER_PATH = PROJECT_ROOT / "Models/ToyModel/ToyTokenizer";
inline const std::filesystem::path MODEL_PATH = PROJECT_ROOT / "Models/ToyModel/ModelFiles";