#include "transformer.h"
#include <sys/stat.h>
#include <chrono>
#include <iostream>
#include "../Tensor/device.h"

Transformer::Transformer(size_t n, size_t embed_dim, 
    size_t num_heads, size_t hidden_dim, Device device) : blocks_count_(n) {

    for (size_t i = 0; i < n; ++i) {
        blocks_.emplace_back(embed_dim, num_heads, hidden_dim, device);
    }
}

void Transformer::UpdateAdamW(float lr, float beta1, float beta2, 
        float eps, float weight_decay, size_t step) {
    for (auto& block : blocks_) {
        block.UpdateAdamW(lr, beta1, beta2, eps, weight_decay, step);
    }
}

std::shared_ptr<Tensor> Transformer::forward(const std::shared_ptr<Tensor>& x) {
    std::shared_ptr<Tensor> input = x;
    for (size_t i = 0; i < blocks_count_; i++) {
        input = blocks_[i].forward(input);
    }

    return input;
}

void Transformer::Update(float lr) {
    for (auto& block : blocks_) {
        block.Update(lr);
    }
}

void Transformer::ClearGrad() {
    for (auto& block : blocks_) {
        block.ClearGrad();
    }
}

void Transformer::ScaleGrad(float factor) {
    for (auto& block : blocks_) {
        block.ScaleGrad(factor);
    }
}

void Transformer::Save(const std::string& folder) const {
    if (mkdir(folder.c_str(), 0777) != 0 && errno != EEXIST) {
        throw std::runtime_error(
            "Cannot create directory: " + folder
        );
    }

    for (size_t i = 0; i < blocks_count_; i++) {
        blocks_[i].Save(
            folder + "/block_" + std::to_string(i)
        );
    }
}

void Transformer::Load(const std::string& folder) {
    for (size_t i = 0; i < blocks_count_; i++) {
        blocks_[i].Load(folder + "/block_" + std::to_string(i));
    }
}

void Transformer::ResetCache() {
    for (auto& block : blocks_) {
        block.ResetCache();
    }
}

void Transformer::SetUseKVCache(bool value) {
    for (auto& block : blocks_) {
        block.SetUseKVCache(value);
    }
}