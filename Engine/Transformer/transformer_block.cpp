#include "transformer_block.h"
#include "../Tensor/tensor.h"
#include <sys/stat.h>
#include <errno.h> 
#include <vector>
#include "../Tensor/device.h"

#include <chrono>
#include <iostream>

TransformerBlock::TransformerBlock(size_t embed_dim, size_t num_heads, size_t hidden_dim, Device device)
    : rms_norm_1_(embed_dim, device), attention_(embed_dim, num_heads, device),
    rms_norm_2_(embed_dim, device), feed_forward_(embed_dim, hidden_dim, device) {}

void TransformerBlock::UpdateAdamW(float lr, float beta1, float beta2,
        float eps, float weight_decay, size_t step) {

    rms_norm_1_.UpdateAdamW(lr, beta1, beta2, eps, weight_decay, step);
    attention_.UpdateAdamW(lr, beta1, beta2, eps, weight_decay, step);
    rms_norm_2_.UpdateAdamW(lr, beta1, beta2, eps, weight_decay, step);
    feed_forward_.UpdateAdamW(lr, beta1, beta2, eps, weight_decay, step);
}

std::shared_ptr<Tensor> TransformerBlock::forward(const std::shared_ptr<Tensor>& x) {
    auto residual = x;
    auto input = rms_norm_1_.forward(x);
    input = attention_.forward(input);
    auto add1 = std::make_shared<AddOp>();
    input = add1->forward({input, residual});

    residual = input;
    input = rms_norm_2_.forward(input);
    input = feed_forward_.forward(input);
    auto add2 = std::make_shared<AddOp>();
    input = add2->forward({input, residual});

    return input;
}

void TransformerBlock::Update(float lr) {
    rms_norm_1_.Update(lr);
    attention_.Update(lr);
    rms_norm_2_.Update(lr);
    feed_forward_.Update(lr);
}

void TransformerBlock::ClearGrad() {
    rms_norm_1_.ClearGrad();
    attention_.ClearGrad();
    rms_norm_2_.ClearGrad();
    feed_forward_.ClearGrad();
}

void TransformerBlock::ScaleGrad(float factor) {
    rms_norm_1_.ScaleGrad(factor);
    attention_.ScaleGrad(factor);
    rms_norm_2_.ScaleGrad(factor);
    feed_forward_.ScaleGrad(factor);
}

void TransformerBlock::Save(const std::string& folder) const {
    if (mkdir(folder.c_str(), 0777) != 0 && errno != EEXIST) {
        throw std::runtime_error("Cannot create directory: " + folder);
    }

    rms_norm_1_.Save(folder + "/rmsnorm_1_gamma");
    attention_.Save(folder + "/attention");
    rms_norm_2_.Save(folder + "/rmsnorm_2_gamma");
    feed_forward_.Save(folder + "/feedforward");
}

void TransformerBlock::Load(const std::string& folder) {
    rms_norm_1_.Load(folder + "/rmsnorm_1_gamma");
    attention_.Load(folder + "/attention");
    rms_norm_2_.Load(folder + "/rmsnorm_2_gamma");
    feed_forward_.Load(folder + "/feedforward");
}

void TransformerBlock::ResetCache() {
    attention_.ResetCache();
}

void TransformerBlock::SetUseKVCache(bool value) {
    attention_.SetUseKVCache(value);
}