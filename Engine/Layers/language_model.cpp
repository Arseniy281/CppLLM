#include "language_model.h"
#include "../Tensor/tensor.h"
#include "../Tensor/device.h"
#include "../Autograd/reshape_op.h"
#include <vector>
#include <memory>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>
#include <errno.h>
#include <chrono>
#include <iostream>
#include <filesystem>

#if ENABLE_CUDA
#include <cuda_runtime.h>
#endif

LanguageModel::LanguageModel(size_t vocab_size, size_t embed_dim, size_t num_blocks,
    size_t num_heads, size_t hidden_dim, Device device) : vocab_size_(vocab_size),
      device_(device),
      embedding_(vocab_size, embed_dim, device),
      transformer_(num_blocks, embed_dim, num_heads, hidden_dim, device),
      lm_head_(embed_dim, vocab_size, device),
      gen_(std::random_device{}()) {}


void LanguageModel::UpdateAdamW(float lr, float beta1, 
        float beta2, float eps, float weight_decay) {
    ++adam_step_;
    embedding_.UpdateAdamW(lr, beta1, beta2, eps, weight_decay, adam_step_);
    transformer_.UpdateAdamW(lr, beta1, beta2, eps, weight_decay, adam_step_);
    lm_head_.UpdateAdamW(lr, beta1, beta2, eps, weight_decay, adam_step_);
}


std::shared_ptr<Tensor> LanguageModel::forward(const std::shared_ptr<Tensor>& tokens) {
    auto x = embedding_.forward(tokens);

    if (x->GetShape().size() == 2) {
        auto reshape_op = std::make_shared<ReshapeOp>(
            std::vector<size_t>{1, x->GetShape()[0], x->GetShape()[1]});

        x = reshape_op->forward({x});
    }

    x = transformer_.forward(x);
    return lm_head_.forward(x);
}


int LanguageModel::Sample(const Tensor& probs) {
    std::vector<float> probs_vec(probs.GetSize());
    for (size_t i = 0; i < probs.GetSize(); i++) {
        probs_vec[i] = probs.at(i);
    }
    std::discrete_distribution<int> dist(probs_vec.begin(), probs_vec.end());
    return dist(gen_);
}


int LanguageModel::SampleGreedy(const Tensor& probs) {
    return (int)(GetBackend(probs.GetDevice()).ArgMax(probs));
}


void LanguageModel::TopP(Tensor& probs, float top_p) {
    if (top_p <= 0.0f || top_p >= 1.0f) { return; }

    std::vector<std::pair<float, int>> indexed_probs;
    indexed_probs.reserve(vocab_size_);

    for (size_t i = 0; i < vocab_size_; i++) {
        indexed_probs.push_back({probs.at(i), (int)(i)});
    }

    std::sort(indexed_probs.begin(), indexed_probs.end(),
        [](const auto& a, const auto& b) {
            return a.first > b.first;
        }
    );

    float cumulative = 0.0f;
    size_t border = 0;

    for (size_t i = 0; i < indexed_probs.size(); i++) {
        cumulative += indexed_probs[i].first;
        border = i;
        if (cumulative >= top_p) { break; }
    }

    for (size_t i = border + 1; i < indexed_probs.size(); i++) {
        probs.at((size_t)(indexed_probs[i].second)) = 0.0f;
    }

    float sum = 0.0f;

    for (size_t i = 0; i <= border; i++) {
        sum += indexed_probs[i].first;
    }

    if (sum <= 0.0f || !std::isfinite(sum)) {
        throw std::runtime_error(
            "LanguageModel::TopP: invalid probability sum"
        );
    }

    for (size_t i = 0; i <= border; i++) {
        probs.at((size_t)(indexed_probs[i].second))
                        = indexed_probs[i].first / sum;
    }
}

std::vector<size_t> LanguageModel::generate(const std::vector<size_t>& prompt,
    int max_new_tokens, float temperature, float top_p, int end_token_id) {
    if (prompt.empty() || max_new_tokens <= 0) {
        return {};
    }

    transformer_.SetUseKVCache(false);
    transformer_.ResetCache();

    std::vector<size_t> sequence = prompt;
    std::vector<size_t> generated;

    generated.reserve(max_new_tokens);

    for (int step = 0; step < max_new_tokens; step++) {
        std::vector<float> input_data(sequence.begin(), sequence.end());

        auto input = std::make_shared<Tensor>(
            std::vector<size_t>{1, sequence.size()}, input_data, device_);

        auto output = forward(input);
#if ENABLE_CUDA

        cudaError_t error = cudaDeviceSynchronize();

        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string("LanguageModel::generate: CUDA error: ") +
                cudaGetErrorString(error)
            );
        }

#endif

        Tensor last_logits({vocab_size_}, 0.0f, Device::CPU);
        const size_t offset = (sequence.size() - 1) * vocab_size_;

#if ENABLE_CUDA

        cudaError_t error = cudaMemcpy(last_logits.Data(), output->Data() + offset,
            vocab_size_ * sizeof(float), cudaMemcpyDeviceToHost );

        if (error != cudaSuccess) {
            throw std::runtime_error(std::string(
                "LanguageModel::generate: "
                "logits copy failed: ") 
                + cudaGetErrorString(error));
        }

#else

        std::copy(output->Data() + offset,
            output->Data() + offset + vocab_size_, last_logits.Data());

#endif

        size_t next_token = (size_t)(SampleGreedy(last_logits));

        if (end_token_id >= 0 && next_token == (size_t)(end_token_id)) {
            break;
        }

        generated.push_back(next_token);
        sequence.push_back(next_token);
    }

    transformer_.SetUseKVCache(false);
    transformer_.ResetCache();

    return generated;
}

void LanguageModel::SaveModel(const std::string& folder) {
    std::filesystem::create_directories(folder);

    embedding_.Save(folder + "/embedding");
    transformer_.Save(folder);
    lm_head_.Save(folder, "lm_head");

    std::ofstream file(folder + "/adam_step", std::ios::binary);

    if (!file) {
        throw std::runtime_error(
            "LanguageModel::SaveModel: "
            "failed to open adam_step"
        );
    }

    file.write(reinterpret_cast<const char*>(&adam_step_), sizeof(adam_step_));

    if (!file) {
        throw std::runtime_error(
            "LanguageModel::SaveModel: "
            "failed to save adam_step"
        );
    }
}

void LanguageModel::LoadModel(const std::string& folder) {
    embedding_.Load(folder + "/embedding");
    transformer_.Load(folder);
    lm_head_.Load(folder, "lm_head");

    std::ifstream file(folder + "/adam_step", std::ios::binary);

    if (!file) {
        throw std::runtime_error(
            "LanguageModel::LoadModel: "
            "failed to open adam_step"
        );
    }

    file.read(reinterpret_cast<char*>(&adam_step_), sizeof(adam_step_));

    if (!file) {
        throw std::runtime_error(
            "LanguageModel::LoadModel: "
            "failed to load adam_step"
        );
    }

    transformer_.ResetCache();
}

void LanguageModel::ResetCache() {
    transformer_.ResetCache();
}

void LanguageModel::Update(float lr) {
    embedding_.Update(lr);
    transformer_.Update(lr);
    lm_head_.Update(lr);
}

void LanguageModel::ClearGrad() {
    embedding_.ClearGrad();
    transformer_.ClearGrad();
    lm_head_.ClearGrad();
}

void LanguageModel::ScaleGrad(float factor) {
    embedding_.ScaleGrad(factor);
    transformer_.ScaleGrad(factor);
    lm_head_.ScaleGrad(factor);
}

void LanguageModel::SetUseKVCache(bool value) {
    transformer_.SetUseKVCache(value);
}