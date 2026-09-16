#pragma once
#include "../Transformer/transformer.h"
#include "embedding_layer.h"
#include "../Tensor/tensor.h"
#include "../Tensor/device.h"
#include "linear_layer.h"
#include <vector>

class LanguageModel {
private:
    EmbeddingLayer embedding_;
    Transformer transformer_;
    LinearLayer lm_head_;
    Softmax softmax_;
    size_t vocab_size_;

    std::mt19937 gen_;

    Device device_;
    size_t adam_step_ = 0;

public:
    LanguageModel(size_t vocab_size, size_t embed_dim, size_t num_blocks,
        size_t num_heads, size_t hidden_dim, Device device = Device::CPU);
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& tokens);
    std::vector<size_t> generate(const std::vector<size_t>& prompt,
        int max_len, float temperature, float top_p, int end_token_id);

    void UpdateAdamW(float lr, float beta1 = 0.9f, float beta2 = 0.999f,
        float eps = 1e-8f, float weight_decay = 0.01f);

    int Sample(const Tensor& probs);
    int SampleGreedy(const Tensor& probs);
    void TopP(Tensor& last_logits, float top_p = 0.9);

    void SaveModel(const std::string& folder);
    void LoadModel(const std::string& folder);

    void ResetCache();
    void Update(float lr);
    void ClearGrad();
    void ScaleGrad(float factor);
    void SetUseKVCache(bool value);
};