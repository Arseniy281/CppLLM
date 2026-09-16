#pragma once
#include "../Tensor/tensor.h"
#include <vector>

class CrossEntropyLoss {
private:
    Tensor logits_;
    Tensor targets_;
    std::vector<float> probabilities_;
    size_t batch_;
    size_t seq_len_;
    size_t vocab_size_;
    float loss_;

public:
    Tensor forward(const Tensor& logits, const Tensor& targets);
    Tensor backward();
};