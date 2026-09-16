#pragma once

#include "../Tensor/tensor.h"
#include "../Autograd/operation.h"
#include <memory>

class EmbeddingLayer;

class EmbeddingOp : public Operation {
private:
    EmbeddingLayer* layer_;
    std::shared_ptr<Tensor> indices_;

public:
    explicit EmbeddingOp(EmbeddingLayer* layer) : layer_(layer) {}

    std::shared_ptr<Tensor> forward(
        const std::vector<std::shared_ptr<Tensor>>& inputs) override;

    std::vector<Tensor> backward(
        const Tensor& grad_output) override;

    std::vector<std::shared_ptr<Tensor>> GetInputs() const override;
};