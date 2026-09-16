#pragma once

#include "operation.h"
#include "../Tensor/tensor.h"

class RopeOp : public Operation {
private:
    std::shared_ptr<Tensor> input_;
    size_t start_pos_;

public:
    explicit RopeOp(size_t start_pos = 0) : start_pos_(start_pos) {}
    std::shared_ptr<Tensor> forward(
        const std::vector<std::shared_ptr<Tensor>>& inputs) override;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
    std::vector<std::shared_ptr<Tensor>> GetInputs() const override;
};