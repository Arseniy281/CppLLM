#pragma once

#include "operation.h"
#include "../Tensor/tensor.h"

#include <vector>
#include <memory>

class ReshapeOp : public Operation {
private:
    std::shared_ptr<Tensor> input_;
    std::vector<size_t> original_shape_;
    std::vector<size_t> new_shape_;

public:
    explicit ReshapeOp(
        const std::vector<size_t>& new_shape) : new_shape_(new_shape) {}

    std::shared_ptr<Tensor> forward(
        const std::vector<std::shared_ptr<Tensor>>& inputs) override;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
    std::vector<std::shared_ptr<Tensor>> GetInputs() const override;
};