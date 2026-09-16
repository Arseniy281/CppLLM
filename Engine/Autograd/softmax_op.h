#pragma once

#include "operation.h"
#include "../Tensor/tensor.h"
#include <memory>
#include <vector>

class SoftmaxOp : public Operation {
private:
    std::shared_ptr<Tensor> input_;
    Tensor output_;

public:
    std::shared_ptr<Tensor> forward(
        const std::vector<std::shared_ptr<Tensor>>& inputs) override;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
    std::vector<std::shared_ptr<Tensor>> GetInputs() const override;
};