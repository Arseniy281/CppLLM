#pragma once

#include "operation.h"
#include "../Tensor/tensor.h"

#include <vector>
#include <memory>

class TransposeOp : public Operation {
private:
    std::shared_ptr<Tensor> input_;

public:
    std::shared_ptr<Tensor> forward(
        const std::vector<std::shared_ptr<Tensor>>& inputs) override;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
    std::vector<std::shared_ptr<Tensor>> GetInputs() const override;
};