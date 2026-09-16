#pragma once

#include "operation.h"
#include "../Tensor/tensor.h"

#include <vector>
#include <memory>

class ConcatenateOp : public Operation {
private:
    std::vector<std::shared_ptr<Tensor>> inputs_;
    size_t axis_;

public:
    explicit ConcatenateOp(size_t axis);
    std::shared_ptr<Tensor> forward(const std::vector<std::shared_ptr<Tensor>>& inputs) override;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
    std::vector<std::shared_ptr<Tensor>> GetInputs() const override;
};