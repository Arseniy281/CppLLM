#pragma once

#include "../Autograd/operation.h"
#include "../Tensor/tensor.h"

#include <memory>

class RMSNormOp : public Operation {
private:
    std::shared_ptr<Tensor> input_;
    std::shared_ptr<Tensor> gamma_;

    Tensor saved_x_norm_;
    Tensor saved_rms_;

public:
    RMSNormOp(const std::shared_ptr<Tensor>& gamma);

    std::shared_ptr<Tensor> forward(
        const std::vector<std::shared_ptr<Tensor>>& inputs) override;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
    std::vector<std::shared_ptr<Tensor>> GetInputs() const override;
};