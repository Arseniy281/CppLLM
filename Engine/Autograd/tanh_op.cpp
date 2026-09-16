#include "tanh_op.h"
#include "../Tensor/tensor.h"
#include "../Tensor/backend.h"

#include <vector>
#include <memory>
#include <stdexcept>

std::shared_ptr<Tensor> TanhOp::forward(
        const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 1) {
        throw std::runtime_error("TanhOp expects exactly one input");
    }

    if (inputs[0] == nullptr) {
        throw std::runtime_error("TanhOp received null input");
    }

    parent_ = inputs[0];
    Tensor result = GetBackend(parent_->GetDevice()).Tanh(*parent_);

    auto output = std::make_shared<Tensor>(std::move(result));
    output->SetGradFn(shared_from_this());

    return output;
}

std::vector<Tensor> TanhOp::backward(const Tensor& grad_output) {
    if (parent_ == nullptr) {
        throw std::runtime_error("TanhOp has no parent");
    }

    if (grad_output.GetShape() != parent_->GetShape()) {
        throw std::runtime_error(
            "TanhOp::backward: grad_output shape mismatch");
    }

    Tensor grad_input = GetBackend(
        parent_->GetDevice()).TanhBackward(*parent_, grad_output);
    return { std::move(grad_input) };
}

std::vector<std::shared_ptr<Tensor>>TanhOp::GetInputs() const {
    return { parent_ };
}