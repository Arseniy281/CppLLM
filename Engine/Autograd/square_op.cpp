#include "square_op.h"
#include "../Tensor/tensor.h"
#include "../Tensor/backend.h"

#include <vector>
#include <memory>
#include <stdexcept>

std::shared_ptr<Tensor> SquareOp::forward(
        const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 1) {
        throw std::runtime_error("SquareOp expects exactly one input");
    }

    if (inputs[0] == nullptr) {
        throw std::runtime_error("SquareOp received null input");
    }

    parent_ = inputs[0];
    Tensor result = GetBackend(parent_->GetDevice()).Mul(*parent_, *parent_);

    auto output = std::make_shared<Tensor>(std::move(result));
    output->SetGradFn(shared_from_this());

    return output;
}

std::vector<Tensor> SquareOp::backward(const Tensor& grad_output) {
    if (parent_ == nullptr) {
        throw std::runtime_error("SquareOp has no parent");
    }

    Tensor grad_input = GetBackend(
        parent_->GetDevice()).Mul(grad_output, *parent_);

    grad_input = GetBackend(parent_->GetDevice()).MulScalar(grad_input, 2.0f);
    return { std::move(grad_input) };
}

std::vector<std::shared_ptr<Tensor>> SquareOp::GetInputs() const {
    return {parent_};
}