#include "sigmoid.h"
#include <stdexcept>
#include <utility>

std::shared_ptr<Tensor> Sigmoid::forward(
        const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 1) {
        throw std::runtime_error("Sigmoid::forward: expected one input");
    }

    if (inputs[0] == nullptr) {
        throw std::runtime_error("Sigmoid::forward: null input");
    }

    parent_ = inputs[0];
    Tensor result = GetBackend(parent_->GetDevice()).Sigmoid(*parent_);

    auto output = std::make_shared<Tensor>(std::move(result));
    output->SetGradFn(shared_from_this());

    return output;
}

std::vector<Tensor> Sigmoid::backward(const Tensor& grad_output) {
    if (parent_ == nullptr) {
        throw std::runtime_error("Sigmoid::backward: input is null");
    }

    if (grad_output.GetShape() != parent_->GetShape()) {
        throw std::runtime_error(
            "Sigmoid::backward: invalid grad_output shape");
    }

    Tensor grad_input = GetBackend(
        parent_->GetDevice()).SigmoidBackward(*parent_, grad_output);
    return { std::move(grad_input) };
}

std::vector<std::shared_ptr<Tensor>> Sigmoid::GetInputs() const {
    return { parent_ };
}