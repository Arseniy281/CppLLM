#include "relu.h"

#include <stdexcept>
#include <utility>

std::shared_ptr<Tensor> ReLU::forward(
        const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 1 || inputs[0] == nullptr) {
        throw std::runtime_error("ReLU::forward: expected one input");
    }

    parent_ = inputs[0];
    Tensor result = GetBackend(parent_->GetDevice()).ReLU(*parent_);
    auto output = std::make_shared<Tensor>(std::move(result));

    output->SetGradFn(shared_from_this());
    return output;
}

std::vector<Tensor> ReLU::backward(const Tensor& grad_output) {
    Tensor grad_input = GetBackend(parent_->GetDevice()).ReLUBackward(
        *parent_, grad_output);

    return { grad_input };
}

std::vector<std::shared_ptr<Tensor>> ReLU::GetInputs() const {
    return {parent_};
}