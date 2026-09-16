#include "gelu.h"
#include "../Tensor/tensor.h"
#include <vector>
#include <memory>
#include <cmath>
#include <numbers>


std::shared_ptr<Tensor> Gelu::forward(
    const std::vector<std::shared_ptr<Tensor>>& inputs) {

    if (inputs.size() != 1 || inputs[0] == nullptr) {
        throw std::runtime_error("Gelu::forward: expected one input");
    }

    parent_ = inputs[0];
    Tensor result = GetBackend(parent_->GetDevice()).Gelu(*parent_);

    auto output = std::make_shared<Tensor>(std::move(result));
    output->SetGradFn(shared_from_this());

    return output;
}

std::vector<Tensor> Gelu::backward( const Tensor& grad_output) {
    Tensor grad_input = GetBackend(parent_->GetDevice()).GeluBackward(*parent_, grad_output);
    return { grad_input };
}

std::vector<std::shared_ptr<Tensor>> Gelu::GetInputs() const {
    return {parent_};
}