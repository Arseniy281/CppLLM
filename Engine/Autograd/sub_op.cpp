#include "sub_op.h"
#include "../Tensor/tensor.h"
#include "../Tensor/backend.h"

#include <vector>
#include <memory>
#include <stdexcept>

std::shared_ptr<Tensor> SubOp::forward(
    const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 2) {
        throw std::runtime_error("SubOp expects exactly two inputs");
    }

    if (inputs[0] == nullptr || inputs[1] == nullptr) {
        throw std::runtime_error("SubOp received null input");
    }

    first_ = inputs[0];
    second_ = inputs[1];

    Tensor result = GetBackend(first_->GetDevice()).Sub(*first_, *second_);

    auto output = std::make_shared<Tensor>(std::move(result));
    output->SetGradFn(shared_from_this());

    return output;
}

std::vector<Tensor> SubOp::backward(const Tensor& grad_output) {
    if (first_ == nullptr || second_ == nullptr) {
        throw std::runtime_error("SubOp has no parents");
    }

    Device device = first_->GetDevice();
    Tensor grad_first = 
        GetBackend(device).ReduceBroadcast(grad_output, first_->GetShape());

    Tensor grad_second = GetBackend(device).Neg(grad_output);
    grad_second = 
        GetBackend(device).ReduceBroadcast(grad_second, second_->GetShape());

    return { std::move(grad_first), std::move(grad_second) };
}

std::vector<std::shared_ptr<Tensor>> SubOp::GetInputs() const {
    return {first_, second_};
}