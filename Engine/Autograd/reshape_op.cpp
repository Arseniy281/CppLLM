#include "reshape_op.h"
#include "../Tensor/tensor.h"
#include "../Tensor/backend.h"

#include <vector>
#include <memory>
#include <stdexcept>

std::shared_ptr<Tensor> ReshapeOp::forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 1) {
        throw std::runtime_error("ReshapeOp expects exactly one input");
    }

    if (inputs[0] == nullptr) {
        throw std::runtime_error("ReshapeOp received null input");
    }

    input_ = inputs[0];
    original_shape_ = input_->GetShape();

    Tensor result = GetBackend(input_->GetDevice()).Reshape(
        *input_, new_shape_);

    auto output = std::make_shared<Tensor>(std::move(result));
    output->SetGradFn(shared_from_this());

    return output;
}

std::vector<Tensor> ReshapeOp::backward(const Tensor& grad_output) {
    if (input_ == nullptr) {
        throw std::runtime_error("ReshapeOp has no input");
    }

    Tensor grad_input = GetBackend(input_->GetDevice()).Reshape(
        grad_output, original_shape_);

    return { std::move(grad_input) };
}

std::vector<std::shared_ptr<Tensor>> ReshapeOp::GetInputs() const {
    return { input_ };
}