#include "softmax_op.h"

#include <stdexcept>
#include <utility>

std::shared_ptr<Tensor> SoftmaxOp::forward(
    const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 1) {
        throw std::runtime_error("SoftmaxOp::forward: expected one input");
    }

    if (inputs[0] == nullptr) {
        throw std::runtime_error("SoftmaxOp::forward: null input");
    }

    input_ = inputs[0];
    const std::vector<size_t>& shape = input_->GetShape();
    if (shape.empty()) {
        throw std::runtime_error(
            "SoftmaxOp::forward: input must have at least one dimension");
    }

    Tensor result = GetBackend(input_->GetDevice()).Softmax(*input_);
    output_ = std::move(result);
    auto output = std::make_shared<Tensor>(output_);
    output->SetGradFn(shared_from_this());

    return output;
}

std::vector<Tensor> SoftmaxOp::backward(const Tensor& grad_output) {
    if (input_ == nullptr) {
        throw std::runtime_error("SoftmaxOp::backward: input is null");
    }

    if (grad_output.GetShape() != output_.GetShape()) {
        throw std::runtime_error("SoftmaxOp::backward: shape mismatch");
    }

    Tensor grad_input = GetBackend(
        output_.GetDevice()).SoftmaxBackward(output_, grad_output);

    return { std::move(grad_input) };
}

std::vector<std::shared_ptr<Tensor>> SoftmaxOp::GetInputs() const {
    return {input_};
}