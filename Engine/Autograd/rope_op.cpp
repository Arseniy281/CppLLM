#include "rope_op.h"
#include <stdexcept>
#include <utility>

std::shared_ptr<Tensor> RopeOp::forward(
        const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 1) {
        throw std::runtime_error("RopeOp::forward: expected one input");
    }

    if (inputs[0] == nullptr) {
        throw std::runtime_error("RopeOp::forward: null input");
    }

    input_ = inputs[0];
    Tensor result = GetBackend(input_->GetDevice()).RoPE(*input_, start_pos_);

    auto output = std::make_shared<Tensor>(std::move(result));
    output->SetGradFn(shared_from_this());
    return output;
}

std::vector<Tensor> RopeOp::backward(const Tensor& grad_output) {
    Tensor grad_input = GetBackend(input_->GetDevice()).RoPEBackward(
        grad_output, start_pos_);
    return { std::move(grad_input) };
}

std::vector<std::shared_ptr<Tensor>> RopeOp::GetInputs() const {
    return {input_};
}