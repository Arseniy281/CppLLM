#include "mul_scalar_op.h"

#include <stdexcept>
#include <utility>

MulScalarOp::MulScalarOp(float scalar) : scalar_(scalar) {}

std::shared_ptr<Tensor> MulScalarOp::forward(
    const std::vector<std::shared_ptr<Tensor>>& inputs) {

    if (inputs.size() != 1) {
        throw std::runtime_error("MulScalarOp::forward: expected one input");
    }

    if (inputs[0] == nullptr) {
        throw std::runtime_error("MulScalarOp::forward: null input");
    }

    input_ = inputs[0];
    Tensor result = (*input_) * scalar_;

    auto output = std::make_shared<Tensor>(std::move(result));
    output->SetGradFn(shared_from_this());

    return output;
}

std::vector<Tensor> MulScalarOp::backward(const Tensor& grad_output) {
    Tensor grad_input = grad_output * scalar_;
    return { grad_input };
}

std::vector<std::shared_ptr<Tensor>> MulScalarOp::GetInputs() const {
    return {input_};
}