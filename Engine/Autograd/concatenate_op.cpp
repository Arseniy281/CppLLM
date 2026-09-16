#include "concatenate_op.h"

#include <stdexcept>
#include <cstring>

ConcatenateOp::ConcatenateOp(size_t axis) : axis_(axis) {}


std::shared_ptr<Tensor> ConcatenateOp::forward(
    const std::vector<std::shared_ptr<Tensor>>& inputs) {

    if (inputs.empty()) {
        throw std::runtime_error("ConcatenateOp::forward: no inputs");
    }

    for (const auto& input : inputs) {
        if (input == nullptr) {
            throw std::runtime_error("ConcatenateOp::forward: null input");
        }
    }

    inputs_ = inputs;

    std::vector<Tensor> tensors;
    tensors.reserve(inputs.size());

    for (const auto& input : inputs) {
        tensors.push_back(*input);
    }

    Tensor result = Tensor::Concatenate(tensors, axis_);
    auto output = std::make_shared<Tensor>(std::move(result));
    output->SetGradFn(shared_from_this());

    return output;
}

std::vector<Tensor> ConcatenateOp::backward(const Tensor& grad_output) {

    if (inputs_.empty()) {
        throw std::runtime_error("ConcatenateOp::backward: no saved inputs");
    }

    std::vector<std::vector<size_t>> shapes;
    shapes.reserve(inputs_.size());

    for (const auto& input : inputs_) {
        shapes.push_back(input->GetShape());
    }

    return GetBackend(grad_output.GetDevice()).Split(
        grad_output, shapes, axis_);
}

std::vector<std::shared_ptr<Tensor>> ConcatenateOp::GetInputs() const {
    return inputs_;
}