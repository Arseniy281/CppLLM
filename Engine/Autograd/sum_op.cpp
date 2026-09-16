#include "sum_op.h"
#include "../Tensor/tensor.h"
#include "../Tensor/backend.h"

#include <vector>
#include <memory>
#include <stdexcept>

std::shared_ptr<Tensor> SumOp::forward(
        const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 1) {
        throw std::runtime_error( "SumOp expects exactly one input");
    }

    if (inputs[0] == nullptr) {
        throw std::runtime_error("SumOp received null input");
    }

    parent_ = inputs[0];
    Tensor result = GetBackend(parent_->GetDevice()).Sum(*parent_);

    auto output = std::make_shared<Tensor>(std::move(result));
    output->SetGradFn(shared_from_this());

    return output;
}

std::vector<Tensor> SumOp::backward( const Tensor& grad_output) {
    if (parent_ == nullptr) {
        throw std::runtime_error("SumOp has no parent");
    }

    if (grad_output.GetSize() != 1) {
        throw std::runtime_error(
            "SumOp::backward: grad_output must contain one element");
    }

    Tensor grad_input(parent_->GetShape(), 1.0f);
    Tensor grad = grad_input * grad_output;
    return { std::move(grad) };
}

std::vector<std::shared_ptr<Tensor>>SumOp::GetInputs() const {
    return { parent_ };
}