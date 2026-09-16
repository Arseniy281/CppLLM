#include "mul_op.h"
#include "../Tensor/tensor.h"

#include <vector>
#include <memory>
#include <stdexcept>

std::shared_ptr<Tensor> MulOp::forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 2) {
        throw std::runtime_error("MulOp expects exactly two inputs" );
    }

    if (inputs[0] == nullptr || inputs[1] == nullptr) {
        throw std::runtime_error("MulOp received null input");
    }

    first_ = inputs[0];
    second_ = inputs[1];

    Tensor result = first_->MatMul(*second_);
    auto output = std::make_shared<Tensor>(std::move(result));
    output->SetGradFn(shared_from_this());

    return output;
}

std::vector<Tensor> MulOp::backward(
    const Tensor& grad_output) {
    if (first_ == nullptr || second_ == nullptr) {
        throw std::runtime_error("MulOp has no parents");
    }

    const auto& first_shape = first_->GetShape();
    const auto& second_shape = second_->GetShape();

    if (first_shape.size() == 2 && second_shape.size() == 2) {
        Tensor second_transposed = second_->Transpose();
        Tensor first_transposed = first_->Transpose();
        Tensor grad_first = grad_output.MatMul(second_transposed);
        Tensor grad_second = first_transposed.MatMul(grad_output);

        return {std::move(grad_first), std::move(grad_second) };
    }

    if (first_shape.size() == 3 && second_shape.size() == 2) {
        const size_t batch = first_shape[0];
        const size_t seq_len = first_shape[1];
        const size_t in = first_shape[2];
        const size_t out = second_shape[1];
        const auto& grad_shape = grad_output.GetShape();

        if (grad_shape.size() != 3 ||
            grad_shape[0] != batch ||
            grad_shape[1] != seq_len ||
            grad_shape[2] != out) {

            throw std::runtime_error("MulOp::backward: invalid grad_output shape");
        }

        Tensor second_transposed = second_->Transpose();
        Tensor grad_first = grad_output.MatMul(second_transposed);
        Tensor first_transposed = first_->Transpose();
        Tensor grad_second_per_batch = first_transposed.MatMul(grad_output);
        Tensor grad_second = grad_second_per_batch.SumAxis(0);
        return { std::move(grad_first), std::move(grad_second) };
    }

    if (first_shape.size() == 3 && second_shape.size() == 3) {
        const size_t batch = first_shape[0];
        const size_t rows = first_shape[1];
        const size_t inner = first_shape[2];
        const size_t cols = second_shape[2];
        const auto& grad_shape = grad_output.GetShape();

        if (grad_shape.size() != 3 ||
            grad_shape[0] != batch ||
            grad_shape[1] != rows ||
            grad_shape[2] != cols) {

            throw std::runtime_error("MulOp::backward: invalid grad_output shape");
        }

        Tensor second_transposed = second_->Transpose();
        Tensor grad_first = grad_output.MatMul(second_transposed);
        Tensor first_transposed = first_->Transpose();
        Tensor grad_second = first_transposed.MatMul(grad_output);
        return { std::move(grad_first), std::move(grad_second) };
    }

    throw std::runtime_error("MulOp::backward: unsupported tensor dimensions");
}

std::vector<std::shared_ptr<Tensor>> MulOp::GetInputs() const {
    return { first_, second_ };
}