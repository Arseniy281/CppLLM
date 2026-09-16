#include "rms_norm_op.h"

#include <stdexcept>
#include <utility>


RMSNormOp::RMSNormOp(const std::shared_ptr<Tensor>& gamma) : gamma_(gamma) {}

std::shared_ptr<Tensor> RMSNormOp::forward(
    const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 1) {
        throw std::runtime_error( "RMSNormOp::forward: expected 1 input");
    }
    if (inputs[0] == nullptr) {
        throw std::runtime_error("RMSNormOp::forward: input is null");
    }
    if (gamma_ == nullptr) {
        throw std::runtime_error("RMSNormOp::forward: gamma is null");
    }

    input_ = inputs[0];
    const std::vector<size_t>& shape = input_->GetShape();

    if (shape.size() != 3) {
        throw std::runtime_error("RMSNormOp::forward: expected 3D input");
    }

    const size_t embed_dim = shape[2];
    if (gamma_->GetShape() != std::vector<size_t>{embed_dim}) {
        throw std::runtime_error("RMSNormOp::forward: invalid gamma shape");
    }

    std::vector<Tensor> results = GetBackend(
        input_->GetDevice()).RMSNormForward(*input_, *gamma_);

    if (results.size() != 3) {
        throw std::runtime_error(
            "RMSNormOp::forward: backend returned invalid number of tensors");
    }

    Tensor result = std::move(results[0]);
    saved_x_norm_ = std::move(results[1]);
    saved_rms_ = std::move(results[2]);
    auto output = std::make_shared<Tensor>(std::move(result));

    output->SetGradFn(shared_from_this());

    return output;
}

std::vector<Tensor> RMSNormOp::backward(const Tensor& grad_output) {
    if (input_ == nullptr) {
        throw std::runtime_error("RMSNormOp::backward: input is null" );
    }

    if (grad_output.GetShape() != input_->GetShape()) {
        throw std::runtime_error("RMSNormOp::backward: invalid gradient shape");
    }

    std::vector<Tensor> results = 
        GetBackend(input_->GetDevice()).RMSNormBackward(
        *input_, *gamma_, saved_x_norm_, saved_rms_, grad_output);

    if (results.size() != 2) {
        throw std::runtime_error("RMSNormOp::backward: backend returned invalid number of tensors");
    }

    Tensor grad_x = std::move(results[0]);
    Tensor grad_gamma = std::move(results[1]);
    gamma_->AddGrad(std::move(grad_gamma));

    return { std::move(grad_x) };
}

std::vector<std::shared_ptr<Tensor>> RMSNormOp::GetInputs() const {
    return { input_ };
}