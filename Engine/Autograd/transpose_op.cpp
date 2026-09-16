#include "transpose_op.h"
#include "../Tensor/tensor.h"

#include <vector>
#include <memory>
#include <stdexcept>

std::shared_ptr<Tensor> TransposeOp::forward(
        const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 1) {
        throw std::runtime_error("TransposeOp expects exactly one input");
    }

    if (inputs[0] == nullptr) {
        throw std::runtime_error("TransposeOp received null input");
    }

    input_ = inputs[0];
    Tensor result = input_->Transpose();
    auto output = std::make_shared<Tensor>(std::move(result));

    output->SetGradFn( shared_from_this() );
    return output;
}

std::vector<Tensor> TransposeOp::backward( const Tensor& grad_output) {
    if (input_ == nullptr) {
        throw std::runtime_error( "TransposeOp has no input" );
    }

    Tensor grad_input = grad_output.Transpose();
    return { std::move(grad_input) };
}

std::vector<std::shared_ptr<Tensor>> TransposeOp::GetInputs() const {
    return { input_ };
}