#include "../Layers/embedding_layer.h"
#include "embedding_op.h"
#include <stdexcept>

std::shared_ptr<Tensor> EmbeddingOp::forward(
        const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 1 || inputs[0] == nullptr) {
        throw std::runtime_error("EmbeddingOp::forward: expected one input");
    }

    indices_ = inputs[0];
    const Tensor& embeddings = layer_->GetEmbeddings();

    if (indices_->GetDevice() != embeddings.GetDevice()) {
        throw std::runtime_error(
            "EmbeddingOp::forward: "
            "indices and embeddings must be on the same device"
        );
    }

    Tensor result = GetBackend(embeddings.GetDevice()).Embedding(
        embeddings, *indices_);

    auto output = std::make_shared<Tensor>(std::move(result));

    output->SetGradFn(shared_from_this());
    return output;
}


std::vector<Tensor> EmbeddingOp::backward(const Tensor& grad_output) {
    if (indices_ == nullptr) {
        throw std::runtime_error("EmbeddingOp::backward: no input");
    }

    layer_->backward(*indices_, grad_output);
    Tensor grad_indices(indices_->GetShape(), 0.0f, indices_->GetDevice());

    return { std::move(grad_indices) };
}

std::vector<std::shared_ptr<Tensor>> EmbeddingOp::GetInputs() const {
    return { indices_ };
}