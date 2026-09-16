#include "../Autograd/embedding_op.h"
#include "embedding_layer.h"
#include "../Tensor/tensor.h"

#include <memory>
#include <stdexcept>

EmbeddingLayer::EmbeddingLayer(size_t vocab_size,
     size_t embedding_dim, Device device) : vocab_size_(vocab_size),
      embedding_dim_(embedding_dim), embeddings_(
        Tensor::Random({vocab_size, embedding_dim}, -0.1f, 0.1f, device)),
        grad_(std::make_shared<Tensor>(
            std::vector<size_t>{vocab_size, embedding_dim}, 0.0f, device)), 
            m_(std::make_shared<Tensor>(
            std::vector<size_t>{vocab_size, embedding_dim}, 0.0f, device)), 
            v_(std::make_shared<Tensor>(
            std::vector<size_t>{vocab_size, embedding_dim}, 0.0f, device)) {}


void EmbeddingLayer::UpdateAdamW(float lr, float beta1, float beta2, 
        float eps, float weight_decay, size_t step) {

    if (grad_ == nullptr) { return; }
    GetBackend(embeddings_.GetDevice()).AdamW(embeddings_, *m_, *v_, *grad_,
        lr, beta1, beta2, eps, weight_decay, step);
}


std::shared_ptr<Tensor> EmbeddingLayer::forward(
        const std::shared_ptr<Tensor>& indices) {
    if (indices == nullptr) {
        throw std::runtime_error("EmbeddingLayer::forward: indices is null");
    }

    if (indices->GetDevice() != embeddings_.GetDevice()) {
        throw std::runtime_error(
            "EmbeddingLayer::forward: "
            "indices and embeddings must be on the same device"
        );
    }

    auto operation = std::make_shared<EmbeddingOp>(this);
    return operation->forward({indices});
}


void EmbeddingLayer::backward(const Tensor& indices, const Tensor& grad_output) {
    if (indices.GetDevice() != embeddings_.GetDevice()) {
        throw std::runtime_error(
            "EmbeddingLayer::backward: "
            "indices and embeddings must be on the same device"
        );
    }

    if (grad_output.GetDevice() != embeddings_.GetDevice()) {
        throw std::runtime_error(
            "EmbeddingLayer::backward: "
            "grad_output and embeddings must be on the same device"
        );
    }

    Tensor grad = GetBackend(embeddings_.GetDevice()).EmbeddingBackward(
        embeddings_, indices, grad_output);

    *grad_ += grad;
}


void EmbeddingLayer::ClearGrad() {
    *grad_ = Tensor({vocab_size_, embedding_dim_}, 0.0f, grad_->GetDevice());
}


void EmbeddingLayer::ScaleGrad(float factor) {
    Tensor scaled = GetBackend(grad_->GetDevice()).MulScalar(*grad_, factor);
    *grad_ = std::move(scaled);
}


void EmbeddingLayer::Update(float lr) {
    Tensor update = GetBackend(embeddings_.GetDevice()).MulScalar(*grad_, lr);
    Tensor new_embeddings = 
        GetBackend(embeddings_.GetDevice()).Sub(embeddings_, update);

    embeddings_ = std::move(new_embeddings);
}


void EmbeddingLayer::Save(const std::string& path) const {
    embeddings_.SaveTensor(path);
    m_->SaveTensor(path + "_m");
    v_->SaveTensor(path + "_v");
}


void EmbeddingLayer::Load(const std::string& path) {
    Device device = embeddings_.GetDevice();

    embeddings_ = Tensor::LoadTensor(path, device);
    *m_ = Tensor::LoadTensor(path + "_m", device);
    *v_ = Tensor::LoadTensor(path + "_v", device);

    if (embeddings_.GetShape().size() != 2 ||
        embeddings_.GetShape()[0] != vocab_size_ ||
        embeddings_.GetShape()[1] != embedding_dim_) {

        throw std::runtime_error(
            "EmbeddingLayer::Load: "
            "loaded tensor has invalid shape"
        );
    }

    if (m_->GetShape() != embeddings_.GetShape() ||
        v_->GetShape() != embeddings_.GetShape()) {

        throw std::runtime_error(
            "EmbeddingLayer::Load: "
            "Adam states have invalid shape"
        );
    }

    grad_ = std::make_shared<Tensor>(
        Tensor({vocab_size_, embedding_dim_}, 0.0f, device)
    );
}


const Tensor& EmbeddingLayer::GetEmbeddings() const {
    return embeddings_;
}