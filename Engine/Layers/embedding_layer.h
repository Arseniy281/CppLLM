#pragma once

#include "../Tensor/tensor.h"
#include <memory>
#include <string>

class EmbeddingLayer {
private:
    size_t vocab_size_;
    size_t embedding_dim_;

    Tensor embeddings_;
    std::shared_ptr<Tensor> grad_;

    std::shared_ptr<Tensor> last_indices_;
    std::shared_ptr<Tensor> last_output_;

    std::shared_ptr<Tensor> m_;
    std::shared_ptr<Tensor> v_;

public:
    EmbeddingLayer(size_t vocab_size, size_t embedding_dim, Device device = Device::CPU);

    void UpdateAdamW(float lr, float beta1, float beta2, float eps, float weight_decay, size_t step);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& indices);
    void backward(const Tensor& indices, const Tensor& grad_output);

    void Update(float lr);
    void ClearGrad();
    void ScaleGrad(float factor);

    void Save(const std::string& path) const;
    void Load(const std::string& path);

    const Tensor& GetEmbeddings() const;
};