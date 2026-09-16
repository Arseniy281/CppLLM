#pragma once

#include "../Tensor/tensor.h"
#include "backend.h"
#include <memory>
#include <vector>
#include <cublas_v2.h>

class CUDABackend : public Backend {
private:
    Tensor MatMul2D(const Tensor& A, const Tensor& B) const;
    Tensor MatMul3D2D(const Tensor& A, const Tensor& B) const;
    Tensor MatMulBatched(const Tensor& A, const Tensor& B) const;

    cublasHandle_t handle_;
public:
    CUDABackend();
    ~CUDABackend();

    Tensor Add(const Tensor& a, const Tensor& b) const override;
    Tensor Sub(const Tensor& a, const Tensor& b) const override;
    Tensor Mul(const Tensor& a, const Tensor& b) const override;
    Tensor Div(const Tensor& a, const Tensor& b) const override;
    Tensor Sum(const Tensor& input) const override;

    Tensor AddScalar(const Tensor& a, float scalar) const override;
    Tensor SubScalar(const Tensor& a, float scalar) const override;
    Tensor MulScalar(const Tensor& a, float scalar) const override;
    Tensor DivScalar(const Tensor& a, float scalar) const override;
    Tensor Neg(const Tensor& a) const override;

    Tensor Transpose(const Tensor& a) const override;
    Tensor Concatenate(const std::vector<Tensor>& tensors, 
        size_t axis) const override;
    std::vector<Tensor> Split(const Tensor& tensor,
        const std::vector<std::vector<size_t>>& shapes, 
        size_t axis) const override;

    Tensor SumAxis(const Tensor& a, int axis) const override;
    Tensor Mean(const Tensor& a, int axis) const override;

    Tensor MatMul(const Tensor& a, const Tensor& b) const override;
    Tensor ScalarDiv(float scalar, const Tensor& t) const override;

    Tensor Embedding(const Tensor& embeddings, 
        const Tensor& indices) const override;
    Tensor EmbeddingBackward(const Tensor& embeddings, const Tensor& indices, 
        const Tensor& grad_output) const override;
    Tensor Gelu(const Tensor& a) const override;
    Tensor GeluBackward(const Tensor& input, 
        const Tensor& grad_output) const override;
    Tensor ReLU(const Tensor& a) const override;
    Tensor ReLUBackward(const Tensor& input, 
        const Tensor& grad_output) const override;

    std::vector<Tensor> RMSNormForward(const Tensor& input, 
        const Tensor& gamma) const override;
    std::vector<Tensor> RMSNormBackward(const Tensor& input, const Tensor& gamma,
        const Tensor& x_norm, const Tensor& rms, 
        const Tensor& grad_output) const override;

    Tensor RoPE(const Tensor& input, size_t start_pos) const override;
    Tensor RoPEBackward(const Tensor& grad_output, 
        size_t start_pos) const override;

    Tensor Softmax(const Tensor& input) const override;
    Tensor SoftmaxBackward(const Tensor& output, 
        const Tensor& grad_output) const override;

    Tensor ReduceBroadcast(const Tensor& grad, 
        const std::vector<size_t>& target_shape) const override;

    Tensor Tanh(const Tensor& input) const override;
    Tensor TanhBackward(const Tensor& input, 
        const Tensor& grad_output) const override;

    Tensor CreateCausalMask(size_t query_len, size_t key_len, 
        size_t query_start) const override;

    Tensor CrossEntropy(const Tensor& logits, 
        const Tensor& targets) const override;
    Tensor CrossEntropyBackward(const Tensor& logits, 
        const Tensor& targets) const override;

    size_t ArgMax(const Tensor& tensor) const override;
    Tensor Sigmoid(const Tensor& input) const override;
    Tensor SigmoidBackward(const Tensor& input, 
        const Tensor& grad_output) const override;
    Tensor Reshape(const Tensor& input, 
        const std::vector<size_t>& new_shape) const override;

    void AdamW(Tensor& parameter, Tensor& m, Tensor& v, const Tensor& gradient,
        float lr, float beta1, float beta2, float eps,
        float weight_decay, size_t step) override;
};