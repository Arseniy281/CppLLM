#pragma once

#include <memory>
#include <vector>
#include "device.h"
#include "tensor.h"

class Backend {
public:
    virtual ~Backend() = default;

    virtual Tensor Add(const Tensor& a, const Tensor& b) const = 0;
    virtual Tensor Sub(const Tensor& a, const Tensor& b) const = 0;
    virtual Tensor Mul(const Tensor& a, const Tensor& b) const = 0;
    virtual Tensor Div(const Tensor& a, const Tensor& b) const = 0;
    virtual Tensor Sum(const Tensor& input) const = 0;

    virtual Tensor AddScalar(const Tensor& a, float scalar) const = 0;
    virtual Tensor SubScalar(const Tensor& a, float scalar) const = 0;
    virtual Tensor MulScalar(const Tensor& a, float scalar) const = 0;
    virtual Tensor DivScalar(const Tensor& a, float scalar) const = 0;
    virtual Tensor Neg(const Tensor& a) const = 0;

    virtual Tensor Transpose(const Tensor& a) const = 0;
    virtual Tensor Concatenate(
        const std::vector<Tensor>& tensors, size_t axis) const = 0;
    virtual std::vector<Tensor> Split(const Tensor& tensor,
        const std::vector<std::vector<size_t>>& shapes, size_t axis) const = 0;

    virtual Tensor SumAxis(const Tensor& a, int axis) const = 0;
    virtual Tensor Mean(const Tensor& a, int axis) const = 0;

    virtual Tensor MatMul(const Tensor& a, const Tensor& b) const = 0;
    virtual Tensor ScalarDiv(float scalar, const Tensor& t) const = 0;

    virtual Tensor Embedding(
        const Tensor& embeddings, const Tensor& indices) const = 0;
    virtual Tensor EmbeddingBackward(const Tensor& embeddings,
        const Tensor& indices, const Tensor& grad_output) const = 0;
    virtual Tensor Gelu(const Tensor& a) const = 0;
    virtual Tensor GeluBackward(
        const Tensor& input, const Tensor& grad_output) const = 0;
    virtual Tensor ReLU(const Tensor& a) const = 0;
    virtual Tensor ReLUBackward(
        const Tensor& input, const Tensor& grad_output) const = 0;

    virtual std::vector<Tensor> RMSNormForward(
        const Tensor& input, const Tensor& gamma) const = 0;
    virtual std::vector<Tensor> RMSNormBackward(const Tensor& input, 
        const Tensor& gamma, const Tensor& x_norm,
        const Tensor& rms, const Tensor& grad_output) const = 0;

    virtual Tensor RoPE(const Tensor& input, size_t start_pos) const = 0;
    virtual Tensor RoPEBackward(
        const Tensor& grad_output, size_t start_pos) const = 0;

    virtual Tensor Sigmoid(const Tensor& input) const = 0;
    virtual Tensor SigmoidBackward(
        const Tensor& input, const Tensor& grad_output) const = 0;

    virtual Tensor Softmax(const Tensor& input) const = 0;
    virtual Tensor SoftmaxBackward(
        const Tensor& output, const Tensor& grad_output) const = 0;

    virtual Tensor ReduceBroadcast(
        const Tensor& grad, const std::vector<size_t>& target_shape) const = 0;

    virtual Tensor Tanh(const Tensor& input) const = 0;
    virtual Tensor TanhBackward(
        const Tensor& input, const Tensor& grad_output) const = 0;

    virtual Tensor CreateCausalMask(
        size_t query_len, size_t key_len, size_t query_start) const = 0;

    virtual Tensor CrossEntropy(
        const Tensor& logits, const Tensor& targets) const = 0;
    virtual Tensor CrossEntropyBackward(
        const Tensor& logits, const Tensor& targets) const = 0;

    virtual size_t ArgMax(const Tensor& tensor) const = 0;
    virtual Tensor Reshape(
        const Tensor& input, const std::vector<size_t>& new_shape) const = 0;

    virtual void AdamW(Tensor& parameter, Tensor& m, Tensor& v,
        const Tensor& gradient, float lr, float beta1, float beta2,
        float eps, float weight_decay, size_t step) = 0;
};

Backend& GetBackend(Device device);