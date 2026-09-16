#include "../Layers/linear_layer.h"
#include "../Tensor/tensor.h"
#include "../Autograd/transpose_op.h"
#include "../Autograd/mul_scalar_op.h"
#include "../Autograd/concatenate_op.h"
#include "../Tensor/device.h"
#include "rope.h"
#include "attention.h"
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <sys/stat.h>
#include <errno.h> 
#include <chrono>

MultiHeadAttention::MultiHeadAttention(size_t embed_dim, size_t num_heads, Device device)
        : embed_dim_(embed_dim),
      num_heads_(num_heads),
      head_dim_(embed_dim / num_heads),
      output_layer_(embed_dim, embed_dim, device) {

    for (size_t i = 0; i < num_heads_; i++) {
        q_layers_.emplace_back(embed_dim_, head_dim_, device);
        k_layers_.emplace_back(embed_dim_, head_dim_, device);
        v_layers_.emplace_back(embed_dim_, head_dim_, device);
        softmax_.emplace_back();
    }
}

void MultiHeadAttention::UpdateAdamW(float lr, float beta1, float beta2,
        float eps, float weight_decay, size_t step) {

    for (auto& layer : q_layers_) {
        layer.UpdateAdamW(lr, beta1, beta2, eps, weight_decay, step);
    }

    for (auto& layer : k_layers_) {
        layer.UpdateAdamW(lr, beta1, beta2, eps, weight_decay, step);
    }

    for (auto& layer : v_layers_) {
        layer.UpdateAdamW(lr, beta1, beta2, eps, weight_decay, step);
    }

    output_layer_.UpdateAdamW(lr, beta1, beta2, eps, weight_decay, step);
}

Tensor MultiHeadAttention::CreateCausalMask(size_t query_len, size_t key_len, 
        size_t query_start, Device device) {
    return GetBackend(device).CreateCausalMask(query_len, key_len, query_start);
}

Tensor MultiHeadAttention::GetLastToken(const Tensor& tensor) {
    std::vector<size_t> shape = tensor.GetShape();
    size_t rank = shape.size();
    size_t seq_len = shape[rank - 2];
    size_t head_dim = shape[rank - 1];
    
    std::vector<size_t> new_shape = shape;
    new_shape[rank - 2] = 1;
    
    Tensor result(new_shape);
    
    for (size_t i = 0; i < result.GetSize(); i++) {
        std::vector<size_t> coords = Tensor::IndexToCoord(i, new_shape);
        std::vector<size_t> src_coords = coords;
        src_coords[rank - 2] = seq_len - 1;
        
        result.at(coords) = tensor.at(src_coords);
    }
    
    return result;
}


std::shared_ptr<Tensor> ApplyMatMul(const std::shared_ptr<Tensor>& first,
        const std::shared_ptr<Tensor>& second) {

    auto operation = std::make_shared<MulOp>();
    return operation->forward({first,second});
}

std::shared_ptr<Tensor> ApplyTranspose(const std::shared_ptr<Tensor>& input) {
    auto operation = std::make_shared<TransposeOp>();
    return operation->forward({input});
}

std::shared_ptr<Tensor> ApplyMulScalar(const std::shared_ptr<Tensor>& input, float scalar) {
    auto operation = std::make_shared<MulScalarOp>(scalar);
    return operation->forward({input});
}

std::shared_ptr<Tensor> ApplyConcatenate(
        const std::vector<std::shared_ptr<Tensor>>& inputs, size_t axis) {
    auto operation = std::make_shared<ConcatenateOp>(axis);
    return operation->forward(inputs);
}

std::shared_ptr<Tensor> ApplyAdd(const std::shared_ptr<Tensor>& first,
        const std::shared_ptr<Tensor>& second) {
    auto operation = std::make_shared<AddOp>();
    return operation->forward({first, second});
}

std::shared_ptr<Tensor> MultiHeadAttention::ComputeAndAssemble(
        const std::vector<std::shared_ptr<Tensor>>& Q,
        const std::vector<std::shared_ptr<Tensor>>& K,
        const std::vector<std::shared_ptr<Tensor>>& V,
        const std::shared_ptr<Tensor>& mask_ptr) {

    std::vector<std::shared_ptr<Tensor>> head_outputs;
    std::vector<Tensor> attention_weights;
    head_outputs.reserve(num_heads_);
    attention_weights.reserve(num_heads_);

    const float scale = 1.0f / std::sqrt(float(head_dim_));

    for (size_t i = 0; i < num_heads_; i++) {
        auto scores = ApplyMatMul(Q[i], ApplyTranspose(K[i]));
        scores = ApplyMulScalar(scores, scale);
        scores = ApplyAdd(scores, mask_ptr);

        auto weights = softmax_[i].forward(scores);
        attention_weights.push_back(*weights);

        head_outputs.push_back(ApplyMatMul(weights, V[i]));
    }

    saved_attention_weights_ = std::move(attention_weights);

    auto concatenated = ApplyConcatenate(head_outputs, 2);
    saved_concatenated_ = *concatenated;

    auto output = output_layer_.forward(concatenated);
    saved_output_ = *output;

    return output;
}

struct MultiHeadAttention::PreparedInputs {
    std::vector<std::shared_ptr<Tensor>> Q;
    std::vector<std::shared_ptr<Tensor>> K;
    std::vector<std::shared_ptr<Tensor>> V;
    std::shared_ptr<Tensor> mask;
};

MultiHeadAttention::PreparedInputs MultiHeadAttention::PrepareNoCache(
        const std::shared_ptr<Tensor>& x) {
    PreparedInputs p;
    p.Q.reserve(num_heads_);
    p.K.reserve(num_heads_);
    p.V.reserve(num_heads_);

    for (size_t i = 0; i < num_heads_; i++) {
        p.Q.push_back(q_layers_[i].forward(x));
        p.K.push_back(k_layers_[i].forward(x));
        p.V.push_back(v_layers_[i].forward(x));
    }

    saved_Q_ = p.Q;
    saved_K_ = p.K;
    saved_V_ = p.V;

    rope_start_pos_ = 0;
    for (size_t i = 0; i < num_heads_; i++) {
        p.Q[i] = ApplyRoPE(p.Q[i], 0);
        p.K[i] = ApplyRoPE(p.K[i], 0);
    }
    saved_Q_rot_ = p.Q;
    saved_K_rot_ = p.K;

    size_t seq_len = x->GetShape()[1];
    Tensor mask = CreateCausalMask(seq_len, seq_len, 0, x->GetDevice());
    mask = mask.Reshape({1, seq_len, seq_len});
    p.mask = std::make_shared<Tensor>(std::move(mask));

    return p;
}

std::shared_ptr<Tensor> MultiHeadAttention::ForwardNoCache(
        const std::shared_ptr<Tensor>& x) {

    auto [Q, K, V, mask_ptr] = PrepareNoCache(x);
    return ComputeAndAssemble(Q, K, V, mask_ptr);
}

void MultiHeadAttention::UpdateKVCache(
        const std::vector<std::shared_ptr<Tensor>>& K,
        const std::vector<std::shared_ptr<Tensor>>& V) {

    if (is_first_token_) {
        is_first_token_ = false;
        kv_cache_K_.clear();
        kv_cache_V_.clear();
        kv_cache_K_.reserve(num_heads_);
        kv_cache_V_.reserve(num_heads_);
        for (size_t i = 0; i < num_heads_; i++) {
            kv_cache_K_.push_back(*K[i]);
            kv_cache_V_.push_back(*V[i]);
        }
    } else {
        for (size_t i = 0; i < num_heads_; i++) {
            kv_cache_K_[i] = Tensor::Concatenate({kv_cache_K_[i], *K[i]}, 1);
            kv_cache_V_[i] = Tensor::Concatenate({kv_cache_V_[i], *V[i]}, 1);
        }
    }
}

MultiHeadAttention::PreparedInputs MultiHeadAttention::PrepareWithCache(
        const std::shared_ptr<Tensor>& x) {
    PreparedInputs p;
    p.Q.reserve(num_heads_);
    p.K.reserve(num_heads_);
    p.V.reserve(num_heads_);

    for (size_t i = 0; i < num_heads_; i++) {
        p.Q.push_back(q_layers_[i].forward(x));
        p.K.push_back(k_layers_[i].forward(x));
        p.V.push_back(v_layers_[i].forward(x));
    }

    saved_Q_ = p.Q;
    saved_K_ = p.K;
    saved_V_ = p.V;

    size_t start_pos = kv_cache_K_.empty() ? 0 : kv_cache_K_[0].GetShape()[1];
    rope_start_pos_ = start_pos;

    for (size_t i = 0; i < num_heads_; i++) {
        p.Q[i] = ApplyRoPE(p.Q[i], start_pos);
        p.K[i] = ApplyRoPE(p.K[i], start_pos);
    }
    saved_Q_rot_ = p.Q;
    saved_K_rot_ = p.K;

    UpdateKVCache(p.K, p.V);

    size_t cur_len     = x->GetShape()[1];
    size_t total_len   = kv_cache_K_[0].GetShape()[1];
    size_t query_start = total_len - cur_len;

    Tensor mask = CreateCausalMask(cur_len, total_len, query_start, x->GetDevice());
    mask = mask.Reshape({1, cur_len, total_len});
    p.mask = std::make_shared<Tensor>(std::move(mask));

    for (size_t i = 0; i < num_heads_; i++) {
        p.K[i] = std::make_shared<Tensor>(kv_cache_K_[i]);
        p.V[i] = std::make_shared<Tensor>(kv_cache_V_[i]);
    }

    return p;
}

std::shared_ptr<Tensor> MultiHeadAttention::ForwardWithCache(
        const std::shared_ptr<Tensor>& x) {

    auto [Q, K, V, mask_ptr] = PrepareWithCache(x);
    return ComputeAndAssemble(Q, K, V, mask_ptr);
}

std::shared_ptr<Tensor> MultiHeadAttention::forward(const std::shared_ptr<Tensor>& x) {
    if (!use_kv_cache_) {
        return ForwardNoCache(x);
    }

    return ForwardWithCache(x);
}

void MultiHeadAttention::Update(float lr) {
    for (auto& layer : q_layers_) layer.Update(lr);
    for (auto& layer : k_layers_) layer.Update(lr);
    for (auto& layer : v_layers_) layer.Update(lr);
    output_layer_.Update(lr);
}

void MultiHeadAttention::ClearGrad() {
    for (auto& layer : q_layers_) layer.ClearGrad();
    for (auto& layer : k_layers_) layer.ClearGrad();
    for (auto& layer : v_layers_) layer.ClearGrad();
    output_layer_.ClearGrad();
}

void MultiHeadAttention::ScaleGrad(float factor) {
    for (auto& layer : q_layers_) layer.ScaleGrad(factor);
    for (auto& layer : k_layers_) layer.ScaleGrad(factor);
    for (auto& layer : v_layers_) layer.ScaleGrad(factor);
    output_layer_.ScaleGrad(factor);
}

void MultiHeadAttention::Save(const std::string& folder) const {
    if (mkdir(folder.c_str(), 0777) != 0 && errno != EEXIST) {
        throw std::runtime_error("Cannot create directory: " + folder);
    }

    for (size_t i = 0; i < num_heads_; i++) {
        q_layers_[i].Save(folder, "q_" + std::to_string(i));
        k_layers_[i].Save(folder, "k_" + std::to_string(i));
        v_layers_[i].Save(folder, "v_" + std::to_string(i));
    }
    output_layer_.Save(folder, "output");
}

void MultiHeadAttention::Load(const std::string& folder) {
    for (size_t i = 0; i < num_heads_; i++) {
        q_layers_[i].Load(folder, "q_" + std::to_string(i));
        k_layers_[i].Load(folder, "k_" + std::to_string(i));
        v_layers_[i].Load(folder, "v_" + std::to_string(i));
    }
    output_layer_.Load(folder, "output");
}

void MultiHeadAttention::ResetCache() {
    is_first_token_ = true;
    kv_cache_K_.clear();
    kv_cache_V_.clear();
    rope_start_pos_ = 0;
}

size_t MultiHeadAttention::GetRopeStartPos() const {
    return rope_start_pos_;
}

void MultiHeadAttention::SetUseKVCache(bool value) {
    use_kv_cache_ = value;
}