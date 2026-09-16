#pragma once
#include "../Tensor/tensor.h"
#include "../Layers/linear_layer.h"
#include "softmax.h"
#include <iostream>
#include <memory>
#include <vector>
#include "../Tensor/device.h"

class MultiHeadAttention {
private:
    std::vector<LinearLayer> q_layers_;
    std::vector<LinearLayer> k_layers_;
    std::vector<LinearLayer> v_layers_;
    
    size_t embed_dim_;
    size_t head_dim_;
    size_t num_heads_;

    std::vector<Softmax> softmax_;
    
    LinearLayer output_layer_;

    std::vector<std::shared_ptr<Tensor>> saved_Q_;
    std::vector<std::shared_ptr<Tensor>> saved_K_;
    std::vector<std::shared_ptr<Tensor>> saved_V_;
    std::vector<std::shared_ptr<Tensor>> saved_Q_rot_;
    std::vector<std::shared_ptr<Tensor>> saved_K_rot_;
    std::vector<Tensor> saved_attention_weights_;
    Tensor saved_concatenated_;
    Tensor saved_output_;
    Tensor saved_mask_;

    std::vector<Tensor> kv_cache_K_;
    std::vector<Tensor> kv_cache_V_;
    bool use_kv_cache_ = false;
    bool is_first_token_ = true;

    size_t rope_start_pos_ = 0;

    Tensor GetLastToken(const Tensor& tensor);
    std::shared_ptr<Tensor> ForwardNoCache(const std::shared_ptr<Tensor>& x);
    std::shared_ptr<Tensor> ForwardWithCache(const std::shared_ptr<Tensor>& x);

    struct PreparedInputs;

    PreparedInputs PrepareNoCache(const std::shared_ptr<Tensor>& x);
    PreparedInputs PrepareWithCache(const std::shared_ptr<Tensor>& x);
    void UpdateKVCache(const std::vector<std::shared_ptr<Tensor>>& K,
        const std::vector<std::shared_ptr<Tensor>>& V);
    
    std::shared_ptr<Tensor> ComputeAndAssemble(
        const std::vector<std::shared_ptr<Tensor>>& Q,
        const std::vector<std::shared_ptr<Tensor>>& K,
        const std::vector<std::shared_ptr<Tensor>>& V,
        const std::shared_ptr<Tensor>& mask_ptr);
public:
    MultiHeadAttention(size_t embed_dim, size_t num_heads = 1, Device device = Device::CPU);

    void UpdateAdamW(float lr, float beta1, float beta2, float eps, float weight_decay, size_t step);

    Tensor CreateCausalMask(size_t query_len, size_t key_len, size_t query_start, Device device);
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x);

    void Update(float lr);
    void ClearGrad();
    void ScaleGrad(float factor);

    void Save(const std::string& folder) const;
    void Load(const std::string& folder);

    void ResetCache();
    size_t GetRopeStartPos() const;
    void SetUseKVCache(bool value);
};