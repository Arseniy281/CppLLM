#pragma once

#include "transformer_block.h"
#include "../Tensor/device.h"
#include <vector>
#include <memory>

class Transformer {
private:
    std::vector<TransformerBlock> blocks_;
    size_t blocks_count_;

public:
    Transformer() = default;

    Transformer(size_t n, size_t embed_dim, size_t num_heads, 
        size_t hidden_dim, Device device = Device::CPU);
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x);

    void UpdateAdamW(float lr, float beta1, float beta2, 
        float eps, float weight_decay, size_t step);

    void Update(float lr);
    void ClearGrad();
    void ScaleGrad(float factor);

    void Save(const std::string& folder) const;
    void Load(const std::string& folder);

    void ResetCache();
    void SetUseKVCache(bool value);
};