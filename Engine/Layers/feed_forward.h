#pragma once
#include "../Layers/linear_layer.h"
#include "../Autograd/gelu.h"
#include "../Tensor/tensor.h"
#include "../Tensor/device.h"

#include <string>

class FeedForward {
private:
    LinearLayer fc1_;
    LinearLayer fc2_;
public:
    FeedForward(size_t embed_dim, 
        size_t hidden_dim, Device device = Device::CPU);

    void UpdateAdamW(float lr, float beta1,
        float beta2, float eps, float weight_decay, size_t step);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x);
    void Update(float lr);
    void ClearGrad();
    void ScaleGrad(float factor);

    void Save(const std::string& folder) const;
    void Load(const std::string& folder);
};