#pragma once

#include "../Tensor/tensor.h"
#include <memory>
#include <string>
#include "../Tensor/device.h"


class RMSNorm {
private:
    std::shared_ptr<Tensor> gamma_;
    std::shared_ptr<Tensor> gamma_m_;
    std::shared_ptr<Tensor> gamma_v_;

public:
    RMSNorm(size_t embed_dim, Device device = Device::CPU);

    void UpdateAdamW(float lr, float beta1, float beta2,
        float eps, float weight_decay, size_t step);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x);

    void Update(float lr);
    void ClearGrad();
    void ScaleGrad(float factor);

    void Save(const std::string& path) const;
    void Load(const std::string& path);
};