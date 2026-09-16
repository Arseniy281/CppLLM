#pragma once
#include "../Autograd/add_op.h"
#include "../Autograd/mul_op.h"
#include "../Tensor/tensor.h"
#include <memory>

class LinearLayer {
private:
    std::shared_ptr<Tensor> W_;
    std::shared_ptr<Tensor> b_;
    size_t input_size_;
    size_t  output_size_;

    std::shared_ptr<Tensor> saved_mult_;
    std::shared_ptr<Tensor> saved_added_;

    std::shared_ptr<Tensor> W_m_;
    std::shared_ptr<Tensor> W_v_;
    std::shared_ptr<Tensor> b_m_;
    std::shared_ptr<Tensor> b_v_;
public:
    LinearLayer(size_t in, size_t out, Device device = Device::CPU);
    LinearLayer() = default;

    void UpdateAdamW(float lr, float beta1, float beta2, 
        float eps, float weight_decay, size_t step);

    void ClearGrad();
    void Update(float lr);
    void ScaleGrad(float factor);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x);

    void Save(const std::string& folder, const std::string& name) const;
    void Load(const std::string& folder, const std::string& name);
};