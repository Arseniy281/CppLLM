#include "../Autograd/add_op.h"
#include "../Autograd/mul_op.h"
#include "../Tensor/tensor.h"
#include "linear_layer.h"

#if ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include <stdexcept>
#include <string>

#include <cmath>
#include <memory>

static std::shared_ptr<Tensor> CreateAdamState(const Tensor& parameter) {
    return std::make_shared<Tensor>(
        parameter.GetShape(), 0.0f, parameter.GetDevice());
}

LinearLayer::LinearLayer(size_t in, size_t out, Device device)
        : input_size_(in), output_size_(out) {

    float limit = std::sqrt(6.0f / (float)(in + out));

    if (device == Device::CPU) {
        W_ = std::make_shared<Tensor>(Tensor::Random({in, out}, -limit, limit));
        b_ = std::make_shared<Tensor>(Tensor({1, out}, 0.0f));
    } else {
        Tensor cpu_weights = Tensor::Random({in, out}, -limit, limit);

        W_ = std::make_shared<Tensor>(std::vector<size_t>{in, out}, Device::CUDA);

#if ENABLE_CUDA

        cudaError_t error = cudaMemcpy(W_->Data(), cpu_weights.Data(),
            cpu_weights.GetSize() * sizeof(float), cudaMemcpyHostToDevice);

        if (error != cudaSuccess) {
            throw std::runtime_error(std::string(
                "LinearLayer: failed to copy "
                "weights to CUDA: ")
                + cudaGetErrorString(error)
            );
        }

#else

        std::copy(cpu_weights.Data(),
            cpu_weights.Data() + cpu_weights.GetSize(), W_->Data());

#endif

        b_ = std::make_shared<Tensor>(
            std::vector<size_t>{1, out}, 0.0f, Device::CUDA);
    }

    W_m_ = CreateAdamState(*W_);
    W_v_ = CreateAdamState(*W_);

    b_m_ = CreateAdamState(*b_);
    b_v_ = CreateAdamState(*b_);
}

void LinearLayer::UpdateAdamW(float lr, float beta1, float beta2,
        float eps, float weight_decay, size_t step) {

    if (W_->Grad() != nullptr) {
        GetBackend(W_->GetDevice()).AdamW(*W_, *W_m_, *W_v_, *W_->Grad(),
                                lr, beta1, beta2, eps, weight_decay, step);
    }

    if (b_->Grad() != nullptr) {
        GetBackend(b_->GetDevice()).AdamW(*b_, *b_m_, *b_v_, *b_->Grad(),
                                        lr, beta1, beta2, eps, 0.0f, step);
    }
}

void LinearLayer::ClearGrad() {
    W_->ClearGrad();
    b_->ClearGrad();
}

void LinearLayer::Update(float lr) {
    if (W_->Grad() != nullptr) {
        Tensor update = GetBackend(W_->GetDevice()).MulScalar(*W_->Grad(), lr);
        Tensor new_weights = GetBackend(W_->GetDevice()).Sub(*W_, update);
        *W_ = std::move(new_weights);
    }

    if (b_->Grad() != nullptr) {
        Tensor update = GetBackend(b_->GetDevice()).MulScalar(*b_->Grad(), lr);
        Tensor new_bias = GetBackend(b_->GetDevice()).Sub(*b_, update);
        *b_ = std::move(new_bias);
    }
}

void LinearLayer::ScaleGrad(float factor) {
    if (W_->Grad() != nullptr) {
        Tensor scaled =GetBackend(
            W_->Grad()->GetDevice()).MulScalar(*W_->Grad(), factor);

        *W_->Grad() = std::move(scaled);
    }

    if (b_->Grad() != nullptr) {
        Tensor scaled = GetBackend(
            b_->Grad()->GetDevice()).MulScalar(*b_->Grad(), factor);
        *b_->Grad() = std::move(scaled);
    }
}

std::shared_ptr<Tensor> LinearLayer::forward(const std::shared_ptr<Tensor>& x) {
    auto mul_op = std::make_shared<MulOp>();
    saved_mult_ = mul_op->forward({x, W_});
    auto add_op = std::make_shared<AddOp>();
    saved_added_ = add_op->forward({saved_mult_, b_});

    return saved_added_;
}

void LinearLayer::Save(const std::string& folder, const std::string& name) const {
    W_->SaveTensor(folder + "/" + name + "_W");
    b_->SaveTensor(folder + "/" + name + "_b");

    W_m_->SaveTensor(folder + "/" + name + "_W_m");
    W_v_->SaveTensor(folder + "/" + name + "_W_v");

    b_m_->SaveTensor(folder + "/" + name + "_b_m");
    b_v_->SaveTensor(folder + "/" + name + "_b_v");
}

void LinearLayer::Load(const std::string& folder, const std::string& name) {
    Device device = W_->GetDevice();
    *W_ = Tensor::LoadTensor(folder + "/" + name + "_W", device);
    *b_ = Tensor::LoadTensor(folder + "/" + name + "_b", device);
    *W_m_ = Tensor::LoadTensor(folder + "/" + name + "_W_m", device);
    *W_v_ = Tensor::LoadTensor(folder + "/" + name + "_W_v", device);
    *b_m_ = Tensor::LoadTensor(folder + "/" + name + "_b_m", device);
    *b_v_ = Tensor::LoadTensor(folder + "/" + name + "_b_v", device);
}
