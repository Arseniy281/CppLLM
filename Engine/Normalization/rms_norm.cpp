#include "rms_norm.h"
#include "../Autograd/rms_norm_op.h"
#include "../Tensor/device.h"


RMSNorm::RMSNorm(size_t embed_dim, Device device) : gamma_(
        std::make_shared<Tensor>(
        std::vector<size_t>{embed_dim}, 1.0f, device)),gamma_m_(
        std::make_shared<Tensor>(
        std::vector<size_t>{embed_dim}, 0.0f, device)), gamma_v_(
        std::make_shared<Tensor>(
        std::vector<size_t>{embed_dim}, 0.0f, device)) {}


void RMSNorm::UpdateAdamW(float lr, float beta1, float beta2,
        float eps, float weight_decay, size_t step) {

    if (gamma_->Grad() == nullptr) { return; }
    GetBackend(gamma_->GetDevice()).AdamW(*gamma_, *gamma_m_, *gamma_v_,
        *gamma_->Grad(), lr, beta1, beta2, eps, 0.0f, step);
}

std::shared_ptr<Tensor> RMSNorm::forward(const std::shared_ptr<Tensor>& x) {
    auto operation = std::make_shared<RMSNormOp>(gamma_);
    return operation->forward({x});
}

void RMSNorm::Update(float lr) {
    if (gamma_->Grad() == nullptr) { return; }

    Tensor update =
        GetBackend(gamma_->GetDevice()).MulScalar(*gamma_->Grad(), lr);

    Tensor new_gamma =
        GetBackend(gamma_->GetDevice()).Sub(*gamma_, update);

    *gamma_ = std::move(new_gamma);
}

void RMSNorm::ClearGrad() {
    gamma_->ClearGrad();
}

void RMSNorm::ScaleGrad(float factor) {
    if (gamma_->Grad() == nullptr) { return; }

    Tensor scaled = GetBackend(
        gamma_->Grad()->GetDevice()).MulScalar(*gamma_->Grad(), factor);

    *gamma_->Grad() = std::move(scaled);
}

void RMSNorm::Save(const std::string& path) const {
    gamma_->SaveTensor(path);
    gamma_m_->SaveTensor(path + "_m");
    gamma_v_->SaveTensor(path + "_v");
}

void RMSNorm::Load(const std::string& path) {
    Device device = gamma_->GetDevice();
    *gamma_ = Tensor::LoadTensor(path, device);
    *gamma_m_ = Tensor::LoadTensor(path + "_m", device);
    *gamma_v_ = Tensor::LoadTensor(path + "_v", device);
}