#include "feed_forward.h"
#include "../Tensor/device.h"

#include <memory>
#include <string>
#include <sys/stat.h>
#include <errno.h>

FeedForward::FeedForward(size_t embed_dim, size_t hidden_dim, Device device)
    : fc1_(embed_dim, hidden_dim, device), fc2_(hidden_dim, embed_dim, device) {}


void FeedForward::UpdateAdamW(float lr, float beta1, float beta2, 
        float eps, float weight_decay, size_t step) {

    fc1_.UpdateAdamW(lr, beta1, beta2, eps, weight_decay, step);
    fc2_.UpdateAdamW(lr, beta1, beta2, eps, weight_decay, step);
}

std::shared_ptr<Tensor> FeedForward::forward(
        const std::shared_ptr<Tensor>& x) {

    auto hidden = fc1_.forward(x);

    auto gelu_op = std::make_shared<Gelu>();
    auto activated = gelu_op->forward({hidden});

    return fc2_.forward(activated);
}

void FeedForward::Update(float lr) {
    fc1_.Update(lr);
    fc2_.Update(lr);
}

void FeedForward::ClearGrad() {
    fc1_.ClearGrad();
    fc2_.ClearGrad();
}

void FeedForward::ScaleGrad(float factor) {
    fc1_.ScaleGrad(factor);
    fc2_.ScaleGrad(factor);
}

void FeedForward::Save(const std::string& folder) const {
    if (mkdir(folder.c_str(), 0777) != 0 && errno != EEXIST) {
        throw std::runtime_error("Cannot create directory: " + folder);
    }

    fc1_.Save(folder, "fc1");
    fc2_.Save(folder, "fc2");
}

void FeedForward::Load(const std::string& folder) {
    fc1_.Load(folder, "fc1");
    fc2_.Load(folder, "fc2");
}