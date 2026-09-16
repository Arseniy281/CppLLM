#include "../Tensor/tensor.h"

#include <algorithm>
#include <climits>
#include <stdexcept>

Tensor Tensor::MatMul(const Tensor& other) const {
    if (device_ != other.device_) {
        throw std::runtime_error("MatMul: tensors must be on the same device");
    }

    return GetBackend(device_).MatMul(*this, other);
}