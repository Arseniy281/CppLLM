#include "tensor.h"
#include "backend.h"

#include <vector>
#include <cmath>

std::vector<size_t> Tensor::AlignTensors(const std::vector<size_t>& first,
        const std::vector<size_t>& second) {

    size_t rank = std::max(first.size(), second.size());

    std::vector<size_t> aligned_first(rank, 1);
    std::vector<size_t> aligned_second(rank, 1);

    for (size_t i = 0; i < first.size(); i++) {
        aligned_first[rank - first.size() + i] = first[i];
    }

    for (size_t i = 0; i < second.size(); i++) {
        aligned_second[rank - second.size() + i] = second[i];
    }

    for (size_t i = 0; i < rank; i++) {
        if (aligned_first[i] != aligned_second[i] 
            && aligned_first[i] != 1 
            && aligned_second[i] != 1) {

            std::cerr << "\nBROADCAST ERROR\n";

            std::cerr << "first = [";
            for (size_t x : first) {
                std::cerr << x << " ";
            }
            std::cerr << "]\n";

            std::cerr << "second = [";
            for (size_t x : second) {
                std::cerr << x << " ";
            }
            std::cerr << "]\n";

            throw std::runtime_error(
                "You cannot Broadcst this matrixes"
            );
        }
    }

    return aligned_first;
}

std::vector<size_t> Tensor::GetFinalShape(const std::vector<size_t>& first, 
        const std::vector<size_t>& second) {
    if (first.size() != second.size()) {
        throw std::runtime_error("You cannot Broadcast this matrixes");
    }
    std::vector<size_t> final_shape(first.size());
    for (size_t i = 0; i < first.size(); i++) {
        final_shape[i] = std::max(first[i], second[i]);
    }
    return final_shape;
}

size_t Tensor::GetFinalSize(const std::vector<size_t>& final_shape) {
    size_t final_size = 1;
    for (const auto& dim : final_shape) {
        final_size *= dim;
    }
    return final_size;
}

std::vector<size_t> Tensor::IndexToCoord(size_t ind, const std::vector<size_t>& shape) {
    std::vector<size_t> coords(shape.size());
    for (int i = shape.size() - 1; i >= 0; i--) {
        coords[i] = ind % shape[i];
        ind /= shape[i];
    }
    return coords;
}

size_t Tensor::CoordToIndex(const std::vector<size_t>& coord, const std::vector<size_t>& shape) {
    size_t index = 0;
    size_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; i--) {
        index += coord[i] * stride;
        stride *= shape[i];
    }
    return index;
}

size_t Tensor::BroadcastIndex(const std::vector<size_t>& real_shape, const std::vector<size_t>& final_shape, size_t ind) {
    const size_t rank = final_shape.size();
    if (rank == 0) {
        return 0;
    }

    size_t final_stride = 1;
    size_t real_index = 0;
    size_t real_stride = 1;

    for (int i = (int)(rank) - 1; i >= 0; --i) {
        const size_t coord = (ind / final_stride) % final_shape[i];
        const size_t real_coord = (real_shape[i] == 1) ? 0 : coord;
        real_index += real_coord * real_stride;
        final_stride *= final_shape[i];
        real_stride *= real_shape[i];
    }

    return real_index;
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (device_ != other.device_) {
        throw std::runtime_error(
            "Cannot operate on tensors from different devices");
    }

    return GetBackend(device_).Add(*this, other);
}

Tensor& Tensor::operator+=(const Tensor& other) {
    *this = *this + other;
    return *this;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (device_ != other.device_) {
        throw std::runtime_error(
            "Cannot operate on tensors from different devices");
    }

    return GetBackend(device_).Sub(*this, other);
}

Tensor& Tensor::operator-=(const Tensor& other) {
    *this = *this - other;
    return *this;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (device_ != other.device_) {
        throw std::runtime_error(
            "Cannot operate on tensors from different devices");
    }

    return GetBackend(device_).Mul(*this, other);
}

Tensor& Tensor::operator*=(const Tensor& other) {
    *this = *this * other;
    return *this;
}

Tensor Tensor::operator/(const Tensor& other) const {
    if (device_ != other.device_) {
        throw std::runtime_error(
            "Cannot operate on tensors from different devices");
    }

    return GetBackend(device_).Div(*this, other);
}

Tensor& Tensor::operator/=(const Tensor& other) {
    *this = *this / other;
    return *this;
}

Tensor Tensor::operator+(const float num) const {
    return GetBackend(device_).AddScalar(*this, num);
}

Tensor& Tensor::operator+=(const float num) {
    *this = *this + num;
    return *this;
}

Tensor Tensor::operator-(const float num) const {
    return GetBackend(device_).SubScalar(*this, num);
}

Tensor& Tensor::operator-=(const float num) {
    *this = *this - num;
    return *this;
}

Tensor Tensor::operator*(const float num) const {
    return GetBackend(device_).MulScalar(*this, num);
}

Tensor& Tensor::operator*=(const float num) {
    *this = *this * num;
    return *this;
}

Tensor Tensor::operator/(const float num) const {
    return GetBackend(device_).DivScalar(*this, num);
}

Tensor& Tensor::operator/=(const float num) {
    *this = *this / num;
    return *this;
}

Tensor Tensor::operator-() const {
    return GetBackend(device_).Neg(*this);
}

Tensor Tensor::Concatenate(const std::vector<Tensor>& tensors, size_t axis) {
    if (tensors.empty()) {
        throw std::runtime_error("Cannot concatenate empty tensor list");
    }

    Device device = tensors[0].GetDevice();

    for (const auto& tensor : tensors) {
        if (tensor.GetDevice() != device) {
            throw std::runtime_error(
                "Cannot concatenate tensors from different devices");
        }
    }

    return GetBackend(device).Concatenate(tensors, axis);
}

Tensor Tensor::Transpose() const {
    return GetBackend(device_).Transpose(*this);
}


Tensor operator/(float scalar, const Tensor& t) {
    return GetBackend(t.GetDevice()).ScalarDiv(scalar, t);
}