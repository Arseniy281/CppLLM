#include "tensor.h"
#include <iostream>

void Tensor::print() const {
    if (shape_.empty()) {
        std::cout << "[]" << std::endl;
        return;
    }

    if (shape_.size() == 1) {
        for (size_t i = 0; i < size_; ++i) {
            std::cout << data_[i] << " ";
        }
        std::cout << std::endl;
        return;
    }

    if (shape_.size() == 2) {
        size_t cols = shape_[1];
        for (size_t i = 0; i < size_; ++i) {
            std::cout << data_[i] << " ";
            if ((i + 1) % cols == 0) {
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
        return;
    }

    std::cout << "Tensor shape: (";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];
        if (i + 1 < shape_.size()) std::cout << ", ";
    }
    std::cout << ")\n";
    for (size_t i = 0; i < size_; ++i) {
        std::cout << data_[i] << " ";
    }
    std::cout << std::endl;
}


void Tensor::PrintInfo() const {
    std::cout << "Rank: " << rank_ << "\n";
    std::cout << "Shape: ";
    for (const size_t num : shape_) {
        std::cout << num << " ";
    }
    std::cout << "\n";
    std::cout << "Size: " << size_ << "\n";
}

void Tensor::PrintGrad() const {
    if (grad_ != nullptr) {
        grad_->print();
    }
}