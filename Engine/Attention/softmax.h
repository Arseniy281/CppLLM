#pragma once

#include "../Tensor/tensor.h"
#include <memory>

class Softmax {
public:
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input);
};