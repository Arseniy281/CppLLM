#pragma once

#include "../Tensor/tensor.h"
#include <memory>

Tensor RoPE(const Tensor& tensor, size_t start_pos);
Tensor RoPE_backward(const Tensor& tensor, size_t start_pos);

std::shared_ptr<Tensor> ApplyRoPE(const std::shared_ptr<Tensor>& tensor,size_t start_pos);