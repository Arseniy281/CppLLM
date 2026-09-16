#include "rope.h"
#include "../Tensor/tensor.h"
#include "../Autograd/rope_op.h"
#include <iostream>
#include <cmath>

Tensor RoPE(const Tensor& tensor, size_t start_pos) {
    size_t batch = tensor.GetShape()[0];
    size_t seq_len = tensor.GetShape()[1];
    size_t head_dim = tensor.GetShape()[2];

    Tensor result = tensor;

    const float* input = tensor.Data();
    float* output = result.Data();

    for (size_t pos = 0; pos < seq_len; pos++) {
        size_t actual_pos = start_pos + pos;
        size_t pos_offset = pos * head_dim;

        for (size_t i = 0; i < head_dim; i += 2) {
            float theta = (float)actual_pos *
                std::pow(10000.0f, -(float)i / head_dim);

            float cos_theta = std::cos(theta);
            float sin_theta = std::sin(theta);

            for (size_t b = 0; b < batch; b++) {
                size_t index = b * seq_len * head_dim + pos_offset + i;

                float x = input[index];
                float y = input[index + 1];

                output[index] = x * cos_theta - y * sin_theta;
                output[index + 1] = x * sin_theta + y * cos_theta;
            }
        }
    }

    return result;
}

Tensor RoPE_backward(const Tensor& tensor, size_t start_pos) {
    size_t batch = tensor.GetShape()[0];
    size_t seq_len = tensor.GetShape()[1];
    size_t head_dim = tensor.GetShape()[2];

    Tensor result = tensor;

    const float* input = tensor.Data();
    float* output = result.Data();

    for (size_t pos = 0; pos < seq_len; pos++) {
        size_t actual_pos = start_pos + pos;

        size_t pos_offset = pos * head_dim;

        for (size_t i = 0; i < head_dim; i += 2) {
            float theta = (float)actual_pos * std::pow(
                10000.0f, -(float)i / head_dim);

            float cos_theta = std::cos(theta);
            float sin_theta = std::sin(theta);

            for (size_t b = 0; b < batch; b++) {
                size_t index = b * seq_len * head_dim + pos_offset + i;

                float x = input[index];
                float y = input[index + 1];

                output[index] = x * cos_theta + y * sin_theta;
                output[index + 1] = -x * sin_theta + y * cos_theta;
            }
        }
    }

    return result;
}

std::shared_ptr<Tensor> ApplyRoPE(const std::shared_ptr<Tensor>& tensor,
        size_t start_pos) {

    auto operation = std::make_shared<RopeOp>(start_pos);
    return operation->forward({tensor});
}