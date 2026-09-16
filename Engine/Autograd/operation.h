#pragma once

#include <vector>
#include <memory>

class Tensor;

class Operation : public std::enable_shared_from_this<Operation> {
public:
    virtual std::shared_ptr<Tensor> forward(const std::vector<std::shared_ptr<Tensor>>& inputs) = 0;
    virtual std::vector<Tensor> backward(const Tensor& grad_output) = 0;
    virtual ~Operation() = default;
    virtual std::vector<std::shared_ptr<Tensor>> GetInputs() const = 0;
};