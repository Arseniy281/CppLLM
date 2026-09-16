#pragma once
#include "../Autograd/sub_op.h"
#include "../Autograd/square_op.h"
#include "../Autograd/sum_op.h"
#include "../Tensor/tensor.h"
#include <vector>

class MSELoss {
private:
    SubOp sub_op_;
    SquareOp square_op_;
    SumOp sum_op_;

public:
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& y_pred, 
        const std::shared_ptr<Tensor>& y_true);
};