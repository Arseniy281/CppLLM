#include "softmax.h"
#include "../Autograd/softmax_op.h"

std::shared_ptr<Tensor> Softmax::forward(const std::shared_ptr<Tensor>& input) {
    auto operation = std::make_shared<SoftmaxOp>();
    return operation->forward({input});
}