#include "mse_loss.h"
#include "../Tensor/tensor.h"

std::shared_ptr<Tensor> MSELoss::forward(const std::shared_ptr<Tensor>& y_pred, 
        const std::shared_ptr<Tensor>& y_true) {

    auto diff = sub_op_.forward({y_pred, y_true});
    auto sq = square_op_.forward({diff});
    auto loss = sum_op_.forward({sq});
    return loss;
}