#include "ce_loss.h"
#include "../Tensor/backend.h"
#include <stdexcept>

Tensor CrossEntropyLoss::forward(const Tensor& logits, const Tensor& targets) {
    if (logits.GetRank() != 3) {
        throw std::runtime_error(
            "CrossEntropyLoss: logits must be 3D "
            "[batch, seq_len, vocab_size]"
        );
    }

    if (targets.GetRank() != 2) {
        throw std::runtime_error(
            "CrossEntropyLoss: targets must be 2D "
            "[batch, seq_len]"
        );
    }

    const auto& logits_shape = logits.GetShape();
    const auto& targets_shape = targets.GetShape();

    if (logits_shape[0] != targets_shape[0] 
        || logits_shape[1] != targets_shape[1]) {

        throw std::runtime_error(
            "CrossEntropyLoss: logits and targets shapes must match");
    }

    if (logits.GetDevice() != targets.GetDevice()) {
        throw std::runtime_error(
            "CrossEntropyLoss: logits and targets must be on the same device");
    }

    logits_ = logits;
    targets_ = targets;

    batch_ = logits_shape[0];
    seq_len_ = logits_shape[1];
    vocab_size_ = logits_shape[2];

    return GetBackend(logits.GetDevice()).CrossEntropy(logits_, targets_);
}

Tensor CrossEntropyLoss::backward() {
    if (logits_.GetSize() == 0) {
        throw std::runtime_error(
            "CrossEntropyLoss::backward: forward must be called first");
    }
    return GetBackend(
        logits_.GetDevice()).CrossEntropyBackward(logits_, targets_);
}