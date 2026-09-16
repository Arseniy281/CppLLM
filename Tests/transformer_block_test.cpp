#include "../Engine/Transformer/transformer_block.h"
#include "../Engine/Tensor/tensor.h"

#include <iostream>
#include <cmath>
#include <memory>

void print_shape(const std::string& name, const Tensor& t) {
    std::cout << name << " shape: (";

    auto shape = t.GetShape();

    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];

        if (i + 1 < shape.size()) {
            std::cout << ", ";
        }
    }

    std::cout << ")\n";
}

bool same_shape(
    const std::vector<size_t>& a,
    const std::vector<size_t>& b
) {
    return a == b;
}

bool has_only_finite_values(const Tensor& t) {
    for (size_t i = 0; i < t.GetSize(); ++i) {
        if (!std::isfinite(t.Data()[i])) {
            return false;
        }
    }

    return true;
}

int main() {
    std::cout << "=== TransformerBlock Test ===\n\n";

    // ============================================================
    // Настройки
    // ============================================================

    const size_t BATCH = 2;
    const size_t SEQ_LEN = 4;
    const size_t EMBED_DIM = 8;
    const size_t NUM_HEADS = 2;
    const size_t HIDDEN_DIM = 32;

    // ============================================================
    // Input
    // ============================================================

    Tensor x(
        {BATCH, SEQ_LEN, EMBED_DIM},
        UninitializedTag{}
    );

    for (size_t i = 0; i < x.GetSize(); ++i) {
        x.Data()[i] = static_cast<float>(i) / 10.0f;
    }

    print_shape("x", x);

    std::cout
        << "x[0,0,0] = "
        << x.at({0, 0, 0})
        << "\n\n";

    // ============================================================
    // Create block
    // ============================================================

    TransformerBlock block(
        EMBED_DIM,
        NUM_HEADS,
        HIDDEN_DIM
    );

    std::cout << "TransformerBlock created.\n\n";

    // ============================================================
    // FORWARD
    // ============================================================

    auto x_ptr = std::make_shared<Tensor>(x);

    auto y = block.forward(x_ptr);

    if (!y) {
        std::cerr
            << "❌ TransformerBlock returned nullptr\n";

        return 1;
    }

    print_shape("y", *y);

    std::cout
        << "y[0,0,0] = "
        << y->at({0, 0, 0})
        << "\n\n";

    // ============================================================
    // Shape check
    // ============================================================

    if (same_shape(y->GetShape(), x.GetShape())) {
        std::cout
            << "✅ Shape preserved\n";
    } else {
        std::cerr
            << "❌ Shape mismatch!\n";

        return 1;
    }

    // ============================================================
    // Forward numerical check
    // ============================================================

    if (has_only_finite_values(*y)) {
        std::cout
            << "✅ Forward values are finite\n";
    } else {
        std::cerr
            << "❌ Forward contains NaN or Inf\n";

        return 1;
    }

    // ============================================================
    // BACKWARD
    // ============================================================
    //
    // TransformerBlock doesn't expose backward().
    //
    // Backward goes through Tensor's autograd graph:
    //
    // y
    // ↓
    // TransformerBlock
    // ↓
    // x_ptr
    //
    // ============================================================

    Tensor grad_output(
        {BATCH, SEQ_LEN, EMBED_DIM},
        1.0f
    );

    y->backward(grad_output);

    std::cout
        << "Backward completed.\n\n";

    // ============================================================
    // Input gradient
    // ============================================================

    auto grad_x = x_ptr->Grad();

    if (!grad_x) {
        std::cerr
            << "❌ Input gradient is nullptr\n";

        return 1;
    }

    print_shape("grad_x", *grad_x);

    std::cout
        << "grad_x[0,0,0] = "
        << grad_x->at({0, 0, 0})
        << "\n\n";

    // ============================================================
    // Gradient shape
    // ============================================================

    if (same_shape(
            grad_x->GetShape(),
            x.GetShape()
        )) {

        std::cout
            << "✅ grad_x shape matches input\n";

    } else {

        std::cerr
            << "❌ grad_x shape mismatch!\n";

        return 1;
    }

    // ============================================================
    // Gradient numerical check
    // ============================================================

    if (has_only_finite_values(*grad_x)) {
        std::cout
            << "✅ Input gradients are finite\n";
    } else {
        std::cerr
            << "❌ Input gradients contain NaN or Inf\n";

        return 1;
    }

    // ============================================================
    // UPDATE
    // ============================================================

    block.Update(0.01f);

    std::cout
        << "✅ Update passed\n";

    // ============================================================
    // CLEAR GRAD
    // ============================================================

    block.ClearGrad();

    std::cout
        << "✅ ClearGrad passed\n";

    // ============================================================
    // RESULT
    // ============================================================

    std::cout
        << "\n=== TransformerBlock Test Completed ===\n";

    std::cout
        << "✅ ALL TESTS PASSED\n";

    return 0;
}