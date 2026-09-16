#include "../Engine/Transformer/transformer.h"
#include "../Engine/Tensor/tensor.h"

#include <iostream>
#include <memory>
#include <cmath>

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

int main() {
    std::cout << "=== Transformer Test ===\n\n";

    // ============================================================
    // Настройки
    // ============================================================

    const size_t NUM_BLOCKS = 2;
    const size_t BATCH = 2;
    const size_t SEQ_LEN = 4;
    const size_t EMBED_DIM = 8;
    const size_t NUM_HEADS = 2;
    const size_t HIDDEN_DIM = 32;

    // ============================================================
    // Создаём вход
    // ============================================================

    Tensor x(
        {BATCH, SEQ_LEN, EMBED_DIM},
        UninitializedTag{}
    );

    for (size_t i = 0; i < x.GetSize(); ++i) {
        x.Data()[i] = static_cast<float>(i) / 10.0f;
    }

    print_shape("x", x);

    // ============================================================
    // Создаём Transformer
    // ============================================================

    Transformer model(
        NUM_BLOCKS,
        EMBED_DIM,
        NUM_HEADS,
        HIDDEN_DIM
    );

    std::cout << "Model created.\n\n";

    // ============================================================
    // FORWARD
    // ============================================================

    auto x_ptr = std::make_shared<Tensor>(x);

    auto y = model.forward(x_ptr);

    if (!y) {
        std::cerr << "❌ Transformer returned nullptr\n";
        return 1;
    }

    print_shape("y", *y);

    // Transformer должен сохранять shape
    if (same_shape(y->GetShape(), x.GetShape())) {
        std::cout
            << "✅ Shape preserved through "
            << NUM_BLOCKS
            << " blocks\n";
    } else {
        std::cerr
            << "❌ Shape mismatch!\n";
        return 1;
    }

    // ============================================================
    // Проверяем, что forward не создал NaN / Inf
    // ============================================================

    bool valid_values = true;

    for (size_t i = 0; i < y->GetSize(); ++i) {
        float value = y->Data()[i];

        if (!std::isfinite(value)) {
            valid_values = false;
            break;
        }
    }

    if (valid_values) {
        std::cout << "✅ Forward values are finite\n";
    } else {
        std::cerr << "❌ Forward contains NaN or Inf\n";
        return 1;
    }

    // ============================================================
    // BACKWARD
    // ============================================================
    //
    // Transformer сам backward не предоставляет.
    //
    // backward идёт через autograd-граф:
    //
    // y
    // ↓
    // Transformer blocks
    // ↓
    // x
    //
    // Передаём градиент dL/dy.
    //
    // ============================================================

    Tensor grad_output(
        {BATCH, SEQ_LEN, EMBED_DIM},
        1.0f
    );

    y->backward(grad_output);

    std::cout << "\nBackward completed.\n";

    // ============================================================
    // Проверяем gradient входа
    // ============================================================

    auto grad_x = x_ptr->Grad();

    if (!grad_x) {
        std::cerr
            << "❌ Input gradient is nullptr\n";
        return 1;
    }

    print_shape("grad_x", *grad_x);

    if (same_shape(
            grad_x->GetShape(),
            x.GetShape()
        )) {

        std::cout
            << "✅ grad_x shape matches input\n";

    } else {

        std::cerr
            << "❌ grad_x shape mismatch\n";

        return 1;
    }

    // ============================================================
    // Проверяем gradient на NaN / Inf
    // ============================================================

    bool valid_grad = true;

    for (size_t i = 0; i < grad_x->GetSize(); ++i) {
        float value = grad_x->Data()[i];

        if (!std::isfinite(value)) {
            valid_grad = false;
            break;
        }
    }

    if (valid_grad) {
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

    model.Update(0.01f);

    std::cout
        << "✅ Update passed\n";

    // ============================================================
    // CLEAR GRAD
    // ============================================================

    model.ClearGrad();

    std::cout
        << "✅ ClearGrad passed\n";

    // ============================================================
    // RESULT
    // ============================================================

    std::cout
        << "\n=== Transformer Test Completed ===\n";

    std::cout
        << "✅ ALL TESTS PASSED\n";

    return 0;
}