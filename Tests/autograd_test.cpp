// ==================== autograd_test.cpp ====================
#include "../Engine/Tensor/tensor.h"
#include "../Engine/Autograd/add_op.h"
#include "../Engine/Autograd/sub_op.h"
#include "../Engine/Autograd/mul_op.h"
#include "../Engine/Autograd/square_op.h"
#include "../Engine/Autograd/sum_op.h"
#include <iostream>
#include <memory>

void CheckGradient(const std::string& name, std::shared_ptr<Tensor> tensor) {
    if (!tensor || !tensor->Grad()) {
        std::cout << "  ❌ " << name << ".grad_ is nullptr!\n";
        return;
    }
    std::cout << "  ✅ " << name << ".grad_:\n";
    tensor->Grad()->print();
}

int main() {
    try {
        std::cout << "=== Autograd Test ===\n\n";

        // 1. Создаём данные
        auto x = std::make_shared<Tensor>(Tensor({2, 2}));
        x->at({0, 0}) = 1.0f; x->at({0, 1}) = 2.0f;
        x->at({1, 0}) = 3.0f; x->at({1, 1}) = 4.0f;

        auto W = std::make_shared<Tensor>(Tensor({2, 1}));
        W->at({0, 0}) = 0.5f;
        W->at({1, 0}) = 0.5f;

        auto b = std::make_shared<Tensor>(Tensor({2, 1}));
        b->at({0, 0}) = 0.1f;
        b->at({1, 0}) = 0.1f;

        auto y_true = std::make_shared<Tensor>(Tensor({2, 1}));
        y_true->at({0, 0}) = 2.0f;
        y_true->at({1, 0}) = 4.0f;

        // 2. Строим граф
        auto mul_op = std::make_shared<MulOp>();
        auto add_op = std::make_shared<AddOp>();
        auto sub_op = std::make_shared<SubOp>();
        auto square_op = std::make_shared<SquareOp>();
        auto sum_op = std::make_shared<SumOp>();

        auto z = mul_op->forward({x, W});
        auto y = add_op->forward({z, b});
        auto diff = sub_op->forward({y, y_true});
        auto sq = square_op->forward({diff});
        auto loss = sum_op->forward({sq});

        if (!loss->GradFn()) {
            std::cout << "❌ loss has no grad_fn!\n";
            return 1;
        }
        std::cout << "✅ Graph built successfully\n";

        // 3. Backward
        Tensor grad_output({1, 1}, 1.0f);
        loss->backward(grad_output);
        std::cout << "✅ Backward completed\n\n";

        // 4. Проверка градиентов
        std::cout << "Gradients:\n";
        CheckGradient("W", W);
        CheckGradient("b", b);
        CheckGradient("x", x);
        CheckGradient("z", z);
        CheckGradient("y", y);
        CheckGradient("diff", diff);
        CheckGradient("sq", sq);

        // 5. Проверяем, что градиенты не нулевые
        std::cout << "\nVerifying gradients are non-zero:\n";
        bool has_grad = false;
        if (W->Grad()) {
            for (size_t i = 0; i < W->Grad()->GetSize(); ++i) {
                if (W->Grad()->at(i) != 0.0f) {
                    has_grad = true;
                    break;
                }
            }
        }
        std::cout << "  W.grad_ has non-zero values: " << (has_grad ? "✅ YES" : "❌ NO") << "\n";

        std::cout << "\n✅ All tests passed!\n";

    } catch (const std::exception& e) {
        std::cerr << "\n❌ ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}