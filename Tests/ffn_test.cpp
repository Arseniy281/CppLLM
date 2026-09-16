#include "../Engine/Layers/feed_forward.h"
#include "../Engine/Tensor/tensor.h"
#include <iostream>

int main() {
    std::cout << "=== FeedForward Test ===\n\n";

    // 1. Создаём вход
    size_t batch = 2;
    size_t seq_len = 3;
    size_t embed_dim = 4;
    size_t hidden_dim = 8;

    auto x = std::make_shared<Tensor>(Tensor({batch, seq_len, embed_dim}));
    // заполняем x (например, случайными числами)

    // 2. Создаём FeedForward
    FeedForward ff(embed_dim, hidden_dim);

    // 3. Forward
    auto y = ff.forward(x);

    // 4. Проверка формы
    std::cout << "Output shape: ";
    for (auto d : y->GetShape()) std::cout << d << " ";
    std::cout << "\n";

    // 5. Backward
    Tensor grad_output({batch, seq_len, embed_dim}, 1.0f);

    y->backward(grad_output);

    auto grad_x = x->Grad();

    if (grad_x == nullptr) {
        std::cerr << "ERROR: grad_x is null\n";
        return 1;
    }

    // 6. Проверка градиентов
    std::cout << "grad_x shape: ";
    for (auto d : grad_x->GetShape()) std::cout << d << " ";
    std::cout << "\n";

    // 7. Update / ClearGrad
    ff.Update(0.01f);
    ff.ClearGrad();

    std::cout << "\n✅ FeedForward test completed!\n";
    return 0;
}