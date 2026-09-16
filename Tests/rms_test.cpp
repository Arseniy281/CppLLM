#include "../Engine/Normalization/rms_norm.h"
#include "../Engine/Tensor/tensor.h"
#include <iostream>
#include <cmath>

void print_tensor(const std::string& name, const Tensor& t) {
    std::cout << name << ":\n";
    t.print();
    std::cout << "\n";
}

int main() {
    std::cout << "=== RMSNorm Test ===\n\n";

    // 1. Создаём входной тензор
    size_t batch = 2;
    size_t seq_len = 3;
    size_t embed_dim = 4;

    auto x = std::make_shared<Tensor>(Tensor({batch, seq_len, embed_dim}));
    for (size_t i = 0; i < x->GetSize(); ++i) {
        x->Data()[i] = static_cast<float>(i) + 0.5f;
    }
    print_tensor("x", *x);

    // 2. Создаём RMSNorm
    RMSNorm rmsnorm(embed_dim);
    print_tensor("gamma (initial)", rmsnorm.GetGamma());

    // 3. Forward
    auto y = rmsnorm.forward(x);
    print_tensor("y (after RMSNorm)", *y);

    // 4. Проверка RMS
    std::cout << "Checking RMS for each position (should be ~1.0):\n";
    for (size_t b = 0; b < batch; ++b) {
        for (size_t pos = 0; pos < seq_len; ++pos) {
            float rms = 0.0f;
            for (size_t d = 0; d < embed_dim; ++d) {
                float val = y->at({b, pos, d});
                rms += val * val;
            }
            rms /= embed_dim;
            rms = std::sqrt(rms);
            std::cout << "  batch=" << b << ", pos=" << pos << " RMS = " << rms << "\n";
        }
    }

    // 5. Backward (градиент)
    Tensor grad_output({batch, seq_len, embed_dim});
    for (size_t i = 0; i < grad_output.GetSize(); ++i) {
        grad_output.Data()[i] = 1.0f;
    }
    print_tensor("grad_output (all ones)", grad_output);

    y->backward(grad_output);

    auto grad_x = x->Grad();

    if (grad_x == nullptr) {
        std::cerr << "ERROR: grad_x is null!\n";
        return 1;
    }

    print_tensor("grad_x (backward)", *grad_x);

    std::cout << "✅ RMSNorm test completed!\n";
    return 0;
}