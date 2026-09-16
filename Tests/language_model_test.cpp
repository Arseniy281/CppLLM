#include "../Engine/Layers/language_model.h"
#include <iostream>

int main() {
    // 1. Создаём модель
    LanguageModel model(100, 8, 2, 2, 16);  // vocab_size=100, embed_dim=8, 2 блока, 2 головы, hidden=16

    // 2. Начинаем с токена 0
    std::vector<size_t> tokens = model.generate({0}, 10, 1.0f, 0.9f, -1);

    std::cout << "Generated tokens: ";
    for (int t : tokens) {
        std::cout << t << " ";
    }
    std::cout << "\n";

    return 0;
}