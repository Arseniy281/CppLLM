#include "../Engine/Layers/language_model.h"
#include "../Engine/Tensor/tensor.h"
#include <iostream>
#include <filesystem>
#include <numeric>
#include <vector>
#include <iomanip>

// ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

bool CompareTensors(const Tensor& t1, const Tensor& t2, float eps = 1e-4) {
    if (t1.GetShape() != t2.GetShape()) {
        std::cout << "Shapes differ!\n";
        return false;
    }
    size_t mismatch_count = 0;
    for (size_t i = 0; i < t1.GetSize(); i++) {
        if (std::abs(t1.at(i) - t2.at(i)) > eps) {
            mismatch_count++;
            if (mismatch_count <= 5) {
                std::cout << "Mismatch at index " << i << ": "
                          << t1.at(i) << " vs " << t2.at(i) << "\n";
            }
        }
    }
    if (mismatch_count > 0) {
        std::cout << "Total mismatches: " << mismatch_count << "\n";
        return false;
    }
    return true;
}

float TensorSum(const Tensor& t) {
    const float* data = t.Data();
    return std::accumulate(data, data + t.GetSize(), 0.0f);
}

void PrintTensorFirst10(const std::string& name, const Tensor& t) {
    std::cout << name << " (first 10): ";
    const float* data = t.Data();
    for (size_t i = 0; i < std::min(t.GetSize(), size_t(10)); ++i) {
        std::cout << std::fixed << std::setprecision(4) << data[i] << " ";
    }
    std::cout << "\n";
}

bool CompareTensor(const Tensor& a, const Tensor& b, const std::string& name) {
    if (a.GetShape() != b.GetShape()) {
        std::cout << "  ❌ " << name << " shape mismatch!\n";
        std::cout << "     Model1: ";
        for (auto x : a.GetShape()) std::cout << x << " ";
        std::cout << "\n     Model2: ";
        for (auto x : b.GetShape()) std::cout << x << " ";
        std::cout << "\n";
        return false;
    }
    
    float sum_a = TensorSum(a);
    float sum_b = TensorSum(b);
    
    std::cout << "  " << name << " sum: " << std::fixed << std::setprecision(6) 
              << sum_a << " vs " << sum_b;
    
    if (sum_a == sum_b) {
        std::cout << " ✅ MATCH\n";
        return true;
    } else {
        std::cout << " ❌ DIFFER\n";
        return false;
    }
}

// ==================== ТЕСТ ====================

int main() {
    std::cout << "=== Save/Load Test ===\n\n";

    // 1. Параметры
    size_t vocab_size = 100;
    size_t embed_dim = 8;
    size_t num_blocks = 2;
    size_t num_heads = 2;
    size_t hidden_dim = 16;
    std::string folder = "test_model";

    // 2. Создаём модель и сохраняем
    std::cout << "Creating model1...\n";
    LanguageModel model1(vocab_size, embed_dim, num_blocks, num_heads, hidden_dim);
    
    std::cout << "Saving model to: " << folder << "\n";
    model1.SaveModel(folder);
    std::cout << "✅ Model saved\n\n";

    // 3. Создаём вторую модель и загружаем
    std::cout << "Creating model2...\n";
    LanguageModel model2(vocab_size, embed_dim, num_blocks, num_heads, hidden_dim);
    
    std::cout << "Loading model from: " << folder << "\n";
    model2.LoadModel(folder);
    std::cout << "✅ Model loaded\n\n";

    // После загрузки model2
    bool emb_match = CompareTensors(model1.GetEmbeddings(), model2.GetEmbeddings());
    std::cout << "Embedding match: " << (emb_match ? "✅" : "❌") << "\n";

    bool lm_w_match = CompareTensors(model1.GetLMHeadWeights(), model2.GetLMHeadWeights());
    std::cout << "LM Head W match: " << (lm_w_match ? "✅" : "❌") << "\n";

    // 4. СРАВНИВАЕМ ВЕСА
    std::cout << "=== Comparing weights ===\n\n";

    // 4a. Embedding
    std::cout << "1. Embedding layer:\n";
    PrintTensorFirst10("  Model1", model1.GetEmbeddings());
    PrintTensorFirst10("  Model2", model2.GetEmbeddings());
    CompareTensor(model1.GetEmbeddings(), model2.GetEmbeddings(), "  Embedding sum");
    std::cout << "\n";

    // 4b. LM Head
    std::cout << "2. LM Head:\n";
    PrintTensorFirst10("  Model1 W", model1.GetLMHeadWeights());
    PrintTensorFirst10("  Model2 W", model2.GetLMHeadWeights());
    CompareTensor(model1.GetLMHeadWeights(), model2.GetLMHeadWeights(), "  LM Head W sum");
    std::cout << "\n";

    PrintTensorFirst10("  Model1 b", model1.GetLMHeadBias());
    PrintTensorFirst10("  Model2 b", model2.GetLMHeadBias());
    CompareTensor(model1.GetLMHeadBias(), model2.GetLMHeadBias(), "  LM Head b sum");
    std::cout << "\n";

    // 5. ГЕНЕРАЦИЯ (для проверки)
    std::cout << "=== Generation test ===\n\n";
    
    size_t start_token = 0;
    int max_len = 5;
    float temperature = 1.0f;
    float top_p = 0.9f;
    int end_token_id = -1;

    std::vector<size_t> tokens1 = model1.generate({start_token}, max_len, temperature, top_p, end_token_id);
    std::vector<size_t> tokens2 = model2.generate({start_token}, max_len, temperature, top_p, end_token_id);

    std::cout << "Model1 tokens: ";
    for (int t : tokens1) std::cout << t << " ";
    std::cout << "\n";

    std::cout << "Model2 tokens: ";
    for (int t : tokens2) std::cout << t << " ";
    std::cout << "\n";

    if (tokens1 == tokens2) {
        std::cout << "✅ Models produce the same output!\n";
    } else {
        std::cout << "❌ Models produce different output (weights mismatch)\n";
    }

    std::cout << "\n=== Test completed ===\n";
    return 0;
}