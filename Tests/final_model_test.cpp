#include "../Engine/Tokenizer/bpe_tokenizer.h"

#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>

// ============================================================
// Настройки
// ============================================================

const size_t VOCAB_SIZE = 16000;

const std::string DATA_PATH =
    "../Data/russian_clean.txt";

const std::string TOKENIZER_PATH =
    "../Models/RussianTokenizer";

// ============================================================
// Загрузка текста
// ============================================================

std::string LoadText(const std::string& path) {
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "Cannot open corpus: " + path
        );
    }

    return std::string(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );
}

// ============================================================
// Main
// ============================================================

int main() {
    try {
        std::cout
            << "========================================\n"
            << "ENGLISH BPE TOKENIZER TRAINING\n"
            << "========================================\n\n";

        // ----------------------------------------------------
        // 1. Загружаем корпус
        // ----------------------------------------------------

        std::cout << "Loading corpus...\n";

        std::string text =
            LoadText(DATA_PATH);

        std::cout
            << "Characters: "
            << text.size()
            << "\n\n";

        if (text.empty()) {
            throw std::runtime_error(
                "Corpus is empty"
            );
        }

        // ----------------------------------------------------
        // 2. Создаём tokenizer
        // ----------------------------------------------------

        BPETokenizer tokenizer;

        std::cout
            << "Training BPE tokenizer...\n";

        std::cout
            << "Target vocabulary: "
            << VOCAB_SIZE
            << "\n\n";

        // ----------------------------------------------------
        // 3. Обучение
        // ----------------------------------------------------

        tokenizer.Train(
            text,
            VOCAB_SIZE
        );

        std::cout
            << "\nTokenizer training finished.\n";

        std::cout
            << "Vocabulary size: "
            << tokenizer.GetVocabSize()
            << "\n\n";

        // ----------------------------------------------------
        // 4. Сохраняем
        // ----------------------------------------------------

        std::cout
            << "Saving tokenizer...\n";

        tokenizer.Save(
            TOKENIZER_PATH
        );

        std::cout
            << "Tokenizer saved to:\n"
            << TOKENIZER_PATH
            << "\n\n";

        // ----------------------------------------------------
        // 5. Проверка Encode / Decode
        // ----------------------------------------------------

        std::string test_text =
            "The Four Horsemen are coming.";

        std::cout
            << "Testing Encode / Decode...\n\n";

        std::cout
            << "Original:\n"
            << test_text
            << "\n\n";

        std::vector<size_t> tokens =
            tokenizer.Encode(test_text);

        std::cout
            << "Tokens: "
            << tokens.size()
            << "\n";

        std::cout
            << "Token IDs:\n";

        for (size_t id : tokens) {
            std::cout
                << id
                << " ";
        }

        std::cout << "\n\n";

        std::string decoded =
            tokenizer.Decode(tokens);

        std::cout
            << "Decoded:\n"
            << decoded
            << "\n\n";

        // ----------------------------------------------------
        // 6. Проверка
        // ----------------------------------------------------

        if (decoded != test_text) {
            std::cout
                << "[WARNING] Encode/Decode changed the text.\n";
        } else {
            std::cout
                << "[OK] Encode/Decode test passed.\n";
        }

        std::cout
            << "\n========================================\n"
            << "TOKENIZER TEST FINISHED\n"
            << "========================================\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr
            << "\n[ERROR] "
            << e.what()
            << "\n";

        return 1;
    }
}