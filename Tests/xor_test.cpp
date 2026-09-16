#include "../Engine/Tokenizer/bpe_tokenizer.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <stdexcept>

const std::string DATA_PATH =
    "../Data/toy_train.txt";

const std::string TOKENIZER_PATH =
    "../Models/ToyTokenizer";

const size_t VOCAB_SIZE = 512;


// ============================================================
// Создание маленького датасета
// ============================================================

void CreateDataset() {

    std::filesystem::create_directories("../Data");

    std::ofstream file(DATA_PATH);

    if (!file.is_open()) {
        throw std::runtime_error(
            "Cannot create dataset: " + DATA_PATH
        );
    }

    /*
        Несколько разных формулировок одной и той же задачи.

        Это всё ещё очень простой датасет для Transformer,
        но BPE получает намного больше различных последовательностей.
    */

    for (size_t repeat = 0; repeat < 500; ++repeat) {

        for (int a = 0; a <= 9; ++a) {

            for (int b = 0; b <= 9; ++b) {

                int result = a + b;

                file
                    << "Вопрос: "
                    << a
                    << " + "
                    << b
                    << " = Ответ: "
                    << result
                    << "\n";

                file
                    << "Сколько будет "
                    << a
                    << " + "
                    << b
                    << "? Ответ: "
                    << result
                    << "\n";

                file
                    << "Посчитай: "
                    << a
                    << " + "
                    << b
                    << ". Ответ: "
                    << result
                    << "\n";

                file
                    << "Результат: "
                    << a
                    << " + "
                    << b
                    << " = "
                    << result
                    << "\n";

                file
                    << "Сложение: "
                    << a
                    << " плюс "
                    << b
                    << " равно "
                    << result
                    << "\n";
            }
        }
    }

    file.close();

    std::cout
        << "Dataset created: "
        << DATA_PATH
        << "\n";
}


// ============================================================
// Чтение всего файла
// ============================================================

std::string ReadFile(
    const std::string& path
) {
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "Cannot open file: " + path
        );
    }

    return std::string(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    );
}


// ============================================================
// Проверка токенайзера
// ============================================================

void TestTokenizer(
    BPETokenizer& tokenizer
) {
    const std::vector<std::string> tests = {

        "Вопрос: 2 + 3 = Ответ: 5",

        "Вопрос: 7 + 8 = Ответ: 15",

        "Вопрос: 9 + 9 = Ответ: 18",

        "Сколько будет 4 + 6? Ответ: 10",

        "Посчитай: 3 + 5. Ответ: 8",

        "Результат: 8 + 7 = 15",

        "Сложение: 2 плюс 9 равно 11"
    };

    std::cout << "\n";
    std::cout << "Tokenizer test\n";
    std::cout << "==============\n";

    for (const auto& text : tests) {

        std::vector<size_t> ids =
            tokenizer.Encode(text);

        std::string decoded =
            tokenizer.Decode(ids);

        std::cout
            << "\nText:\n"
            << text
            << "\n";

        std::cout
            << "Tokens: "
            << ids.size()
            << "\n";

        std::cout
            << "IDs: ";

        for (size_t id : ids) {
            std::cout << id << " ";
        }

        std::cout << "\n";

        std::cout
            << "Decoded:\n"
            << decoded
            << "\n";

        if (decoded == text) {
            std::cout << "OK\n";
        }
        else {
            std::cout
                << "ERROR: decoded text differs!\n";
        }
    }
}


// ============================================================
// Проверка сохранения / загрузки
// ============================================================

void TestSaveLoad(
    BPETokenizer& tokenizer
) {
    std::cout << "\n";
    std::cout << "Testing Save / Load\n";
    std::cout << "==================\n";

    tokenizer.Save(TOKENIZER_PATH);

    std::cout
        << "Tokenizer saved to: "
        << TOKENIZER_PATH
        << "\n";

    std::cout
        << "Vocabulary size: "
        << tokenizer.GetVocabSize()
        << "\n";

    BPETokenizer loaded;

    loaded.Load(TOKENIZER_PATH);

    std::cout
        << "Loaded vocabulary size: "
        << loaded.GetVocabSize()
        << "\n";

    if (
        tokenizer.GetVocabSize()
        !=
        loaded.GetVocabSize()
    ) {
        throw std::runtime_error(
            "Vocabulary size changed after Load"
        );
    }

    TestTokenizer(loaded);
}


// ============================================================
// main
// ============================================================

int main() {

    try {

        std::cout
            << "=== Toy Russian BPE training ===\n\n";


        // ----------------------------------------------------
        // 1. Создаём датасет
        // ----------------------------------------------------

        CreateDataset();


        // ----------------------------------------------------
        // 2. Загружаем датасет
        // ----------------------------------------------------

        std::string corpus =
            ReadFile(DATA_PATH);

        std::cout
            << "Corpus size: "
            << corpus.size()
            << " bytes\n";


        // ----------------------------------------------------
        // 3. Создаём новый токенайзер
        // ----------------------------------------------------

        BPETokenizer tokenizer;


        std::cout
            << "Initial vocabulary size: "
            << tokenizer.GetVocabSize()
            << "\n";


        // ----------------------------------------------------
        // 4. Обучаем BPE
        // ----------------------------------------------------

        std::cout
            << "\nTraining BPE...\n";

        tokenizer.Train(
            corpus,
            VOCAB_SIZE
        );


        // ----------------------------------------------------
        // 5. Проверяем размер словаря
        // ----------------------------------------------------

        std::cout
            << "\nFinal vocabulary size: "
            << tokenizer.GetVocabSize()
            << "\n";

        if (
            tokenizer.GetVocabSize()
            != VOCAB_SIZE
        ) {

            std::cout
                << "WARNING: tokenizer stopped before "
                << VOCAB_SIZE
                << " tokens.\n";
        }
        else {

            std::cout
                << "Vocabulary reached target size.\n";
        }


        // ----------------------------------------------------
        // 6. Проверяем Encode -> Decode
        // ----------------------------------------------------

        TestTokenizer(tokenizer);


        // ----------------------------------------------------
        // 7. Проверяем Save -> Load
        // ----------------------------------------------------

        TestSaveLoad(tokenizer);


        std::cout
            << "\n=== Done ===\n";
    }
    catch (const std::exception& e) {

        std::cerr
            << "\nERROR: "
            << e.what()
            << "\n";

        return 1;
    }

    return 0;
}