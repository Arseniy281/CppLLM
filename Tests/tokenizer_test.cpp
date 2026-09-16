#include "../Engine/Tokenizer/bpe_tokenizer.h"
#include <iostream>

int main() {
    std::cout << "=== BPE Tokenizer Test ===\n\n";

    BPETokenizer tokenizer;

    std::string corpus = "hello world hello hello world";
    std::cout << "Training on corpus: \"" << corpus << "\"\n";
    tokenizer.Train(corpus, 20);

    tokenizer.Save("vocab.txt");
    std::cout << "Vocabulary saved to vocab.txt\n\n";

    std::string test_text = "hello";
    std::cout << "Encoding: \"" << test_text << "\"\n";
    std::vector<size_t> ids = tokenizer.Encode(test_text);

    std::cout << "Token IDs: ";
    for (int id : ids) {
        std::cout << id << " ";
    }
    std::cout << "\n";

    std::string decoded = tokenizer.Decode(ids);
    std::cout << "Decoded: \"" << decoded << "\"\n\n";

    std::string unknown = "xyz";
    std::cout << "Encoding unknown word: \"" << unknown << "\"\n";
    ids = tokenizer.Encode(unknown);
    std::cout << "Token IDs: ";
    for (int id : ids) {
        std::cout << id << " ";
    }
    std::cout << "\n";
    std::cout << "Decoded: \"" << tokenizer.Decode(ids) << "\"\n";

    std::cout << "\n✅ Test completed!\n";
    return 0;
}