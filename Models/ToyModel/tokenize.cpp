#include "../../Engine/Tokenizer/bpe_tokenizer.h"
#include "config.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

std::string LoadText(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open data file: " + path);
    }

    return std::string(
        (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

int main() {

    try {

        std::cout
            << "========================================\n"
            << "ENGLISH BPE TOKENIZER\n"
            << "========================================\n\n";
            
        std::string text = LoadText(DATA_PATH);

        if (text.empty()) {
            throw std::runtime_error("Dataset is empty");
        }

        BPETokenizer tokenizer;
        tokenizer.Train(text, VOCAB_SIZE);

        if (tokenizer.GetVocabSize() != VOCAB_SIZE) {
            throw std::runtime_error("Tokenizer has unexpected vocabulary size");
        }

        const size_t end_token_id = tokenizer.GetTokenId("<END>");

        if (end_token_id != tokenizer.GetVocabSize() - 1) {
            throw std::runtime_error( "<END> is not the last vocabulary token");
        }

        std::vector<size_t> end_tokens = tokenizer.Encode("<END>");

        if (end_tokens.size() != 1 || end_tokens[0] != end_token_id) {
            throw std::runtime_error("<END> was not encoded as the reserved token");
        }

        if (tokenizer.Decode(end_tokens) != "<END>") {
            throw std::runtime_error("<END> was not decoded correctly");
        }

        tokenizer.Save(TOKENIZER_PATH);

        std::cout << "Dataset characters: " << text.size() << "\n"
            << "Vocabulary size:    " << tokenizer.GetVocabSize() << "\n"
            << "<END> token id:     " << end_token_id << "\n"
            << "Tokenizer saved to: " << TOKENIZER_PATH << "\n\n";

        std::cout << "Tokenizer ready.\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n";
        return 1;
    }
}