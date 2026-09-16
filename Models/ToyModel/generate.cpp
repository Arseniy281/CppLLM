#include "../../Engine/Layers/language_model.h"
#include "../../Engine/Tokenizer/bpe_tokenizer.h"
#include "config.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

const int MAX_NEW_TOKENS = 8;
const size_t TESTS_PER_TYPE = 5;


enum class ExampleType {
    Fact,
    Definition,
    Comparison,
    Logic,
    Unknown
};

struct Example {
    std::string prompt;
    std::string expected;
    ExampleType type;
};

bool IsDefinitionAnswer(const std::string& answer) {
    static const std::vector<std::string> definition_answers = {
        "A liquid.", "A star.", "An animal.", "A flower.", "A plant.", "Frozen water.",
        "Water from clouds.",  "A machine.", "A vehicle.", "A fruit.", "A vegetable.",
        "A natural satellite.", "A planet.", "Furniture.", "Food.", "A drink."
    };

    return std::find(definition_answers.begin(), definition_answers.end(),
        answer) != definition_answers.end();
}

ExampleType ClassifyExample(const std::string& prompt, const std::string& answer) {
    bool comparison_question = 
        prompt.find("Which number is larger") != std::string::npos ||
        prompt.find("Which number is smaller") != std::string::npos ||
        prompt.find("greater than") != std::string::npos ||
        prompt.find("smaller than") != std::string::npos;

    if (comparison_question) {
        if (answer == "Yes." || answer == "No.") {
            return ExampleType::Logic;
        }

        return ExampleType::Comparison;
    }

    if (prompt.find("equal to") != std::string::npos ||
        prompt.find("not equal to") != std::string::npos) {
        return ExampleType::Logic;
    }

    if (IsDefinitionAnswer(answer)) {
        return ExampleType::Definition;
    }

    return ExampleType::Fact;
}

std::vector<Example> LoadDataset(const std::string& path) {
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open dataset: " + path);
    }

    std::vector<Example> examples;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) { continue; }

        const std::string separator = " Answer: ";
        size_t answer_pos = line.find(separator);
        if (answer_pos == std::string::npos) { continue; }

        std::string prompt = line.substr(0, answer_pos + separator.size());
        std::string answer = line.substr(answer_pos + separator.size());
        const std::string end_marker = " <END>";

        if (answer.size() >= end_marker.size() && 
            answer.compare(answer.size() - end_marker.size(), 
            end_marker.size(), end_marker) == 0) {

            answer.erase(answer.size() - end_marker.size());
        }

        Example example;

        example.prompt = prompt;
        example.expected = answer;
        example.type = ClassifyExample(prompt, answer);

        examples.push_back(example);
    }

    return examples;
}

std::vector<Example> SelectTests(const std::vector<Example>& dataset) {
    std::mt19937 gen(SEED);
    std::vector<Example> result;

    const std::vector<ExampleType> types = {
        ExampleType::Fact,
        ExampleType::Definition,
        ExampleType::Comparison,
        ExampleType::Logic
    };

    for (ExampleType type : types) {
        std::vector<Example> candidates;
        for (const Example& example : dataset) {
            if (example.type == type) {
                candidates.push_back(example);
            }
        }

        std::shuffle(candidates.begin(), candidates.end(), gen);
        size_t count = std::min(TESTS_PER_TYPE, candidates.size());

        for (size_t i = 0; i < count; i++) {
            result.push_back(candidates[i]);
        }
    }

    std::shuffle(result.begin(), result.end(), gen);
    return result;
}

std::string Normalize(std::string text) {

    while (!text.empty() && (text.back() == ' ' ||
            text.back() == '\n' || text.back() == '\r')) {

        text.pop_back();
    }

    return text;
}

int main() {

    try {
        BPETokenizer tokenizer;

        tokenizer.Load(TOKENIZER_PATH);
        const int END_TOKEN_ID = (int)(tokenizer.GetTokenId("<END>"));

        std::vector<Example> dataset = LoadDataset(DATA_PATH);
        std::vector<Example> tests = SelectTests(dataset);

        LanguageModel model(VOCAB_SIZE, EMBED_DIM, BLOCKS,
            HEADS, HIDDEN, TRAINING_DEVICE);

        model.LoadModel(MODEL_PATH);
        model.SetUseKVCache(false);
        size_t correct = 0;

        for (const Example& example : tests) {
            std::vector<size_t> prompt_tokens = tokenizer.Encode(example.prompt);
            if (prompt_tokens.empty()) { continue; }

            std::vector<size_t> generated = model.generate(
                prompt_tokens, MAX_NEW_TOKENS, 0.1f, 1.0f, END_TOKEN_ID);

            std::string output = Normalize(tokenizer.Decode(generated));

            std::cout << "Prompt: " << example.prompt << "\n";
            std::cout << "Output: " << output << "\n\n";

            if (output == Normalize(example.expected)) {
                ++correct;
            }
        }

        std::cout << "========================================\n" << "Correct: "
            << correct << " / " << tests.size() << "\n"
            << "========================================\n";
    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}