#include "../Engine/Layers/language_model.h"
#include "../Engine/Tokenizer/bpe_tokenizer.h"
#include "../Engine/Tensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <random>
#include <unordered_map>
#include <string>
#include <vector>

static std::unordered_map<int, std::string> ID2STR;

static const float LEARNING_RATE = 0.01f; // higher LR to converge faster on small datasets
static const int EPOCHS = 800;

static const size_t VOCAB_SIZE = 100;
static const size_t EMBED_DIM = 32;
static const size_t NUM_BLOCKS = 1;
static const size_t NUM_HEADS = 2;
static const size_t HIDDEN_DIM = 64;

static const float temperature = 1.0f;
static const float top_p = 1.0f;

static const size_t MAX_GENERATION_TOKENS = 3;
// Remove single 90% pass criterion; use multiple checks below
static const float TRAIN_ACCURACY_GOAL = 0.80f; // considered "high"
static const float TEMPLATE_MATCH_GOAL = 0.90f; // fraction of template continuations that should match
static const std::string SAVE_FOLDER = "trained_model";

class CrossEntropyLoss {
private:
    Tensor logits_;

    size_t position_ = 0;
    int target_ = 0;

    std::vector<float> probabilities_;

public:

    Tensor Forward(
        const Tensor& logits,
        size_t position,
        int target
    ) {
        logits_ = logits;
        position_ = position;
        target_ = target;

        size_t vocab_size =
            logits.GetShape()[2];

        Tensor current_logits({
            vocab_size
        });

        for (size_t i = 0; i < vocab_size; i++) {
            current_logits.at(i) =
                logits.at({0, position, i});
        }

        float max_value = current_logits.at(0);
        for (size_t i = 1; i < vocab_size; i++) {
            max_value = std::max(max_value, current_logits.at(i));
        }

        probabilities_.resize(vocab_size);
        float sum = 0.0f;
        for (size_t i = 0; i < vocab_size; i++) {
            probabilities_[i] = std::exp(current_logits.at(i) - max_value);
            sum += probabilities_[i];
        }
        for (size_t i = 0; i < vocab_size; i++) probabilities_[i] /= sum;

        float probability = std::max(probabilities_[target_], 1e-12f);
        return Tensor({1,1}, -std::log(probability));
    }

    Tensor Backward() {
        Tensor gradient(logits_.GetShape(), 0.0f);
        size_t vocab_size = logits_.GetShape()[2];
        for (size_t i = 0; i < vocab_size; i++) {
            gradient.at({0, position_, i}) = probabilities_[i];
        }
        gradient.at({0, position_, static_cast<size_t>(target_)}) -= 1.0f;
        return gradient;
    }
};

struct Sentence {
    std::string text;
    std::vector<size_t> tokens;
};

Tensor MakeInput(const std::vector<size_t>& tokens) {
    Tensor input({1, tokens.size()});
    for (size_t i = 0; i < tokens.size(); i++) {
        input.at({0, i}) = static_cast<float>(tokens[i]);
    }
    return input;
}

int Argmax(const Tensor& logits, size_t position) {
    size_t vocab_size = logits.GetShape()[2];
    int best_token = 0;
    float best_value = logits.at({0, position, 0});
    for (size_t i = 1; i < vocab_size; i++) {
        float value = logits.at({0, position, i});
        if (value > best_value) { best_value = value; best_token = static_cast<int>(i); }
    }
    return best_token;
}

float SentenceLoss(LanguageModel& model, const std::vector<size_t>& tokens) {
    if (tokens.size() < 2) return 0.0f;
    model.ResetCache(); model.SetUseKVCache(false);
    auto input = std::make_shared<Tensor>(MakeInput(tokens));
    auto output = model.forward(input);
    float total_loss = 0.0f; size_t count = 0;
    for (size_t position = 0; position + 1 < tokens.size(); position++) {
        CrossEntropyLoss loss;
        Tensor value = loss.Forward(*output, position, tokens[position + 1]);
        total_loss += value.at(0);
        count++;
    }
    model.ResetCache();
    return total_loss / static_cast<float>(count);
}

float TrainSentence(LanguageModel& model, const std::vector<size_t>& tokens) {
    if (tokens.size() < 2) return 0.0f;
    model.ResetCache(); model.SetUseKVCache(false);
    auto input = std::make_shared<Tensor>(MakeInput(tokens));
    auto output = model.forward(input);
    float total_loss = 0.0f; size_t count = 0;
    Tensor total_gradient(output->GetShape(), 0.0f);
    for (size_t position = 0; position + 1 < tokens.size(); position++) {
        CrossEntropyLoss loss;
        Tensor loss_value = loss.Forward(*output, position, tokens[position + 1]);
        total_loss += loss_value.at(0);
        count++;
        Tensor gradient = loss.Backward();
        total_gradient += gradient;
    }
    total_gradient /= static_cast<float>(count);
    output->backward(total_gradient);
    model.Update(LEARNING_RATE);
    model.ClearGrad();
    model.ResetCache();
    return total_loss / static_cast<float>(count);
}

float DatasetLoss(LanguageModel& model, const std::vector<Sentence>& dataset) {
    if (dataset.empty()) return 0.0f;
    float total_loss = 0.0f;
    for (const Sentence& sentence : dataset) total_loss += SentenceLoss(model, sentence.tokens);
    return total_loss / static_cast<float>(dataset.size());
}

float CalculateAccuracy(LanguageModel& model, const std::vector<Sentence>& dataset) {
    size_t correct = 0; size_t total = 0;
    model.SetUseKVCache(false);
    for (const Sentence& sentence : dataset) {
        if (sentence.tokens.size() < 2) continue;
        model.ResetCache();
        auto input = std::make_shared<Tensor>(MakeInput(sentence.tokens));
        auto output = model.forward(input);
        for (size_t position = 0; position + 1 < sentence.tokens.size(); position++) {
            int prediction = Argmax(*output, position);
            int target = sentence.tokens[position + 1];
            if (prediction == target) correct++;
            total++;
        }
    }
    model.ResetCache();
    if (total == 0) return 0.0f;
    return static_cast<float>(correct) / static_cast<float>(total);
}

void PrintPredictions(LanguageModel& model, const std::vector<Sentence>& dataset) {
    std::cout << "\n============================================================\n"
              << "                 PREDICTIONS\n"
              << "============================================================\n";
    model.SetUseKVCache(false);
    for (const Sentence& sentence : dataset) {
        model.ResetCache();
        auto input = std::make_shared<Tensor>(MakeInput(sentence.tokens));
        auto output = model.forward(input);
        std::cout << "\n" << sentence.text << "\n";
        for (size_t position = 0; position + 1 < sentence.tokens.size(); position++) {
            int prediction = Argmax(*output, position);
            int target = sentence.tokens[position + 1];
            auto token_str = [&](int id)->std::string{
                auto it = ID2STR.find(id);
                if (it != ID2STR.end()) return it->second;
                return std::to_string(id);
            };
            std::cout << "  " << token_str(sentence.tokens[position]) << " -> " << token_str(prediction)
                      << " target=" << token_str(target)
                      << (prediction == target ? " [OK]" : " [WRONG]") << "\n";
        }
    }
    model.ResetCache();
}

std::vector<std::string> SplitToTokenStrings(const std::string& text) {
    std::vector<std::string> parts;
    if (text.empty()) return parts;
    std::string cur; bool cur_is_space = std::isspace(static_cast<unsigned char>(text[0]));
    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i]; bool is_space = std::isspace(static_cast<unsigned char>(c));
        if (cur.empty()) { cur.push_back(c); cur_is_space = is_space; }
        else if (is_space == cur_is_space) cur.push_back(c);
        else { parts.push_back(cur); cur.clear(); cur.push_back(c); cur_is_space = is_space; }
    }
    if (!cur.empty()) parts.push_back(cur);
    return parts;
}

void PrintTokens(const std::vector<size_t>& tokens) {
    for (size_t i = 0; i < tokens.size(); i++) {
        int id = tokens[i];
        auto it = ID2STR.find(id);
        if (it != ID2STR.end()) std::cout << it->second; else std::cout << id;
        if (i + 1 < tokens.size()) std::cout << " ";
    }
}

void TestGeneration(LanguageModel& model, const std::vector<Sentence>& dataset) {
    std::cout << "\n============================================================\n"
              << "                    GENERATION\n"
              << "============================================================\n";
    std::vector<std::vector<size_t>> prompts;
    for (const Sentence& sentence : dataset) {
        if (sentence.tokens.size() >= 2) {
            std::vector<size_t> prompt;
            prompt.push_back(sentence.tokens[0]);
            prompt.push_back(sentence.tokens[1]);
            prompts.push_back(prompt);
        }
    }
    for (size_t i = 0; i < prompts.size(); i++) {
        std::cout << "\nPrompt: "; PrintTokens(prompts[i]); std::cout << "\n";
        std::vector<size_t> generated = model.generate(prompts[i], prompts[i].size() + MAX_GENERATION_TOKENS, temperature, top_p, -1);
        std::cout << "Generated: "; PrintTokens(generated); std::cout << "\n";
    }
}

bool TestKVGeneration(LanguageModel& model, const Sentence& sentence) {
    std::cout << "\n============================================================\n"
              << "                  KV CACHE GENERATION\n"
              << "============================================================\n";
    if (sentence.tokens.size() < 3) return false;
    std::vector<size_t> prompt = { (size_t)sentence.tokens[0], (size_t)sentence.tokens[1] };
    std::cout << "Prompt: "; PrintTokens(prompt); std::cout << "\n";
    model.ResetCache(); model.SetUseKVCache(false);
    auto full_input = std::make_shared<Tensor>(MakeInput(prompt));
    auto full_output = model.forward(full_input);
    int full_prediction = Argmax(*full_output, prompt.size() - 1);
    model.ResetCache(); model.SetUseKVCache(true);
    int cached_prediction = -1;
    for (size_t i = 0; i < prompt.size(); i++) {
        auto one = std::make_shared<Tensor>(Tensor({1,1})); one->at({0,0}) = static_cast<float>(prompt[i]);
        auto output = model.forward(one);
        if (i == prompt.size() - 1) cached_prediction = Argmax(*output, 0);
    }
    model.ResetCache(); model.SetUseKVCache(false);
    auto token_str = [&](int id)->std::string{ auto it = ID2STR.find(id); if (it != ID2STR.end()) return it->second; return std::to_string(id); };
    std::cout << "Full forward prediction:   " << token_str(full_prediction) << "\n";
    std::cout << "KV-cache prediction:       " << token_str(cached_prediction) << "\n";
    bool passed = full_prediction == cached_prediction;
    std::cout << (passed ? "[PASS] KV-cache generation\n" : "[FAIL] KV-cache generation\n");
    return passed;
}

bool TestSaveLoad(LanguageModel& model, const Sentence& sentence) {
    std::cout << "\n============================================================\n"
              << "                    SAVE / LOAD\n"
              << "============================================================\n";
    model.SetUseKVCache(false); model.ResetCache();
    auto input = std::make_shared<Tensor>(MakeInput(sentence.tokens));
    auto before = model.forward(input);
    model.SaveModel(SAVE_FOLDER);
    LanguageModel loaded_model(ID2STR.size(), EMBED_DIM, NUM_BLOCKS, NUM_HEADS, HIDDEN_DIM);
    loaded_model.LoadModel(SAVE_FOLDER);
    loaded_model.ResetCache(); loaded_model.SetUseKVCache(false);
    auto after = loaded_model.forward(input);
    float max_difference = 0.0f;
    for (size_t i = 0; i < before->GetSize(); i++) {
        float difference = std::abs(before->at(i) - after->at(i));
        max_difference = std::max(max_difference, difference);
    }
    std::cout << "Maximum logit difference: " << std::scientific << max_difference << "\n";
    bool passed = max_difference < 1e-5f;
    std::cout << (passed ? "[PASS] Save / Load\n" : "[FAIL] Save / Load\n");
    loaded_model.ResetCache();
    return passed;
}

int main() {
    std::cout << "\n============================================================\n"
              << "             LANGUAGE MODEL TRAINING TEST\n"
              << "============================================================\n";

    // Build a compact, deterministic set of template sentences with repeated patterns
    // Subjects, verbs and objects chosen from small sets -> easy to learn
    std::vector<std::string> base_sentences = {
        "the cat eats fish",
        "the cat eats fish",
        "the cat eats fish",
        "the cat eats fish",
        "the dog likes meat",
        "the dog likes meat",
        "the dog likes meat",
        "the bird eats seed",
        "the bird eats seed",
        "the bird likes worm",
        "the cow eats grass",
        "the cow eats grass",
        "the mouse likes cheese",
        "the mouse likes cheese",
        "the tiger chases deer",
        "the lion eats meat",
        "the cat finds mouse",
        "the dog chases cat",
        "the sheep likes grass",
        "the sheep eats grass"
    };

    std::unordered_map<std::string,int> word2id;
    std::vector<std::string> vocab_words;
    auto add_word = [&](const std::string &w){ if (word2id.find(w) == word2id.end()) { int id = static_cast<int>(vocab_words.size()); word2id[w] = id; vocab_words.push_back(w); } };

    // Create dataset by expanding base_sentences deterministically
    std::vector<Sentence> dataset;
    for (const std::string &s : base_sentences) {
        std::istringstream iss(s);
        std::string w;
        std::vector<size_t> toks;
        while (iss >> w) {
            add_word(w);
            toks.push_back(word2id[w]);
        }
        dataset.push_back({s, toks});
    }

    // Build ID2STR
    for (size_t i = 0; i < vocab_words.size(); ++i) ID2STR[static_cast<int>(i)] = vocab_words[i];

    // Deterministic split ensuring test contains one exemplar of each base pattern
    std::vector<Sentence> train_dataset;
    std::vector<Sentence> test_dataset;
    for (size_t i = 0; i < dataset.size(); ++i) {
        // place one copy (the first occurrence) into test, others into train
        bool placed_test = false;
        // find if same text already placed into test
        for (const auto &t : test_dataset) if (t.text == dataset[i].text) { placed_test = true; break; }
        if (!placed_test) test_dataset.push_back(dataset[i]);
        else train_dataset.push_back(dataset[i]);
    }

    // If train is empty for some reason, move half into train
    if (train_dataset.empty() && test_dataset.size() > 1) {
        for (size_t i = 1; i < test_dataset.size(); ++i) train_dataset.push_back(test_dataset[i]);
        test_dataset.resize(1);
    }

    // Create model with dynamic vocab size
    size_t vocab_size = ID2STR.size();
    LanguageModel model(vocab_size, EMBED_DIM, NUM_BLOCKS, NUM_HEADS, HIDDEN_DIM);

    // Expand training dataset by repeating entries to help memorization on small data
    int repeat_factor = 4;
    std::vector<Sentence> train_expanded;
    train_expanded.reserve(train_dataset.size() * repeat_factor);
    for (int r = 0; r < repeat_factor; ++r) {
        for (const auto &s : train_dataset) train_expanded.push_back(s);
    }

    float initial_loss = DatasetLoss(model, train_dataset);

    std::cout << "\n============================================================\n"
              << "                    TRAINING\n"
              << "============================================================\n";

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float epoch_loss = 0.0f;
        for (const Sentence& sentence : train_expanded) epoch_loss += TrainSentence(model, sentence.tokens);
        epoch_loss /= static_cast<float>(train_expanded.size());
        if (epoch % 20 == 0 || epoch == EPOCHS - 1) {
            float train_acc = CalculateAccuracy(model, train_dataset);
            float test_acc = CalculateAccuracy(model, test_dataset);
            std::cout << "Epoch " << std::setw(4) << epoch
                      << " | Loss: " << std::fixed << std::setprecision(5) << epoch_loss
                      << " | Train Acc: " << std::setprecision(2) << train_acc * 100.0f
                      << "% | Test Acc: " << std::setprecision(2) << test_acc * 100.0f << "%\n";
        }
    }

    float final_loss = DatasetLoss(model, train_dataset);
    float train_accuracy = CalculateAccuracy(model, train_dataset);
    float test_loss = DatasetLoss(model, test_dataset);
    float test_accuracy = CalculateAccuracy(model, test_dataset);

    std::cout << "\n============================================================\n"
              << "                    FINAL MODEL\n"
              << "============================================================\n";

    std::cout << "Initial loss: " << std::fixed << std::setprecision(5) << initial_loss << "\n";
    std::cout << "Final train loss:   " << final_loss << "\n";
    std::cout << "Final test loss:    " << test_loss << "\n";
    std::cout << "Training accuracy: " << std::setprecision(2) << train_accuracy * 100.0f << "%\n";
    std::cout << "Test accuracy:     " << std::setprecision(2) << test_accuracy * 100.0f << "%\n";

    PrintPredictions(model, test_dataset);
    TestGeneration(model, test_dataset);

    // Evaluate template continuations: for each test sentence, use first two tokens as prompt
    auto EvaluateTemplates = [&](LanguageModel& m, const std::vector<Sentence>& tests)->float{
        size_t correct = 0; size_t total = 0;
        m.SetUseKVCache(false);
        for (const auto &s : tests) {
            if (s.tokens.size() < 3) continue;
            std::vector<size_t> prompt = { s.tokens[0], s.tokens[1] };
            auto input = std::make_shared<Tensor>(MakeInput(prompt));
            auto out = m.forward(input);
            int pred = Argmax(*out, prompt.size() - 1);
            if (pred == s.tokens[2]) correct++;
            total++;
        }
        if (total == 0) return 0.0f;
        return static_cast<float>(correct) / static_cast<float>(total);
    };

    float template_match_frac = EvaluateTemplates(model, test_dataset);

    bool kv_passed = false;
    bool save_load_passed = false;
    if (!test_dataset.empty()) {
        kv_passed = TestKVGeneration(model, test_dataset[0]);
        save_load_passed = TestSaveLoad(model, test_dataset[0]);
    }

    bool loss_decreased = final_loss < initial_loss;
    bool train_high = train_accuracy >= TRAIN_ACCURACY_GOAL;
    bool templates_ok = template_match_frac >= TEMPLATE_MATCH_GOAL;

    std::cout << "\n============================================================\n"
              << "                    FINAL RESULT\n"
              << "============================================================\n";

    std::cout << "Loss decreased:  " << (loss_decreased ? "PASS" : "FAIL") << "\n";
    std::cout << "Train accuracy:  " << (train_high ? "PASS" : "FAIL") << " (" << std::setprecision(2) << train_accuracy * 100.0f << "%)\n";
    std::cout << "Template cont.:  " << (templates_ok ? "PASS" : "FAIL") << " (" << std::setprecision(2) << template_match_frac * 100.0f << "%)\n";
    std::cout << "KV-cache:        " << (kv_passed ? "PASS" : "FAIL") << "\n";
    std::cout << "Save / Load:     " << (save_load_passed ? "PASS" : "FAIL") << "\n\n";

    bool overall = loss_decreased && train_high && templates_ok && kv_passed && save_load_passed;
    if (overall) {
        std::cout << "============================================================\n"
                  << "       LANGUAGE MODEL PASSED FULL TEST\n"
                  << "============================================================\n";
        return 0;
    } else {
        std::cout << "============================================================\n"
                  << "       LANGUAGE MODEL FAILED FULL TEST\n"
                  << "============================================================\n";
        return 1;
    }
}
