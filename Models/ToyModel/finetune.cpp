#include "../../Engine/Layers/language_model.h"
#include "../../Engine/Tokenizer/bpe_tokenizer.h"
#include "../../Engine/Layers/ce_loss.h"
#include "config.h"

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <random>
#include <chrono>
#include <stdexcept>
#include <algorithm>

#if ENABLE_CUDA
#include <cuda_runtime.h>
#endif

std::string LoadText(const std::string& path) {
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open data file: " + path);
    }

    return std::string((std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );
}

void MakeBatch(const std::vector<size_t>& tokens, size_t begin,
    size_t end, std::mt19937& rng, std::vector<size_t>& input_ids,
    std::vector<size_t>& target_ids) {

    if (end <= begin + CONTEXT) {
        throw std::runtime_error("Not enough tokens to create batch");
    }

    std::uniform_int_distribution<size_t> dist(begin, end - CONTEXT - 1);

    input_ids.clear();
    target_ids.clear();

    input_ids.reserve(BATCH_SIZE * CONTEXT);
    target_ids.reserve(BATCH_SIZE * CONTEXT);

    for (size_t b = 0; b < BATCH_SIZE; b++) {
        size_t pos = dist(rng);

        for (size_t i = 0; i < CONTEXT; i++) {
            input_ids.push_back(tokens[pos + i]);
            target_ids.push_back(tokens[pos + i + 1]);
        }
    }
}

float GetLossValue(const Tensor& loss) {
    float loss_value = 0.0f;

#if ENABLE_CUDA

    cudaError_t cuda_error = cudaMemcpy(
        &loss_value,
        loss.Data(),
        sizeof(float),
        cudaMemcpyDeviceToHost
    );

    if (cuda_error != cudaSuccess) {
        throw std::runtime_error(
            std::string("Loss cudaMemcpy failed: ") +
            cudaGetErrorString(cuda_error)
        );
    }

#else

    loss_value = loss.Data()[0];

#endif

    return loss_value;
}

float EvaluateValidation(LanguageModel& model,
        const std::vector<size_t>& tokens, size_t validation_begin,
        size_t validation_end, std::mt19937& rng) {

    if (validation_end <= validation_begin + CONTEXT) {
        throw std::runtime_error("Validation set is too small");
    }

    std::uniform_int_distribution<size_t> dist(
        validation_begin, validation_end - CONTEXT - 1
    );

    float total_loss = 0.0f;
    CrossEntropyLoss loss_fn;

    for (size_t batch = 0; batch < VALIDATION_BATCHES; batch++) {
        std::vector<size_t> input_ids;
        std::vector<size_t> target_ids;

        input_ids.reserve(BATCH_SIZE * CONTEXT);
        target_ids.reserve(BATCH_SIZE * CONTEXT);

        for (size_t b = 0; b < BATCH_SIZE; b++) {
            size_t pos = dist(rng);

            for (size_t i = 0; i < CONTEXT; i++) {
                input_ids.push_back(tokens[pos + i]);
                target_ids.push_back(tokens[pos + i + 1]);
            }
        }

        std::vector<float> input_data(input_ids.begin(), input_ids.end());
        std::vector<float> target_data(target_ids.begin(), target_ids.end());

        auto input = std::make_shared<Tensor>(
            std::vector<size_t>{BATCH_SIZE, CONTEXT},
            input_data, TRAINING_DEVICE);

        auto targets = std::make_shared<Tensor>(
            std::vector<size_t>{BATCH_SIZE, CONTEXT},
            target_data, TRAINING_DEVICE);

        auto logits = model.forward(input);
        Tensor loss = loss_fn.forward(*logits, *targets);
        total_loss += GetLossValue(loss);
    }

    return total_loss / (float)VALIDATION_BATCHES;
}

int main() {
    try {
        std::cout << "========================================\n"
            << "TOY TRANSFORMER - FINETUNE\n"
            << "========================================\n\n";

        std::string text = LoadText(DATA_PATH);

        if (text.empty()) {
            throw std::runtime_error("Dataset is empty");
        }

        BPETokenizer tokenizer;
        tokenizer.Load(TOKENIZER_PATH);

        if (tokenizer.GetVocabSize() != VOCAB_SIZE) {
            throw std::runtime_error(
                "Tokenizer vocabulary size does not "
                "match VOCAB_SIZE"
            );
        }

        std::vector<size_t> tokens = tokenizer.Encode(text);

        if (tokens.size() <= CONTEXT + 1) {
            throw std::runtime_error("Not enough tokens for training");
        }

        const size_t validation_size = tokens.size() / 10;
        const size_t train_end = tokens.size() - validation_size;
        const size_t validation_begin = train_end;
        const size_t validation_end = tokens.size();

        std::cout
            << "Dataset\n"
            << "  Characters: " << text.size() << "\n"
            << "  Tokens:     " << tokens.size() << "\n"
            << "  Train:      " << train_end << "\n"
            << "  Validation: "
            << validation_end - validation_begin << "\n\n";

        LanguageModel model(VOCAB_SIZE, EMBED_DIM, BLOCKS,
            HEADS, HIDDEN, TRAINING_DEVICE);

        std::cout << "Loading model...\n"
            << "  Path: " << MODEL_PATH << "\n";

        model.LoadModel(MODEL_PATH);
        model.SetUseKVCache(false);

        std::cout << "\nModel\n"
            << "  Device:     " << DeviceName(TRAINING_DEVICE) << "\n"
            << "  Vocabulary: " << VOCAB_SIZE << "\n"
            << "  Embedding:  " << EMBED_DIM << "\n"
            << "  Blocks:     " << BLOCKS << "\n"
            << "  Heads:      " << HEADS << "\n"
            << "  Hidden:     " << HIDDEN << "\n"
            << "  Context:    " << CONTEXT << "\n"
            << "  Batch:      " << BATCH_SIZE << "\n"
            << "  Steps:      " << STEPS << "\n"
            << "  Learning rate: " << LEARNING_RATE << "\n\n";

        std::mt19937 rng(SEED);
        std::mt19937 validation_rng(SEED);

        float initial_val_loss = EvaluateValidation(model, tokens,
                validation_begin, validation_end, validation_rng);

        float best_val_loss = initial_val_loss;
        size_t best_step = 0;

        std::cout << "Validation loss before finetuning: "
            << initial_val_loss << "\n\n";

        std::cout << "Finetuning...\n";

        float loss_sum = 0.0f;
        size_t loss_count = 0;

        std::vector<size_t> input_ids;
        std::vector<size_t> target_ids;

        auto training_start = std::chrono::high_resolution_clock::now();

        for (size_t step = 1; step <= STEPS; step++) {
            auto step_start = std::chrono::high_resolution_clock::now();

            MakeBatch(tokens, 0, train_end, rng, input_ids, target_ids);

            std::vector<float> input_data(input_ids.begin(), input_ids.end());
            std::vector<float> target_data(target_ids.begin(), target_ids.end());

            auto input = std::make_shared<Tensor>(
                std::vector<size_t>{BATCH_SIZE, CONTEXT},
                input_data, TRAINING_DEVICE);

            auto targets = std::make_shared<Tensor>(
                std::vector<size_t>{BATCH_SIZE, CONTEXT},
                target_data, TRAINING_DEVICE);

            auto logits = model.forward(input);

            CrossEntropyLoss loss_fn;
            Tensor loss = loss_fn.forward(*logits, *targets);

            float loss_value = GetLossValue(loss);
            Tensor loss_grad = loss_fn.backward();
            logits->backward(loss_grad);

            model.UpdateAdamW(LEARNING_RATE, BETA1,
                BETA2, EPS, WEIGHT_DECAY);

#if ENABLE_CUDA
            cudaError_t cuda_error = cudaDeviceSynchronize();

            if (cuda_error != cudaSuccess) {
                throw std::runtime_error(std::string(
                    "CUDA error after AdamW: ") 
                    + cudaGetErrorString(cuda_error)
                );
            }
#endif

            model.ClearGrad();

            loss_sum += loss_value;
            ++loss_count;

            auto step_end =std::chrono::high_resolution_clock::now();

            double step_ms = std::chrono::duration<double, std::milli>(
                step_end - step_start).count();

            if (step == 1 || step % VALIDATION_EVERY == 0 || step == STEPS) {
                float avg_loss = loss_sum / (float)loss_count;

                loss_sum = 0.0f;
                loss_count = 0;

                validation_rng.seed(SEED);

                float val_loss = EvaluateValidation(model, tokens, 
                    validation_begin, validation_end, validation_rng);

                std::cout << "Step " << step << " | Loss: " << loss_value
                    << " | Avg: " << avg_loss << " | Val: " << val_loss
                    << " | " << step_ms << " ms";

                if (val_loss < best_val_loss) {
                    best_val_loss = val_loss;
                    best_step = step;

                    model.SaveModel(MODEL_PATH);
                    std::cout << " | saved";
                }

                std::cout << "\n";
            }
        }

        auto training_end = std::chrono::high_resolution_clock::now();

        double total_seconds = std::chrono::duration<double>(
            training_end - training_start).count();

        model.SaveModel(MODEL_PATH);

        std::cout
            << "\n========================================\n"
            << "FINETUNING COMPLETE\n"
            << "========================================\n"
            << "Initial validation loss: " << initial_val_loss << "\n"
            << "Best validation loss:    " << best_val_loss << "\n"
            << "Best step:               " << best_step << "\n"
            << "Training time:           " << total_seconds << " sec\n"
            << "Model:                   " << MODEL_PATH << "\n"
            << "\nModel saved successfully.\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << "\n";
        return 1;
    }
}