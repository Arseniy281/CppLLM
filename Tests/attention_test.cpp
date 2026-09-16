#include "../Engine/Layers/language_model.h"
#include "../Engine/Tokenizer/bpe_tokenizer.h"
#include "../Engine/Layers/ce_loss.h"

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <random>
#include <chrono>
#include <stdexcept>

#include <cuda_runtime.h>

// ============================================================
// Настройки
// ============================================================

const size_t VOCAB_SIZE = 16000;

const size_t EMBED_DIM = 128;
const size_t BLOCKS = 4;
const size_t HEADS = 4;
const size_t HIDDEN = 512;

const size_t CONTEXT = 128;
const size_t BATCH_SIZE = 16;

const size_t STEPS = 20000;

const float LR = 0.0003f;
const float WEIGHT_DECAY = 0.05f;

const float BETA1 = 0.9f;
const float BETA2 = 0.999f;
const float EPS = 1e-8f;

const size_t VALIDATION_EVERY = 100;
const size_t VALIDATION_BATCHES = 20;

const unsigned int SEED = 42;

// ============================================================
// Пути
// ============================================================

const std::string DATA_PATH =
    "../Data/russian_clean.txt";

const std::string TOKENIZER_PATH =
    "../Models/RussianTokenizer";

const std::string MODEL_PATH =
    "../Models/Russian_best";

// ============================================================
// Загрузка текста
// ============================================================

std::string LoadText(const std::string& path) {
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "Cannot open data file: " + path
        );
    }

    return std::string(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );
}

// ============================================================
// Создание batch
// ============================================================

void MakeBatch(
    const std::vector<size_t>& tokens,
    size_t begin,
    size_t end,
    std::mt19937& rng,
    std::vector<size_t>& input_ids,
    std::vector<size_t>& target_ids
) {
    if (end <= begin + CONTEXT) {
        throw std::runtime_error(
            "Not enough tokens to create batch"
        );
    }

    std::uniform_int_distribution<size_t> dist(
        begin,
        end - CONTEXT - 1
    );

    input_ids.clear();
    target_ids.clear();

    input_ids.reserve(BATCH_SIZE * CONTEXT);
    target_ids.reserve(BATCH_SIZE * CONTEXT);

    for (size_t b = 0; b < BATCH_SIZE; ++b) {
        size_t pos = dist(rng);

        for (size_t i = 0; i < CONTEXT; ++i) {
            input_ids.push_back(tokens[pos + i]);
            target_ids.push_back(tokens[pos + i + 1]);
        }
    }
}

// ============================================================
// Validation
// ============================================================

float EvaluateValidation(
    LanguageModel& model,
    const std::vector<size_t>& tokens,
    size_t validation_begin,
    size_t validation_end,
    std::mt19937& rng
) {
    if (validation_end <= validation_begin + CONTEXT) {
        throw std::runtime_error(
            "Validation set is too small"
        );
    }

    std::uniform_int_distribution<size_t> dist(
        validation_begin,
        validation_end - CONTEXT - 1
    );

    float total_loss = 0.0f;

    for (size_t batch = 0;
         batch < VALIDATION_BATCHES;
         ++batch) {

        std::vector<size_t> input_ids;
        std::vector<size_t> target_ids;

        input_ids.reserve(BATCH_SIZE * CONTEXT);
        target_ids.reserve(BATCH_SIZE * CONTEXT);

        for (size_t b = 0; b < BATCH_SIZE; ++b) {
            size_t pos = dist(rng);

            for (size_t i = 0; i < CONTEXT; ++i) {
                input_ids.push_back(tokens[pos + i]);
                target_ids.push_back(tokens[pos + i + 1]);
            }
        }

        std::vector<float> input_data(
            input_ids.begin(),
            input_ids.end()
        );

        std::vector<float> target_data(
            target_ids.begin(),
            target_ids.end()
        );

        auto input = std::make_shared<Tensor>(
            std::vector<size_t>{
                BATCH_SIZE,
                CONTEXT
            },
            input_data,
            Device::CUDA
        );

        auto targets = std::make_shared<Tensor>(
            std::vector<size_t>{
                BATCH_SIZE,
                CONTEXT
            },
            target_data,
            Device::CUDA
        );

        auto logits = model.forward(input);

        CrossEntropyLoss loss_fn;

        Tensor loss = loss_fn.forward(
            *logits,
            *targets
        );

        float loss_value = 0.0f;

        cudaError_t error = cudaMemcpy(
            &loss_value,
            loss.Data(),
            sizeof(float),
            cudaMemcpyDeviceToHost
        );

        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string(
                    "Validation loss cudaMemcpy failed: "
                ) + cudaGetErrorString(error)
            );
        }

        total_loss += loss_value;
    }

    return total_loss /
           static_cast<float>(VALIDATION_BATCHES);
}

// ============================================================
// Main
// ============================================================

int main() {
    try {
        std::cout
            << "========================================\n"
            << "CUDA Russian LM TRAINING TEST\n"
            << "========================================\n\n";

        // ====================================================
        // 1. Corpus
        // ====================================================

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

        // ====================================================
        // 2. Tokenizer
        // ====================================================

        std::cout << "Loading tokenizer...\n";

        BPETokenizer tokenizer;

        tokenizer.Load(TOKENIZER_PATH);

        std::cout
            << "Tokenizer vocabulary: "
            << tokenizer.GetVocabSize()
            << "\n\n";

        if (tokenizer.GetVocabSize() != VOCAB_SIZE) {
            throw std::runtime_error(
                "Tokenizer vocabulary size does not match VOCAB_SIZE"
            );
        }

        // ====================================================
        // 3. Tokenization
        // ====================================================

        std::cout << "Encoding corpus...\n";

        std::vector<size_t> tokens =
            tokenizer.Encode(text);

        std::cout
            << "Tokens: "
            << tokens.size()
            << "\n\n";

        if (tokens.size() <= CONTEXT + 1) {
            throw std::runtime_error(
                "Not enough tokens"
            );
        }

        // ====================================================
        // 4. Train / validation split
        // ====================================================

        const size_t validation_size =
            tokens.size() / 10;

        const size_t train_end =
            tokens.size() - validation_size;

        const size_t validation_begin =
            train_end;

        const size_t validation_end =
            tokens.size();

        std::cout
            << "Train tokens: "
            << train_end
            << "\n";

        std::cout
            << "Validation tokens: "
            << validation_end - validation_begin
            << "\n\n";

        // ====================================================
        // 5. Model
        // ====================================================

        std::cout
            << "Creating CUDA model...\n";

        LanguageModel model(
            VOCAB_SIZE,
            EMBED_DIM,
            BLOCKS,
            HEADS,
            HIDDEN,
            Device::CUDA
        );

        // KV cache при обучении не используется.
        model.SetUseKVCache(false);

        std::cout
            << "Model created.\n\n";

        std::cout
            << "Model configuration:\n"
            << "  Vocab:    " << VOCAB_SIZE << "\n"
            << "  Embed:    " << EMBED_DIM << "\n"
            << "  Blocks:   " << BLOCKS << "\n"
            << "  Heads:    " << HEADS << "\n"
            << "  Hidden:   " << HIDDEN << "\n"
            << "  Context:  " << CONTEXT << "\n"
            << "  Batch:    " << BATCH_SIZE << "\n"
            << "  Steps:    " << STEPS << "\n"
            << "  LR:       " << LR << "\n"
            << "  WD:       " << WEIGHT_DECAY << "\n\n";

        // ====================================================
        // 6. RNG
        // ====================================================

        std::mt19937 rng(SEED);
        std::mt19937 validation_rng(SEED);

        // ====================================================
        // 7. Initial validation
        // ====================================================

        std::cout
            << "Running initial validation...\n";

        float initial_val_loss =
            EvaluateValidation(
                model,
                tokens,
                validation_begin,
                validation_end,
                validation_rng
            );

        std::cout
            << "Initial validation loss: "
            << initial_val_loss
            << "\n\n";

        float best_val_loss =
            initial_val_loss;

        size_t best_step = 0;

        // ====================================================
        // 8. Training
        // ====================================================

        std::cout
            << "========================================\n"
            << "START TRAINING\n"
            << "========================================\n\n";

        float loss_sum = 0.0f;
        size_t loss_count = 0;

        std::vector<size_t> input_ids;
        std::vector<size_t> target_ids;

        auto training_start =
            std::chrono::high_resolution_clock::now();

        for (size_t step = 1;
             step <= STEPS;
             ++step) {

            auto step_start =
                std::chrono::high_resolution_clock::now();

            // ------------------------------------------------
            // Batch
            // ------------------------------------------------

            MakeBatch(
                tokens,
                0,
                train_end,
                rng,
                input_ids,
                target_ids
            );

            std::vector<float> input_data(
                input_ids.begin(),
                input_ids.end()
            );

            std::vector<float> target_data(
                target_ids.begin(),
                target_ids.end()
            );

            auto input = std::make_shared<Tensor>(
                std::vector<size_t>{
                    BATCH_SIZE,
                    CONTEXT
                },
                input_data,
                Device::CUDA
            );

            auto targets = std::make_shared<Tensor>(
                std::vector<size_t>{
                    BATCH_SIZE,
                    CONTEXT
                },
                target_data,
                Device::CUDA
            );

            // ------------------------------------------------
            // Forward
            // ------------------------------------------------

            auto logits =
                model.forward(input);

            CrossEntropyLoss loss_fn;

            Tensor loss =
                loss_fn.forward(
                    *logits,
                    *targets
                );

            float loss_value = 0.0f;

            cudaError_t error = cudaMemcpy(
                &loss_value,
                loss.Data(),
                sizeof(float),
                cudaMemcpyDeviceToHost
            );

            if (error != cudaSuccess) {
                throw std::runtime_error(
                    std::string(
                        "Training loss cudaMemcpy failed: "
                    ) + cudaGetErrorString(error)
                );
            }

            // ------------------------------------------------
            // Backward
            // ------------------------------------------------

            Tensor loss_grad =
                loss_fn.backward();

            logits->backward(
                loss_grad
            );

            // ------------------------------------------------
            // AdamW
            // ------------------------------------------------

            model.UpdateAdamW(
                LR,
                BETA1,
                BETA2,
                EPS,
                WEIGHT_DECAY
            );

            // Ждём завершения CUDA optimizer kernels.
            error = cudaDeviceSynchronize();

            if (error != cudaSuccess) {
                throw std::runtime_error(
                    std::string(
                        "CUDA error after AdamW: "
                    ) + cudaGetErrorString(error)
                );
            }

            // ------------------------------------------------
            // ОЧЕНЬ ВАЖНО:
            //
            // После optimizer step градиенты текущего
            // шага больше не нужны.
            //
            // На следующем backward Tensor::AddGrad()
            // должен начать новый gradient, а не
            // накапливать старый.
            // ------------------------------------------------

            model.ClearGrad();

            // ------------------------------------------------
            // Statistics
            // ------------------------------------------------

            loss_sum += loss_value;
            ++loss_count;

            auto step_end =
                std::chrono::high_resolution_clock::now();

            double step_ms =
                std::chrono::duration<double, std::milli>(
                    step_end - step_start
                ).count();

            // ------------------------------------------------
            // Logging + validation
            // ------------------------------------------------

            if (step == 1 ||
                step % VALIDATION_EVERY == 0 ||
                step == STEPS) {

                float avg_loss =
                    loss_sum /
                    static_cast<float>(loss_count);

                loss_sum = 0.0f;
                loss_count = 0;

                // Validation должна использовать
                // один и тот же набор случайных окон
                // на каждом шаге.
                validation_rng.seed(SEED);

                float val_loss =
                    EvaluateValidation(
                        model,
                        tokens,
                        validation_begin,
                        validation_end,
                        validation_rng
                    );

                std::cout
                    << "Step "
                    << step
                    << " | Loss: "
                    << loss_value
                    << " | Avg: "
                    << avg_loss
                    << " | Val: "
                    << val_loss
                    << " | Step: "
                    << step_ms
                    << " ms\n";

                // ------------------------------------------------
                // Best checkpoint
                // ------------------------------------------------

                if (val_loss < best_val_loss) {
                    best_val_loss = val_loss;
                    best_step = step;

                    std::cout
                        << "  New best validation loss: "
                        << best_val_loss
                        << "\n";

                    model.SaveModel(
                        MODEL_PATH
                    );
                }

                std::cout << "\n";
            }
        }

        // ====================================================
        // 9. Finish
        // ====================================================

        auto training_end =
            std::chrono::high_resolution_clock::now();

        double total_seconds =
            std::chrono::duration<double>(
                training_end - training_start
            ).count();

        std::cout
            << "========================================\n"
            << "TRAINING FINISHED\n"
            << "========================================\n\n";

        std::cout
            << "Initial validation loss: "
            << initial_val_loss
            << "\n";

        std::cout
            << "Best validation loss:    "
            << best_val_loss
            << "\n";

        std::cout
            << "Best step:               "
            << best_step
            << "\n";

        std::cout
            << "Training time:           "
            << total_seconds
            << " sec\n";

        std::cout
            << "Model path:              "
            << MODEL_PATH
            << "\n\n";

        if (best_step == 0) {
            std::cout
                << "[WARNING] Validation loss did not improve.\n";
        } else {
            std::cout
                << "[OK] Best model saved successfully.\n";
        }

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