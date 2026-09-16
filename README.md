# CppLLM

CppLLM is an experimental implementation of a Large Language Model written from scratch in C++.

The project implements the core components needed to build, train, fine-tune, and run a small Transformer model without using high-level deep learning frameworks.

🌐 English version · 🇷🇺 [Русская версия](README.ru.md)

---

## Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Build](#-build)
- [Quick Start](#-quick-start)
- [Dataset](#-dataset)
- [Tokenizer](#-tokenizer)
- [Model Configuration](#-model-configuration)
- [Training](#-training)
- [Fine-tuning](#-fine-tuning)
- [Generation](#-generation)
- [CPU and CUDA Backends](#-cpu-and-cuda-backends)
- [Checkpoint Format](#-checkpoint-format)
- [Troubleshooting](#-troubleshooting)
- [Current Status](#-current-status)
- [Future Direction](#-future-direction)
- [License](#-license)

---

## 🚀 Features

* 🧠 Transformer architecture in C++
* 🔢 Custom Tensor implementation
* 🔄 Automatic Differentiation (Autograd)
* 💻 CPU backend
* ⚡ Optional CUDA backend
* 🚀 cuBLAS for CUDA
* 🔤 Custom BPE tokenizer
* 🧩 Token Embedding
* 📐 Linear Layers
* 👀 Multi-Head Self-Attention
* 🔄 Rotary Positional Embeddings (RoPE)
* 📏 RMSNorm
* 🔥 Feed-Forward Network
* 🧱 Transformer Blocks
* 📉 Cross-Entropy Loss
* ⚙️ AdamW Optimizer
* 🗃️ KV Cache for autoregressive generation
* 💾 Model saving and loading
* 🎓 Training from scratch
* 🔧 Fine-tuning an existing checkpoint
* ✍️ Greedy Text Generation
* ⚙️ Configurable model and training parameters

---

## 🏗️ Architecture

### Forward pipeline (inference)

```text
Input text
    │
    ▼
BPE Tokenizer
    │
    ▼
Token IDs
    │
    ▼
Embedding
    │
    ▼
Transformer
    │
    ├── Multi-Head Attention
    │       ├── Q / K / V Projections
    │       ├── RoPE
    │       ├── Causal Mask
    │       └── KV Cache
    │
    ├── RMSNorm
    │
    └── Feed-Forward Network
    │
    ▼
Output Head
    │
    ▼
Logits
    │
    ▼
Sampling
    │
    ▼
Generated text
```

### Training pipeline

```text
Dataset
   │
   ▼
BPE Tokenizer
   │
   ▼
Token IDs
   │
   ▼
Training Batches
   │
   ▼
Transformer Forward Pass
   │
   ▼
Cross-Entropy Loss
   │
   ▼
Autograd Backward Pass
   │
   ▼
AdamW
   │
   ▼
Updated model
```

---

## 📁 Project Structure

```text
CppLLM/
├── CMakeLists.txt
│
├── Engine/
│   ├── Tensor/
│   ├── Attention/
│   ├── Autograd/
│   ├── Layers/
│   ├── Transformer/
│   ├── Normalization/
│   └── Tokenizer/
│
├── Models/
│   └── ToyModel/
│       ├── config.h
│       ├── tokenize.cpp
│       ├── train.cpp
│       ├── finetune.cpp
│       └── generate.cpp
│
├── Data/
│   └── data_generator.cpp
│
├── quickstart.sh
├── finetune.sh
├── README.md
└── README.ru.md
```

Generated datasets, tokenizer files, model checkpoints, and build files are intentionally not stored in the repository.

---

## 🛠️ Requirements

### 💻 CPU

* C++20 compiler
* CMake 3.18+
* BLAS on non-Apple systems
* macOS, Linux, or another system with the required C++ toolchain

On macOS, the project uses Apple Accelerate for BLAS.

On Linux, you can install OpenBLAS, for example:

```bash
sudo apt update
sudo apt install libopenblas-dev
```

### ⚡ CUDA (optional)

CUDA is optional. To build with CUDA support, you need:

* NVIDIA GPU
* CUDA Toolkit
* C++ compiler compatible with the installed CUDA Toolkit version

When CUDA is enabled, the project is built for NVIDIA GPUs with compute capability 7.5.

During development, the CUDA part was tested on an NVIDIA Tesla T4.

---

## 🔨 Build

### 💻 CPU (default)

From the project root:

```bash
cmake -S . -B build
cmake --build build -j2
```

### ⚡ CUDA

```bash
cmake -S . -B build -DENABLE_CUDA=ON
cmake --build build -j2
```

CUDA is disabled by default, so the project can be built on machines without the CUDA Toolkit.

CUDA-specific sources are compiled only when CUDA support is enabled. The backend architecture:

```text
Tensor
  │
  ▼
Backend
  │
  ├── CPUBackend
  │
  └── CUDABackend
```

---

## 🚀 Quick Start

The project provides a full pipeline for the toy model:

```text
tokenize → train → generate
```

**Before the first run**, build the project and generate the dataset (see [Dataset](#-dataset)):

```bash
cmake -S . -B build
cmake --build build -j2
./build/data_generator
```

Then run quickstart **from the project root**:

```bash
./quickstart.sh
```

The script:

1. configures CMake;
2. builds the project;
3. runs the tokenizer;
4. trains the model;
5. runs generation.

By default, `quickstart.sh` uses the CPU build. `finetune` is **not** part of quickstart — it runs separately (see [Fine-tuning](#-fine-tuning)).

### CUDA

To run the pipeline with CUDA:

```bash
cmake -S . -B build -DENABLE_CUDA=ON
cmake --build build -j2
./build/tokenize
./build/train
./build/generate
```

### What each program does

| Program | Purpose |
|---|---|
| `./build/data_generator` | Generates the toy dataset into `Data/data.txt` |
| `./build/tokenize` | Trains the BPE tokenizer and saves it to `Models/ToyModel/ToyTokenizer` |
| `./build/train` | Trains the model from scratch and saves a checkpoint to `Models/ToyModel/ModelFiles` |
| `./build/finetune` | Loads a checkpoint and continues training |
| `./build/generate` | Loads a checkpoint and generates text from a prompt |

### Expected time

With the current toy configuration:

| Backend | Time per epoch |
|---|---|
| CPU | ~1 second |
| CUDA (Tesla T4) | ~20 ms |

---

## 📊 Dataset

The repository contains a **generator** for a small dataset, not the generated dataset itself. This is intentional, so the dataset does not take up space and can be regenerated at any time.

### Generation

First, build the project (see [Build](#-build)):

```bash
cmake -S . -B build
cmake --build build -j2
```

Then run the generator:

```bash
./build/data_generator
```

### Contents

The current toy dataset contains:

* 📌 facts;
* 📖 definitions;
* 🔗 associations;
* 🧩 simple logical questions.

Dataset language: **English**.

### Why it exists

The dataset is intentionally simple so that each part of the pipeline can be tested in isolation:

* the model’s ability to learn;
* loss correctness;
* Autograd behavior;
* weight updates;
* model saving;
* text generation.

The dataset generator is located here:

```text
Data/data_generator.cpp
```

---

## 🔤 Tokenizer

CppLLM uses its own **byte-level BPE tokenizer**.

The tokenizer starts with 256 possible byte values and then learns additional tokens until the configured vocabulary size is reached.

The `<END>` token is reserved as the **last** token in the vocabulary.

For example, with vocabulary size 1000:

```text
Vocabulary size: 1000
<END> token id: 999
```

The tokenizer supports:

* vocabulary training;
* BPE merges;
* text encoding;
* token decoding;
* tokenizer saving;
* tokenizer loading.

The tokenizer is trained by `./build/tokenize` and saved to `Models/ToyModel/ToyTokenizer`.

---

## ⚙️ Model Configuration

The main parameters of the toy model are located in:

```text
Models/ToyModel/config.h
```

### Model parameters

| Constant | Value | Description |
|---|---|---|
| `VOCAB_SIZE` | 1000 | Tokenizer vocabulary size. Minimum is 256 (byte-level BPE). |
| `EMBED_DIM` | 32 | Token embedding dimension. |
| `BLOCKS` | 2 | Number of Transformer blocks. |
| `HEADS` | 2 | Number of heads in Multi-Head Attention. `EMBED_DIM` must be divisible by `HEADS`. |
| `HIDDEN` | 64 | Hidden layer size in the Feed-Forward Network. |
| `CONTEXT` | 32 | Maximum context length (in tokens). |

### Training parameters

| Constant | Value | Description |
|---|---|---|
| `BATCH_SIZE` | 16 | Batch size. |
| `STEPS` | 100 | Number of training steps. |
| `LEARNING_RATE` | 0.001 | Learning rate for AdamW. |
| `WEIGHT_DECAY` | 0.0 | Weight decay for AdamW. |
| `BETA1` | 0.9 | AdamW β₁ (exponential smoothing of the first moment). |
| `BETA2` | 0.999 | AdamW β₂ (exponential smoothing of the second moment). |
| `EPS` | 1e-8 | ε for AdamW numerical stability. |

### Miscellaneous

| Constant | Value | Description |
|---|---|---|
| `SEED` | 42 | Seed for the random number generator. |
| `VALIDATION_EVERY` | 100 | How often (in steps) to run validation. |
| `VALIDATION_BATCHES` | 10 | How many batches to use for validation. |
| `TRAINING_DEVICE` | CPU / CUDA | Selected automatically based on the `ENABLE_CUDA` flag. |

### Paths

| Constant | Value | Description |
|---|---|---|
| `PROJECT_ROOT` | — | Project root. Located by the presence of the `Data/` folder. |
| `DATA_PATH` | `Data/data.txt` | Path to the dataset. |
| `TOKENIZER_PATH` | `Models/ToyModel/ToyTokenizer` | Where the tokenizer is saved. |
| `MODEL_PATH` | `Models/ToyModel/ModelFiles` | Where the checkpoint is saved. |

Model and training parameters are separated from the training programs themselves: to change the configuration, edit `config.h` and rebuild the project.

---

## 🎓 Training

Training is implemented in:

```text
Models/ToyModel/train.cpp
```

The training process performs:

1. 📂 dataset loading;
2. 🔤 tokenizer loading;
3. 🔢 text encoding;
4. ✂️ train/validation split;
5. 🎲 random batch generation;
6. ➡️ Forward Pass;
7. 📉 Cross-Entropy Loss computation;
8. ⬅️ Backward Pass through Autograd;
9. ⚙️ parameter updates via AdamW;
10. 🧪 Validation;
11. 💾 checkpoint saving.

Run:

```bash
./build/train
```

The model can run on CPU or CUDA depending on the CMake configuration.

---

## 🔧 Fine-tuning

Fine-tuning is implemented separately in:

```text
Models/ToyModel/finetune.cpp
```

Unlike `train.cpp`, the fine-tuning program loads an existing checkpoint and continues training an already trained model.

```text
train
  │
  ▼
Model Checkpoint
  │
  ▼
finetune
  │
  ├── Load Model
  ├── Continue Training
  └── Save Model
      │
      ▼
Updated Checkpoint
```

The optimizer state is also saved in the checkpoint, so training can resume from the current AdamW state.

Run:

```bash
./finetune.sh
```

or directly:

```bash
./build/finetune
```

`finetune` is intentionally not part of `quickstart.sh`, since quickstart is meant to run a fresh full training pipeline.

---

## ✍️ Generation

Text generation is implemented in:

```text
Models/ToyModel/generate.cpp
```

Generation pipeline:

```text
Prompt
  │
  ▼
Tokenizer
  │
  ▼
Token IDs
  │
  ▼
Transformer
  │
  ▼
Logits
  │
  ▼
Sampling
  │
  ▼
Generated Text
```

### Run

```bash
./build/generate
```

### Example output

The model is trained on a toy dataset, so the answers are often meaningless — this is expected and confirms that the pipeline works end-to-end, not that the model is "smart".

```text
Prompt: Question: Can you answer this question? What is a computer? Answer:
Output: A vehicle.

Prompt: Question: Can you answer this question? What is the opposite of fast? Answer:
Output: Dublin.

Prompt: Question: Tell me the answer to this: What color is the sky on a clear day? Answer:
Output: Ice.

Prompt: Question: Is 55 equal to 56? Answer:
Output: No.

Prompt: Question: Is 29 smaller than 93? Answer:
Output: No.

Prompt: Question: Is 100 not equal to 23? Answer:
Output: Yes.

Prompt: Question: Is 46 smaller than 11? Answer:
Output: No.

Prompt: Question: Can you answer this question? How many minutes are in two hours? Answer:
Output: 60.

Prompt: Question: Is 64 not equal to 19? Answer:
Output: Yes.
```

### KV Cache

The Transformer supports **KV Cache** for autoregressive inference.

The current generation path optionally **disables** KV Cache to simplify correctness testing.

The KV Cache implementation itself is part of the Transformer architecture and is used in inference mode.

---

## 💻 CPU and CUDA Backends

The project uses a device-aware backend architecture. Build instructions and requirements are described in [Build](#-build) and [Requirements](#-requirements).

- **CPU** is used by default.
- **CUDA** is enabled with the `-DENABLE_CUDA=ON` flag.

CUDA-specific sources are compiled only when CUDA support is enabled. This allows the project to be built on systems without NVIDIA CUDA.

---

## 💾 Checkpoint Format

The checkpoint is saved to `Models/ToyModel/ModelFiles` and contains **model weights**.

Loading happens in `finetune.cpp` and `generate.cpp`.

---

## 🧪 Troubleshooting

**`Cannot find project root`**

`GetProjectRoot()` looks for the `Data/` folder in the current directory and in the parent directory. Run programs **from the project root** or make sure `Data/` exists.

**`corpus is empty`**

The dataset has not been generated. Build the project and run the generator:

```bash
cmake -S . -B build
cmake --build build -j2
./build/data_generator
```

**`vocab_size must be >= 256`**

The BPE tokenizer requires at least 256 tokens (byte-level). Increase `VOCAB_SIZE` in `config.h`.

**OpenBLAS not found on Linux**

Install it:

```bash
sudo apt install libopenblas-dev
```

and rebuild the project.

**CUDA Toolkit not found with `-DENABLE_CUDA=ON`**

Make sure `nvcc` is available in `PATH`, and that the CUDA Toolkit version is compatible with your C++ compiler.

---

## 📌 Current Status

The main pipeline works:

```text
Text
 ↓
BPE
 ↓
Token IDs
 ↓
Embedding
 ↓
Transformer
 ↓
Loss
 ↓
Autograd
 ↓
AdamW
 ↓
Checkpoint
 ↓
Generation
```

The project is at an **experimental stage** and is primarily intended for:

* training;
* experiments;
* studying the internals of a Transformer;
* studying Autograd;
* studying GPU/CUDA backends;
* experiments with training small language models.

### Current model

The current toy model configuration is intentionally small:

```text
Vocabulary:           1000
Embedding size:       32
Transformer blocks:   2
Attention heads:      2
Hidden size:          64
Context length:       32
```

These parameters are primarily intended for development and testing of the implementation.

During development, the project was also used with larger configurations, including a Transformer trained on a book-sized corpus.

### Limitations

* Only **greedy sampling** is supported.
* The model is not meant to compete with production LLM frameworks or large pretrained language models.
* Training is designed for small corpora.

---

## 🎯 Future Direction

The current focus is on studying how LLMs work. The project will primarily evolve as an educational one — emphasizing clarity of implementation rather than performance or scale.

---

## 📄 License

The project is intended for educational and experimental purposes.