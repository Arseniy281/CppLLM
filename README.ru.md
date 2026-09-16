# CppLLM

CppLLM — экспериментальная реализация Large Language Model, написанная с нуля на C++.

Проект реализует основные компоненты, необходимые для создания, обучения, дообучения и запуска небольшой Transformer-модели без использования высокоуровневых deep learning-фреймворков.

🌐 [English version](README.md) · 🇷🇺 Русская версия

---

## Содержание

- [Возможности](#-возможности)
- [Архитектура](#-архитектура)
- [Структура проекта](#-структура-проекта)
- [Требования](#-требования)
- [Сборка](#-сборка)
- [Быстрый запуск](#-быстрый-запуск)
- [Dataset](#-dataset)
- [Tokenizer](#-tokenizer)
- [Конфигурация модели](#-конфигурация-модели)
- [Обучение](#-обучение)
- [Дообучение](#-дообучение)
- [Генерация](#-генерация)
- [CPU и CUDA backends](#-cpu-и-cuda-backends)
- [Формат checkpoint](#-формат-checkpoint)
- [Troubleshooting](#-troubleshooting)
- [Текущий статус](#-текущий-статус)
- [Направление развития](#-направление-развития)
- [License](#-license)

---

## 🚀 Возможности

* 🧠 Transformer-архитектура на C++
* 🔢 Собственная реализация Tensor
* 🔄 Automatic Differentiation (Autograd)
* 💻 CPU backend
* ⚡ Опциональный CUDA backend
* 🚀 Использование cuBLAS для CUDA
* 🔤 Собственная реализация BPE tokenizer
* 🧩 Token Embedding
* 📐 Linear Layers
* 👀 Multi-Head Self-Attention
* 🔄 Rotary Positional Embeddings (RoPE)
* 📏 RMSNorm
* 🔥 Feed-Forward Network
* 🧱 Transformer Blocks
* 📉 Cross-Entropy Loss
* ⚙️ AdamW Optimizer
* 🗃️ KV Cache для авторегрессионной генерации
* 💾 Сохранение и загрузка модели
* 🎓 Обучение модели с нуля
* 🔧 Дообучение существующего checkpoint
* ✍️ Greedy Text Generation
* ⚙️ Настраиваемые параметры модели и обучения

---

## 🏗️ Архитектура

### Forward pipeline (inference)

```text
Входной текст
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
Сгенерированный текст
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
Обновлённая модель
```

---

## 📁 Структура проекта

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

Сгенерированные датасеты, tokenizer files, checkpoints моделей и build-файлы намеренно не хранятся в репозитории.

---

## 🛠️ Требования

### 💻 CPU

* C++20 compiler
* CMake 3.18+
* BLAS на системах, отличных от Apple
* macOS, Linux или другая система с необходимым C++ toolchain

На macOS проект использует Apple Accelerate для BLAS.

На Linux можно установить, например, OpenBLAS:

```bash
sudo apt update
sudo apt install libopenblas-dev
```

### ⚡ CUDA (опционально)

CUDA является опциональной. Для CUDA-сборки необходимы:

* NVIDIA GPU
* CUDA Toolkit
* C++ compiler, совместимый с установленной версией CUDA Toolkit

При включении CUDA проект собирается под NVIDIA GPU с compute capability 7.5.

Во время разработки CUDA-часть тестировалась на NVIDIA Tesla T4.

---

## 🔨 Сборка

### 💻 CPU (по умолчанию)

Из корня проекта:

```bash
cmake -S . -B build
cmake --build build -j2
```

### ⚡ CUDA

```bash
cmake -S . -B build -DENABLE_CUDA=ON
cmake --build build -j2
```

CUDA по умолчанию отключена, поэтому проект можно собрать на компьютере без CUDA Toolkit.

CUDA-specific исходники компилируются только при включённой поддержке CUDA. Архитектура backend’ов:

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

## 🚀 Быстрый запуск

Проект содержит полный pipeline для toy-модели:

```text
tokenize → train → generate
```

**Перед первым запуском** нужно собрать проект и сгенерировать датасет (см. [Dataset](#-dataset)):

```bash
cmake -S . -B build
cmake --build build -j2
./build/data_generator
```

Затем запустить quickstart **из корня проекта**:

```bash
./quickstart.sh
```

Скрипт:

1. конфигурирует CMake;
2. собирает проект;
3. запускает tokenizer;
4. обучает модель;
5. запускает генерацию.

По умолчанию `quickstart.sh` использует CPU-сборку. `finetune` в quickstart **не входит** — он запускается отдельно (см. [Дообучение](#-дообучение)).

### CUDA

Для запуска pipeline с CUDA:

```bash
cmake -S . -B build -DENABLE_CUDA=ON
cmake --build build -j2
./build/tokenize
./build/train
./build/generate
```

### Что делает каждая программа

| Программа | Назначение |
|---|---|
| `./build/data_generator` | Генерирует toy-датасет в `Data/data.txt` |
| `./build/tokenize` | Обучает BPE tokenizer и сохраняет его в `Models/ToyModel/ToyTokenizer` |
| `./build/train` | Обучает модель с нуля и сохраняет checkpoint в `Models/ToyModel/ModelFiles` |
| `./build/finetune` | Загружает checkpoint и продолжает обучение |
| `./build/generate` | Загружает checkpoint и генерирует текст по prompt’у |

### Ожидаемое время

С текущей toy-конфигурацией:

| Backend | Время на эпоху |
|---|---|
| CPU | ~1 секунда |
| CUDA (Tesla T4) | ~20 мс |

---

## 📊 Dataset

В репозитории находится **генератор** небольшого датасета, а не сам сгенерированный dataset. Это сделано намеренно, чтобы датасет не занимал место и его можно было пересоздать.

### Генерация

Сначала соберите проект (см. [Сборка](#-сборка)):

```bash
cmake -S . -B build
cmake --build build -j2
```

Затем запустите генератор:

```bash
./build/data_generator
```

### Содержимое

Текущий toy dataset содержит:

* 📌 факты;
* 📖 определения;
* 🔗 ассоциации;
* 🧩 простые логические вопросы.

Язык датасета: **английский**.

### Зачем он нужен

Датасет специально сделан простым, чтобы можно было отдельно проверять:

* способность модели обучаться;
* корректность loss;
* работу Autograd;
* обновление весов;
* сохранение модели;
* генерацию текста.

Генератор датасета находится здесь:

```text
Data/data_generator.cpp
```

---

## 🔤 Tokenizer

CppLLM использует собственный **byte-level BPE tokenizer**.

Tokenizer начинает с 256 возможных значений байта и затем обучает дополнительные токены, пока не будет достигнут заданный размер vocabulary.

Токен `<END>` зарезервирован как **последний** токен vocabulary.

Например, при размере vocabulary 1000:

```text
Vocabulary size: 1000
<END> token id: 999
```

Tokenizer поддерживает:

* обучение vocabulary;
* BPE merges;
* кодирование текста;
* декодирование токенов;
* сохранение tokenizer;
* загрузку tokenizer.

Tokenizer обучается программой `./build/tokenize` и сохраняется в `Models/ToyModel/ToyTokenizer`.

---

## ⚙️ Конфигурация модели

Основные параметры toy-модели находятся в:

```text
Models/ToyModel/config.h
```

### Параметры модели

| Константа | Значение | Описание |
|---|---|---|
| `VOCAB_SIZE` | 1000 | Размер словаря tokenizer’а. Минимум — 256 (byte-level BPE). |
| `EMBED_DIM` | 32 | Размерность эмбеддингов токенов. |
| `BLOCKS` | 2 | Количество Transformer-блоков. |
| `HEADS` | 2 | Количество голов в Multi-Head Attention. `EMBED_DIM` должно делиться на `HEADS`. |
| `HIDDEN` | 64 | Размер скрытого слоя в Feed-Forward Network. |
| `CONTEXT` | 32 | Максимальная длина контекста (в токенах). |

### Параметры обучения

| Константа | Значение | Описание |
|---|---|---|
| `BATCH_SIZE` | 16 | Размер батча. |
| `STEPS` | 100 | Количество шагов обучения. |
| `LEARNING_RATE` | 0.001 | Learning rate для AdamW. |
| `WEIGHT_DECAY` | 0.0 | Weight decay для AdamW. |
| `BETA1` | 0.9 | Параметр β₁ AdamW (экспоненциальное сглаживание первого момента). |
| `BETA2` | 0.999 | Параметр β₂ AdamW (экспоненциальное сглаживание второго момента). |
| `EPS` | 1e-8 | ε для численной стабильности AdamW. |

### Прочее

| Константа | Значение | Описание |
|---|---|---|
| `SEED` | 42 | Seed для генератора случайных чисел. |
| `VALIDATION_EVERY` | 100 | Как часто (в шагах) запускать validation. |
| `VALIDATION_BATCHES` | 10 | Сколько батчей использовать для validation. |
| `TRAINING_DEVICE` | CPU / CUDA | Выбирается автоматически по флагу `ENABLE_CUDA`. |

### Пути

| Константа | Значение | Описание |
|---|---|---|
| `PROJECT_ROOT` | — | Корень проекта. Ищется по наличию папки `Data/`. |
| `DATA_PATH` | `Data/data.txt` | Путь к датасету. |
| `TOKENIZER_PATH` | `Models/ToyModel/ToyTokenizer` | Куда сохраняется tokenizer. |
| `MODEL_PATH` | `Models/ToyModel/ModelFiles` | Куда сохраняется checkpoint. |

Параметры модели и обучения отделены от реализации training-программ: чтобы изменить конфигурацию, достаточно отредактировать `config.h` и пересобрать проект.

---

## 🎓 Обучение

Обучение реализовано в:

```text
Models/ToyModel/train.cpp
```

Процесс обучения выполняет:

1. 📂 загрузку dataset;
2. 🔤 загрузку tokenizer;
3. 🔢 кодирование текста;
4. ✂️ разделение на train/validation;
5. 🎲 генерацию случайных батчей;
6. ➡️ Forward Pass;
7. 📉 расчёт Cross-Entropy Loss;
8. ⬅️ Backward Pass через Autograd;
9. ⚙️ обновление параметров через AdamW;
10. 🧪 Validation;
11. 💾 сохранение checkpoint.

Запуск:

```bash
./build/train
```

Модель может работать как на CPU, так и на CUDA в зависимости от конфигурации CMake.

---

## 🔧 Дообучение

Дообучение реализовано отдельно в:

```text
Models/ToyModel/finetune.cpp
```

В отличие от `train.cpp`, программа дообучения загружает существующий checkpoint и продолжает обучение уже готовой модели.

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
Обновлённый Checkpoint
```

Состояние optimizer также сохраняется в checkpoint, поэтому обучение может продолжаться с текущего состояния AdamW.

Запуск:

```bash
./finetune.sh
```

или напрямую:

```bash
./build/finetune
```

`finetune` намеренно не включён в `quickstart.sh`, поскольку quickstart предназначен для запуска нового полного pipeline обучения.

---

## ✍️ Генерация

Генерация текста реализована в:

```text
Models/ToyModel/generate.cpp
```

Pipeline генерации:

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

### Запуск

```bash
./build/generate
```

### Пример вывода

Модель обучается на toy-датасете, поэтому ответы часто не имеют смысла — это ожидаемо и подтверждает, что pipeline работает end-to-end, а не что модель «умная».

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

Transformer содержит поддержку **KV Cache** для авторегрессионного inference.

Текущий путь генерации при необходимости **отключает** KV Cache для более простого тестирования корректности.

Сама реализация KV Cache является частью архитектуры Transformer и используется в inference-режиме.

---

## 💻 CPU и CUDA backends

Проект использует device-aware архитектуру backend’ов. Сборка и требования описаны в разделах [Сборка](#-сборка) и [Требования](#-требования).

- **CPU** используется по умолчанию.
- **CUDA** включается флагом `-DENABLE_CUDA=ON`.

CUDA-specific исходники компилируются только при включённой поддержке CUDA. Благодаря этому проект можно собирать на системах без NVIDIA CUDA.

---

## 💾 Формат checkpoint

Checkpoint сохраняется в `Models/ToyModel/ModelFiles` и содержит **веса модели**.

Загрузка происходит в `finetune.cpp` и `generate.cpp`.

---

## 🧪 Troubleshooting

**`Cannot find project root`**

`GetProjectRoot()` ищет папку `Data/` в текущей директории и в родительской. Запускайте программы **из корня проекта** или убедитесь, что `Data/` существует.

**`corpus is empty`**

Не сгенерирован датасет. Соберите проект и запустите генератор:

```bash
cmake -S . -B build
cmake --build build -j2
./build/data_generator
```

**`vocab_size must be >= 256`**

BPE tokenizer требует минимум 256 токенов (byte-level). Увеличьте `VOCAB_SIZE` в `config.h`.

**OpenBLAS не найден на Linux**

Установите:

```bash
sudo apt install libopenblas-dev
```

и пересоберите проект.

**CUDA Toolkit не найден при `-DENABLE_CUDA=ON`**

Убедитесь, что `nvcc` доступен в `PATH`, и что версия CUDA Toolkit совместима с вашим C++ compiler’ом.

---

## 📌 Текущий статус

Основной pipeline работает:

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

Проект находится на **экспериментальной стадии** и в первую очередь предназначен для:

* обучения;
* экспериментов;
* изучения устройства Transformer;
* изучения Autograd;
* изучения GPU/CUDA backend’ов;
* экспериментов с обучением небольших языковых моделей.

### Текущая модель

Текущая конфигурация toy-модели намеренно небольшая:

```text
Vocabulary:           1000
Embedding size:       32
Transformer blocks:   2
Attention heads:      2
Hidden size:          64
Context length:       32
```

Эти параметры в первую очередь предназначены для разработки и тестирования реализации.

Во время разработки проект также использовался с более крупными конфигурациями, включая Transformer, обучавшийся на корпусе размером с книгу.

### Ограничения

* Поддерживается только **greedy sampling**.
* Модель не предназначена для конкуренции с production LLM-фреймворками или большими предобученными языковыми моделями.
* Обучение рассчитано на небольшие корпуса.

---

## 🎯 Направление развития

Сейчас упор делается на изучение того, как работают LLM. Проект будет развиваться в первую очередь как обучающий — с акцентом на понятность реализации, а не на производительность или масштаб.

---

## 📄 License

Проект предназначен для образовательных и экспериментальных целей.