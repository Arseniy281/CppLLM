#include "bpe_tokenizer.h"
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <queue>

uint64_t MakePairKey(int a, int b) {
    return ((uint64_t)((uint32_t)(a)) << 32) | (uint32_t)(b);
}

std::vector<std::vector<std::string>> BPETokenizer::SplitInVector(
        const std::string& str) {
    std::vector<std::vector<std::string>> tokens;
    std::vector<std::string> current_word;

    for (char c : str) {
        if (c == ' ') {
            if (!current_word.empty()) {
                current_word.push_back(" ");
                tokens.push_back(current_word);
                current_word.clear();
            } else {
                tokens.push_back({" "});
            }
        } else if (c == '\n') {
            if (!current_word.empty()) {
                tokens.push_back(current_word);
                current_word.clear();
            }

            tokens.push_back({"\n"});
        } else {
            current_word.push_back(std::string(1, c));
        }
    }

    if (!current_word.empty()) {
        tokens.push_back(current_word);
    }

    return tokens;
}

std::vector<std::string> BPETokenizer::Split(const std::string& str) {
    std::vector<std::string> tokens;
    std::istringstream iss(str);
    std::string token;

    while (iss >> token) {
        tokens.push_back(token);
    }

    return tokens;
}

BPETokenizer::BPETokenizer() {
    for (int i = 0; i < 256; i++) {
        AddToken(std::string(1, (char)(i)));
    }
}

void BPETokenizer::AddToken(const std::string& token) {
    token_to_id_[token] = vocab_.size();
    vocab_.push_back(token);
    trie.Insert(token);
}

void BPETokenizer::UpdateWords(std::vector<std::vector<std::string>>& words,
        const std::string& best_pair) {
    for (auto& word : words) {
        size_t i = 0;
        while (i + 1 < word.size()) {
            if (word[i] + word[i + 1] == best_pair) {
                word[i] = best_pair;
                word.erase(word.begin() + i + 1);
            } else {
                ++i;
            }
        }
    }
}

void BPETokenizer::ValidateTrainInput(
        const std::string& corpus, size_t vocab_size) {

    if (vocab_size < 257) {
        throw std::runtime_error(
            "BPETokenizer::Train: vocab_size must be >= 257");
    }

    if (corpus.empty()) {
        throw std::runtime_error(
            "BPETokenizer::Train: corpus is empty");
    }
}

void BPETokenizer::RunMerges(
        std::vector<Word>& words, size_t vocab_size) {

    std::unordered_map<uint64_t, size_t> pair_counts;
    std::unordered_map<uint64_t, std::unordered_set<size_t>> pair_words;

    BuildPairStats(words, pair_counts, pair_words);

    size_t merge_number = 0;

    // One vocabulary slot is reserved for <END>.
    while (vocab_.size() < vocab_size - 1) {

        uint64_t best_key = 0;
        size_t best_frequency = 0;

        if (!FindBestPair(
                pair_counts,
                best_key,
                best_frequency)) {
            break;
        }

        int left = (int)(best_key >> 32);
        int right = (int)(best_key & 0xffffffffULL);

        if (left < 0 ||
            right < 0 ||
            (size_t)(left) >= vocab_.size() ||
            (size_t)(right) >= vocab_.size()) {

            throw std::runtime_error(
                "BPETokenizer::Train: invalid pair");
        }

        std::string merged =
            vocab_[left] + vocab_[right];

        if (token_to_id_.find(merged) != token_to_id_.end()) {
            pair_counts.erase(best_key);
            pair_words.erase(best_key);
            continue;
        }

        int new_id = (int)(vocab_.size());

        AddToken(merged);

        auto affected_it = pair_words.find(best_key);

        if (affected_it == pair_words.end()) {
            pair_counts.erase(best_key);
            continue;
        }

        std::vector<size_t> affected_words(
            affected_it->second.begin(),
            affected_it->second.end());

        for (size_t word_id : affected_words) {

            Word& word = words[word_id];

            if (word.tokens.size() < 2) {
                continue;
            }

            RemoveWordPairs(
                word,
                word_id,
                pair_counts,
                pair_words);

            ApplyMergeToWord(
                word,
                left,
                right,
                new_id);

            AddWordPairs(
                word,
                word_id,
                pair_counts,
                pair_words);
        }

        ++merge_number;

        if (merge_number % 25 == 0 ||
            vocab_.size() == vocab_size - 1) {

            std::cout
                << "BPE merge " << merge_number
                << " | vocab: " << vocab_.size()
                << " | pair frequency: "
                << best_frequency
                << "\n";
        }
    }
}

bool BPETokenizer::ApplyMergeToWord(
        Word& word,
        int left,
        int right,
        int new_id) {

    bool changed = false;

    size_t i = 0;

    while (i + 1 < word.tokens.size()) {

        if (word.tokens[i] == left &&
            word.tokens[i + 1] == right) {

            word.tokens[i] = new_id;

            word.tokens.erase(
                word.tokens.begin() + i + 1);

            changed = true;
        } else {
            ++i;
        }
    }

    return changed;
}

void BPETokenizer::RemoveWordPairs(
        const Word& word,
        size_t word_id,
        std::unordered_map<uint64_t, size_t>& pair_counts,
        std::unordered_map<uint64_t, std::unordered_set<size_t>>& pair_words) {

    for (size_t i = 0;
         i + 1 < word.tokens.size();
         i++) {

        uint64_t key =
            MakePairKey(
                word.tokens[i],
                word.tokens[i + 1]);

        auto count_it =
            pair_counts.find(key);

        if (count_it != pair_counts.end()) {

            if (count_it->second <= word.frequency) {
                pair_counts.erase(count_it);
            } else {
                count_it->second -= word.frequency;
            }
        }

        auto words_it =
            pair_words.find(key);

        if (words_it != pair_words.end()) {

            words_it->second.erase(word_id);

            if (words_it->second.empty()) {
                pair_words.erase(words_it);
            }
        }
    }
}

void BPETokenizer::AddWordPairs(
        const Word& word,
        size_t word_id,
        std::unordered_map<uint64_t, size_t>& pair_counts,
        std::unordered_map<uint64_t, std::unordered_set<size_t>>& pair_words) {

    for (size_t i = 0;
         i + 1 < word.tokens.size();
         i++) {

        uint64_t key =
            MakePairKey(
                word.tokens[i],
                word.tokens[i + 1]);

        pair_counts[key] += word.frequency;
        pair_words[key].insert(word_id);
    }
}

bool BPETokenizer::FindBestPair(
        const std::unordered_map<uint64_t, size_t>& pair_counts,
        uint64_t& best_key,
        size_t& best_frequency) {

    bool found = false;

    for (const auto& [key, frequency] : pair_counts) {

        if (!found ||
            frequency > best_frequency ||
            (frequency == best_frequency &&
             key < best_key)) {

            best_key = key;
            best_frequency = frequency;
            found = true;
        }
    }

    return found && best_frequency > 0;
}

void BPETokenizer::BuildPairStats(
        const std::vector<Word>& words,
        std::unordered_map<uint64_t, size_t>& pair_counts,
        std::unordered_map<uint64_t, std::unordered_set<size_t>>& pair_words) {

    for (size_t word_id = 0;
         word_id < words.size();
         word_id++) {

        const Word& word = words[word_id];

        if (word.tokens.size() < 2) {
            continue;
        }

        for (size_t i = 0;
             i + 1 < word.tokens.size();
             i++) {

            uint64_t key =
                MakePairKey(
                    word.tokens[i],
                    word.tokens[i + 1]);

            pair_counts[key] += word.frequency;
            pair_words[key].insert(word_id);
        }
    }
}

std::vector<BPETokenizer::Word>
BPETokenizer::BuildWords(
        const std::string& corpus) {

    std::unordered_map<std::string, size_t> word_counts;

    std::string current;
    current.reserve(64);

    auto flush_current = [&]() {

        if (!current.empty()) {
            ++word_counts[current];
            current.clear();
        }
    };

    for (char c : corpus) {

        if (c == '\n') {

            flush_current();

            ++word_counts["\n"];

        } else if (c == ' ') {

            if (!current.empty()) {

                current.push_back(' ');

                ++word_counts[current];

                current.clear();

            } else {
                ++word_counts[" "];
            }

        } else {

            current.push_back(c);
        }
    }

    flush_current();

    std::vector<Word> words;

    words.reserve(word_counts.size());

    for (const auto& [text, frequency] : word_counts) {

        Word word;

        word.frequency = frequency;

        word.tokens.reserve(text.size());

        for (unsigned char c : text) {
            word.tokens.push_back((int)(c));
        }

        words.push_back(std::move(word));
    }

    return words;
}

void BPETokenizer::Train(
        const std::string& corpus,
        size_t vocab_size) {

    ValidateTrainInput(corpus, vocab_size);

    if (vocab_.size() >= vocab_size) {
        return;
    }

    // <END> is a special token.
    // It must not participate in BPE training.
    std::string training_corpus = corpus;

    size_t pos = 0;

    while ((pos = training_corpus.find("<END>", pos))
           != std::string::npos) {

        training_corpus.erase(pos, 5);
    }

    std::vector<Word> words =
        BuildWords(training_corpus);

    // Train normal BPE tokens.
    // The last vocabulary slot is reserved for <END>.
    RunMerges(words, vocab_size);

    // Add <END> as the final token.
    if (vocab_.size() == vocab_size - 1) {
        AddToken("<END>");
    }

    if (vocab_.size() != vocab_size) {
        throw std::runtime_error(
            "BPETokenizer::Train: failed to create requested vocabulary");
    }
}

void BPETokenizer::Save(
        const std::string& filename) {

    std::ofstream file(
        filename,
        std::ios::binary);

    if (!file.is_open()) {

        throw std::runtime_error(
            "Cannot open file for writing: " +
            filename);
    }

    uint64_t vocab_size =
        vocab_.size();

    file.write(
        reinterpret_cast<const char*>(&vocab_size),
        sizeof(vocab_size));

    for (const auto& token : vocab_) {

        uint64_t length =
            token.size();

        file.write(
            reinterpret_cast<const char*>(&length),
            sizeof(length));

        if (length > 0) {

            file.write(
                token.data(),
                static_cast<std::streamsize>(
                    token.size()));
        }
    }

    if (!file) {

        throw std::runtime_error(
            "Error while writing tokenizer: " +
            filename);
    }

    file.close();
}

void BPETokenizer::Load(
        const std::string& filename) {

    std::ifstream file(
        filename,
        std::ios::binary);

    if (!file.is_open()) {

        throw std::runtime_error(
            "Cannot open file for reading: " +
            filename);
    }

    vocab_.clear();
    token_to_id_.clear();
    trie.Clear();

    uint64_t vocab_size = 0;

    file.read(
        reinterpret_cast<char*>(&vocab_size),
        sizeof(vocab_size));

    if (!file) {

        throw std::runtime_error(
            "Invalid tokenizer file: " +
            filename);
    }

    for (uint64_t i = 0;
         i < vocab_size;
         i++) {

        uint64_t length = 0;

        file.read(
            reinterpret_cast<char*>(&length),
            sizeof(length));

        if (!file) {

            throw std::runtime_error(
                "Invalid tokenizer file: " +
                filename);
        }

        std::string token(
            length,
            '\0');

        if (length > 0) {

            file.read(
                token.data(),
                static_cast<std::streamsize>(
                    length));
        }

        if (!file) {

            throw std::runtime_error(
                "Invalid tokenizer file: " +
                filename);
        }

        AddToken(token);
    }

    file.close();

    if (vocab_.empty() ||
        vocab_.back() != "<END>") {

        throw std::runtime_error(
            "BPETokenizer::Load: <END> token is not reserved as the last token");
    }
}

std::vector<size_t>
BPETokenizer::Encode(
        const std::string& str) {

    std::vector<size_t> ids;

    ids.reserve(str.size());

    size_t i = 0;

    while (i < str.size()) {

        // <END> is a special token.
        if (i + 5 <= str.size() &&
            str.compare(i, 5, "<END>") == 0) {

            ids.push_back(
                vocab_.size() - 1);

            i += 5;

            continue;
        }

        std::string prefix =
            trie.LongestToken(str, i);

        if (!prefix.empty()) {

            auto it =
                token_to_id_.find(prefix);

            if (it == token_to_id_.end()) {

                throw std::runtime_error(
                    "Tokenizer trie contains unknown token");
            }

            // <END> must only be encoded
            // through the special-token path above.
            if (prefix == "<END>") {

                ids.push_back(
                    vocab_.size() - 1);

                i += 5;

                continue;
            }

            ids.push_back(
                it->second);

            i += prefix.size();

        } else {

            auto it =
                token_to_id_.find(
                    std::string(1, str[i]));

            if (it == token_to_id_.end()) {

                throw std::runtime_error(
                    "Cannot encode character");
            }

            ids.push_back(
                it->second);

            ++i;
        }
    }

    return ids;
}

std::string BPETokenizer::Decode(
        std::vector<size_t> ids) {

    std::string result;

    for (size_t id : ids) {

        if (id >= vocab_.size()) {

            throw std::runtime_error(
                "BPETokenizer::Decode: invalid token id");
        }

        result += vocab_[id];
    }

    return result;
}

size_t BPETokenizer::GetVocabSize() const {
    return vocab_.size();
}

size_t BPETokenizer::GetTokenId(
        const std::string& token) const {

    return token_to_id_.at(token);
}