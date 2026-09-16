#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <unordered_set>
#include <cstdint>
#include "trie.h"

class BPETokenizer {
private:
    struct Word {
        size_t frequency = 0;
        std::vector<int> tokens;
    };

    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> token_to_id_;
    Trie trie;

    std::vector<std::vector<std::string>> SplitInVector(const std::string& str);
    std::vector<std::string> Split(const std::string& str);

    void UpdateWords(std::vector<std::vector<std::string>>& words, 
        const std::string& best_pair);
    std::string GetBestPair(const std::unordered_map<std::string, int>& pair_counts);

    void RemoveWordPairs(const Word& word, size_t word_id,
        std::unordered_map<uint64_t, size_t>& pair_counts,
        std::unordered_map<uint64_t, std::unordered_set<size_t>>& pair_words);

    void AddWordPairs(const Word& word, size_t word_id,
        std::unordered_map<uint64_t, size_t>& pair_counts,
        std::unordered_map<uint64_t, std::unordered_set<size_t>>& pair_words);

    bool FindBestPair(
        const std::unordered_map<uint64_t, size_t>& pair_counts,
        uint64_t& best_key, size_t& best_frequency);

    void BuildPairStats(const std::vector<Word>& words,
        std::unordered_map<uint64_t, size_t>& pair_counts,
        std::unordered_map<uint64_t, std::unordered_set<size_t>>& pair_words);

    std::vector<Word> BuildWords(const std::string& corpus);

    void ValidateTrainInput(const std::string& corpus, size_t vocab_size);

    bool ApplyMergeToWord(Word& word, int left, int right, int new_id);
    void RunMerges(std::vector<Word>& words, size_t vocab_size);

public:
    BPETokenizer();

    void AddToken(const std::string& token);
    void Train(const std::string& corpus, size_t vocab_size=1000);

    void Save(const std::string& filename);
    void Load(const std::string& filename);

    std::vector<size_t> Encode(const std::string& str);
    std::string Decode(std::vector<size_t> ids);
    size_t GetVocabSize() const;
    size_t GetTokenId(const std::string& token) const;

};