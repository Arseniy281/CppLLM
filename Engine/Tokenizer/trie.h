#pragma once

#include <vector>
#include <string>
#include <unordered_map>

class Trie {
private:
    struct Node {
        std::unordered_map<char, Node*> children;
        bool is_end_ = false;
    };

    Node root_;
    void ClearNode(Node* node);

public:
    Trie() = default;
    Trie(const std::vector<std::string>& vocab);
    ~Trie();

    Trie(const Trie&) = delete;
    Trie& operator=(const Trie&) = delete;

    void Insert(const std::string& word);
    void Clear();

    std::string LongestToken(
        const std::string& text,
        size_t index
    );
};