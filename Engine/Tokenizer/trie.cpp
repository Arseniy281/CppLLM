#include "trie.h"

#include <string>
#include <vector>

Trie::Trie(const std::vector<std::string>& vocab) {
    for (const auto& word : vocab) {
        Insert(word);
    }
}

Trie::~Trie() {
    Clear();
}

void Trie::ClearNode(Node* node) {
    for (auto& [symbol, child] : node->children) {
        ClearNode(child);
        delete child;
    }

    node->children.clear();
    node->is_end_ = false;
}

void Trie::Clear() {
    ClearNode(&root_);
}

void Trie::Insert(const std::string& word) {
    Node* cur_node = &root_;

    for (const char& letter : word) {
        if (cur_node->children.find(letter) == cur_node->children.end()) {
            cur_node->children[letter] = new Node();
        }

        cur_node = cur_node->children[letter];
    }
    cur_node->is_end_ = true;
}

std::string Trie::LongestToken(const std::string& text, size_t index) {
    Node* cur_node = &root_;
    std::string ans;
    std::string cur_prefix;

    for (size_t i = index; i < text.size(); i++) {
        auto it = cur_node->children.find(text[i]);

        if (it == cur_node->children.end()) { return ans; }
        cur_node = it->second;
        cur_prefix += text[i];

        if (cur_node->is_end_) {
            ans = cur_prefix;
        }
    }

    return ans;
}