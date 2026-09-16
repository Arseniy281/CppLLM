#include "../Engine/Tensor/tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>

void TestMatMul2D() {
    std::cout << "Testing MatMul 2D...\n";
    
    Tensor A({2, 3}, {1.0f, 2.0f, 3.0f,
                      4.0f, 5.0f, 6.0f});
    Tensor B({3, 2}, {7.0f, 8.0f,
                      9.0f, 10.0f,
                      11.0f, 12.0f});
    
    Tensor C = A.MatMul(B);
    
    assert(C.GetShape()[0] == 2);
    assert(C.GetShape()[1] == 2);
    assert(std::abs(C.at({0, 0}) - 58.0f) < 1e-5f);
    assert(std::abs(C.at({0, 1}) - 64.0f) < 1e-5f);
    assert(std::abs(C.at({1, 0}) - 139.0f) < 1e-5f);
    assert(std::abs(C.at({1, 1}) - 154.0f) < 1e-5f);
    
    std::cout << "  ✅ MatMul 2D OK\n";
}

void TestMatMul3Dx2D() {
    std::cout << "Testing MatMul 3D x 2D...\n";
    
    Tensor A({2, 2, 3});
    A.at({0, 0, 0}) = 1.0f; A.at({0, 0, 1}) = 2.0f; A.at({0, 0, 2}) = 3.0f;
    A.at({0, 1, 0}) = 4.0f; A.at({0, 1, 1}) = 5.0f; A.at({0, 1, 2}) = 6.0f;
    A.at({1, 0, 0}) = 7.0f; A.at({1, 0, 1}) = 8.0f; A.at({1, 0, 2}) = 9.0f;
    A.at({1, 1, 0}) = 10.0f; A.at({1, 1, 1}) = 11.0f; A.at({1, 1, 2}) = 12.0f;
    
    Tensor B({3, 2}, {1.0f, 2.0f,
                      3.0f, 4.0f,
                      5.0f, 6.0f});
    
    Tensor C = A.MatMul(B);
    
    assert(C.GetShape()[0] == 2);
    assert(C.GetShape()[1] == 2);
    assert(C.GetShape()[2] == 2);
    
    assert(std::abs(C.at({0, 0, 0}) - 22.0f) < 1e-5f);
    assert(std::abs(C.at({0, 0, 1}) - 28.0f) < 1e-5f);
    assert(std::abs(C.at({0, 1, 0}) - 49.0f) < 1e-5f);
    assert(std::abs(C.at({0, 1, 1}) - 64.0f) < 1e-5f);
    
    std::cout << "  ✅ MatMul 3D x 2D OK\n";
}

int main() {
    std::cout << "\n=== MatMul Tests ===\n\n";
    TestMatMul2D();
    TestMatMul3Dx2D();
    std::cout << "\n✅ All MatMul tests passed!\n";
    return 0;
}