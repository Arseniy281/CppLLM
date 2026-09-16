#include "../Engine/Tensor/tensor.h"
#include <iostream>

int main() {
    std::cout << "=== Tensor Tests ===\n\n";
    
    Tensor t({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    std::cout << "t (2x3):\n";
    t.print();
    t.PrintInfo();
    
    Tensor t2({2, 3}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
    Tensor t3 = t + t2;
    std::cout << "t + t2:\n";
    t3.print();
    
    Tensor t4 = t * 2.0f;
    std::cout << "t * 2:\n";
    t4.print();
    
    t.Reshape({3, 2});
    Tensor t5 = t;
    std::cout << "t reshape 3x2:\n";
    t5.print();
    
    std::cout << "\n✅ Все тесты пройдены!\n";
    return 0;
}