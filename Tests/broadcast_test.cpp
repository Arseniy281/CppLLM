#include "../Engine/Tensor/tensor.h"
#include <iostream>

void print_shape(const std::string& name, const Tensor& t) {
    std::cout << name << " shape: (";
    auto shape = t.GetShape();
    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i];
        if (i + 1 < shape.size()) std::cout << ", ";
    }
    std::cout << ")\n";
}

void print_data(const std::string& name, const Tensor& t) {
    std::cout << name << " data: ";
    t.print();
}

int main() {
    std::cout << "=== Broadcast Test ===\n\n";

    std::cout << "Test 1: 1D + 1D\n";
    {
        Tensor a({3}, {1, 2, 3});
        Tensor b({3}, {4, 5, 6});
        print_shape("a", a);
        print_shape("b", b);
        Tensor c = a + b;
        print_data("c", c);
    }
    std::cout << "\n";

    std::cout << "Test 2: 2D + 1D\n";
    {
        Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
        Tensor b({3}, {10, 20, 30});
        print_shape("a", a);
        print_shape("b", b);
        Tensor c = a + b;
        print_data("c", c);
    }
    std::cout << "\n";

    std::cout << "Test 3: 3D + 2D\n";
    {
        Tensor a({2, 3, 4});
        Tensor b({3, 4});
        for (size_t i = 0; i < a.GetSize(); i++) {
            a.Data()[i] = static_cast<float>(i);
        }
        for (size_t i = 0; i < b.GetSize(); i++) {
            b.Data()[i] = static_cast<float>(i + 100);
        }
        print_shape("a", a);
        print_shape("b", b);
        Tensor c = a + b;
        print_data("c", c);
    }
    std::cout << "\n";

    std::cout << "Test 4: 3D + 1D\n";
    {
        Tensor a({2, 3, 4});
        Tensor b({1}, 5.0f);
        for (size_t i = 0; i < a.GetSize(); i++) {
            a.Data()[i] = static_cast<float>(i);
        }
        print_shape("a", a);
        print_shape("b", b);
        Tensor c = a + b;
        print_data("c", c);
    }
    std::cout << "\n";

    std::cout << "Test 5: incompatible shapes\n";
    {
        Tensor a({2, 3});
        Tensor b({4, 5});
        print_shape("a", a);
        print_shape("b", b);
        try {
            Tensor c = a + b;
            std::cout << "❌ Should not reach here!\n";
        } catch (const std::exception& e) {
            std::cout << "✅ Exception: " << e.what() << "\n";
        }
    }

    std::cout << "\n=== Broadcast Test Completed ===\n";
    return 0;
}