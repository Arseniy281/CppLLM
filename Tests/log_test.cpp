#include "../Engine/Tensor/tensor.h"
#include "../Engine/Layers/linear_layer.h"
#include "../Engine/Layers/mse_loss.h"
#include "../Engine/Autograd/relu.h"
#include "../Engine/Autograd/sigmoid.h"
#include "../Engine/Autograd/tanh_op.h"
#include <cmath>
#include <vector>
#include <iostream>

class LogModel {
public:
    LinearLayer layer1;
    std::shared_ptr<TanhOp> tanh1;
    LinearLayer layer2;
    MSELoss mse_loss;

    LogModel()
        : layer1(1, 8),
          tanh1(std::make_shared<TanhOp>()),
          layer2(8, 1) {}

    std::shared_ptr<Tensor> forward(
        const std::shared_ptr<Tensor>& x
    ) {
        auto z1 = layer1.forward(x);
        auto a1 = tanh1->forward({z1});
        auto y_pred = layer2.forward(a1);

        return y_pred;
    }

    void ClearGrad() {
        layer1.ClearGrad();
        layer2.ClearGrad();
    }

    void Update(float lr) {
        layer1.Update(lr);
        layer2.Update(lr);
    }
};

int main() {
    std::cout << "=== Log Approximation Test ===\n";
    
    // --- TRAIN DATA ---
    std::vector<float> train_x = {0.1f, 0.2f, 0.25f, 0.3f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 5.0f, 7.0f, 10.0f};
    std::vector<std::shared_ptr<Tensor>> x_train;
    std::vector<std::shared_ptr<Tensor>> y_train;
    
    for (float val : train_x) {
        x_train.push_back(std::make_shared<Tensor>(Tensor({1, 1}, val)));
        y_train.push_back(std::make_shared<Tensor>(Tensor({1, 1}, std::log(val))));
    }
    
    // --- TEST DATA (model hasn't seen these) ---
    std::vector<float> test_x = {0.2f, 0.8f, 1.2f, 2.5f, 4.0f, 6.0f, 8.0f, 12.0f};
    std::vector<std::shared_ptr<Tensor>> x_test;
    std::vector<std::shared_ptr<Tensor>> y_test;
    
    for (float val : test_x) {
        x_test.push_back(std::make_shared<Tensor>(Tensor({1, 1}, val)));
        y_test.push_back(std::make_shared<Tensor>(Tensor({1, 1}, std::log(val))));
    }
    
    // --- MODEL ---
    LogModel model;
    float learning_rate = 0.01f;
    int epochs = 2000;
    
    std::cout << "\nTraining...\n";
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        
        for (size_t i = 0; i < x_train.size(); ++i) {
            auto y_pred = model.forward(x_train[i]);
            auto loss = model.mse_loss.forward(y_pred, y_train[i]);
            total_loss += loss->at(0);
            
            Tensor grad_output({1, 1}, 1.0f);
            loss->backward(grad_output);
        }
        
        model.Update(learning_rate);
        model.ClearGrad();

        for (auto& x : x_train) {
            x->ClearGrad();
        }

        if (epoch % 200 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / x_train.size() << "\n";
        }
    }
    
    std::cout << "\n=== Train Predictions ===\n";
    for (size_t i = 0; i < x_train.size(); ++i) {
        auto y_pred = model.forward(x_train[i]);
        float pred = y_pred->at(0);
        float target = y_train[i]->at(0);
        float error = std::abs(pred - target);
        std::cout << "x = " << train_x[i] 
                  << " | pred = " << pred 
                  << " | target = " << target 
                  << " | error = " << error << "\n";
    }
    
    std::cout << "\n=== Test Predictions (unseen data) ===\n";
    float total_error = 0.0f;
    for (size_t i = 0; i < x_test.size(); ++i) {
        auto y_pred = model.forward(x_test[i]);
        float pred = y_pred->at(0);
        float target = y_test[i]->at(0);
        float error = std::abs(pred - target);
        total_error += error;
        std::cout << "x = " << test_x[i] 
                  << " | pred = " << pred 
                  << " | target = " << target 
                  << " | error = " << error << "\n";
    }
    std::cout << "\nAverage test error: " << total_error / x_test.size() << "\n";
    
    std::cout << "\n✅ Test completed!\n";
    return 0;
}