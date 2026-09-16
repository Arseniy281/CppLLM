#pragma once
#include <iostream>
#include <vector>
#include <functional>
#include <numeric>
#include <random>
#include <memory>
#include <fstream>
#include <unordered_set>
#include "../Autograd/operation.h"
#include "device.h"
#include "backend.h"

static const float EPSILON = 1e-8f;

struct UninitializedTag {};

class Tensor {
private:
    std::vector<size_t> shape_ = {};
    float* data_ = nullptr;
    size_t size_ = 0;
    size_t rank_ = 0;

    std::shared_ptr<Tensor> grad_ = nullptr;
    std::shared_ptr<Operation> grad_fn_ = nullptr;

    Device device_ = Device::CPU;

    friend class CPUBackend;
    friend class CUDABackend;

    size_t ComputeIndex(const std::vector<size_t>& indexes) const;
    static void CheckBeforeConcatenate(const std::vector<Tensor>& head_outputs,
        size_t axis = 0);
    void BuildBackwardGraph(std::vector<Tensor*>& graph, 
        std::unordered_set<Tensor*>& visited);

    void Allocate();
    void Free();
public:

    explicit Tensor(std::vector<size_t> shape);
    Tensor(std::vector<size_t> shape, float k);
    explicit Tensor(std::vector<size_t> shape, UninitializedTag);
    static Tensor Random(std::vector<size_t> shape, float min = 0.0f, 
        float max = 1.0f, Device device = Device::CPU);
    Tensor(std::vector<size_t> shape, std::vector<float> data, 
        Device device = Device::CPU);
    Tensor(std::vector<size_t> shape, float k, Device device);
    Tensor(std::vector<size_t> shape, Device device);
    Tensor() = default;

    ~Tensor();

    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    float& at(size_t index);
    const float& at(size_t index) const;
    float& at(const std::vector<size_t>& indexes);
    const float& at(const std::vector<size_t>& indexes) const;
    void Set(std::vector<size_t> indexes, float value);
    auto GetIter(const std::vector<size_t>& indexes);
    const auto GetIter(const std::vector<size_t>& indexes) const;

    static std::vector<size_t> IndexToCoord(size_t ind, 
        const std::vector<size_t>& shape);
    static size_t CoordToIndex(const std::vector<size_t>& coord, 
        const std::vector<size_t>& shape);

    Tensor operator+(const Tensor& other) const;
    Tensor& operator+=(const Tensor& other);
    Tensor operator-(const Tensor& other) const;
    Tensor& operator-=(const Tensor& other);
    Tensor operator*(const Tensor& other) const;
    Tensor& operator*=(const Tensor& other);
    Tensor operator/(const Tensor& other) const;
    Tensor& operator/=(const Tensor& other);

    Tensor operator+(const float num) const;
    Tensor& operator+=(const float num);
    Tensor operator-(const float num) const;
    Tensor& operator-=(const float num);
    Tensor operator*(const float num) const;
    Tensor& operator*=(const float num);
    Tensor operator/(const float num) const;
    Tensor& operator/=(const float num);
    Tensor operator-() const;

    Tensor MatMul(const Tensor& other) const;

    static Tensor Concatenate(const std::vector<Tensor>& head_outputs, 
        size_t axis = 0);
    static size_t BroadcastIndex(const std::vector<size_t>& real_shape, 
        const std::vector<size_t>& final_shape, size_t ind);
    static std::vector<size_t> AlignTensors(const std::vector<size_t>& small, 
        const std::vector<size_t>& big);
    static std::vector<size_t> GetFinalShape(const std::vector<size_t>& first, 
        const std::vector<size_t>& second);
    static size_t GetFinalSize(const std::vector<size_t>& final_shape);

    void SaveTensor(const std::string& name) const;
    static Tensor LoadTensor(const std::string& path, Device device = Device::CPU);

    Tensor Transpose() const;
    Tensor Reshape(std::vector<size_t> new_shape) const;
    size_t GetSize() const;
    size_t GetRank() const;
    const std::vector<size_t>& GetShape() const;
    float* Data();
    const float* Data() const;

    void print() const;
    void PrintInfo() const;

    std::shared_ptr<Tensor> Grad() const;
    void SetGrad(std::shared_ptr<Tensor> grad);
    void AddGrad(Tensor grad);
    void PrintGrad() const;
    void ClearGrad();
    std::shared_ptr<Operation> GradFn() const;
    void SetGradFn(std::shared_ptr<Operation> op);
    void backward(const Tensor& grad_output = Tensor({1, 1}, 1.0f));

    Tensor SumAxis(int axis) const;
    Tensor Mean(int axis) const;

    size_t Numel() const;
    Device GetDevice() const;

    void CopyToCUDA(Tensor& destination) const;
};

Tensor operator/(float scalar, const Tensor& t);