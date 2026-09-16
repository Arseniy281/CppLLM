#include "../Tensor/tensor.h"
#include "cuda_backend.h"
#include <cublas_v2.h>

CUDABackend::CUDABackend() {
    cublasCreate(&handle_);
}

CUDABackend::~CUDABackend() {
    cublasDestroy(handle_);
}

__global__ void AddKernel(const float* a, const float* b, float* result,
    size_t result_size, size_t rank, const size_t* shape_a, const size_t* shape_b,
    const size_t* final_shape) {

    size_t index = (size_t)(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= result_size) { return; }

    size_t index_a = 0;
    size_t index_b = 0;

    size_t final_stride = 1;
    size_t stride_a = 1;
    size_t stride_b = 1;

    for (int i = (int)(rank) - 1; i >= 0; i--) {
        size_t coord = (index / final_stride) % final_shape[i];
        size_t coord_a = (shape_a[i] == 1) ? 0 : coord;
        size_t coord_b = (shape_b[i] == 1) ? 0 : coord;

        index_a += coord_a * stride_a;
        index_b += coord_b * stride_b;

        final_stride *= final_shape[i];
        stride_a *= shape_a[i];
        stride_b *= shape_b[i];
    }

    result[index] = a[index_a] + b[index_b];
}


__global__ void SubKernel(const float* a, const float* b, float* result,
    size_t result_size, size_t rank, const size_t* shape_a,
    const size_t* shape_b, const size_t* final_shape) {

    size_t index = (size_t)(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= result_size) { return; }

    size_t index_a = 0;
    size_t index_b = 0;

    size_t final_stride = 1;
    size_t stride_a = 1;
    size_t stride_b = 1;

    for (int i = (int)(rank) - 1; i >= 0; i--) {
        size_t coord = (index / final_stride) % final_shape[i];
        size_t coord_a = (shape_a[i] == 1) ? 0 : coord;
        size_t coord_b = (shape_b[i] == 1) ? 0 : coord;

        index_a += coord_a * stride_a;
        index_b += coord_b * stride_b;

        final_stride *= final_shape[i];
        stride_a *= shape_a[i];
        stride_b *= shape_b[i];
    }

    result[index] = a[index_a] - b[index_b];
}


__global__ void MulKernel(const float* a, const float* b, float* result,
    size_t result_size, size_t rank, const size_t* shape_a,
    const size_t* shape_b, const size_t* final_shape) {

    size_t index = (size_t)(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= result_size) { return; }

    size_t index_a = 0;
    size_t index_b = 0;

    size_t final_stride = 1;
    size_t stride_a = 1;
    size_t stride_b = 1;

    for (int i = (int)(rank) - 1; i >= 0; i--) {
        size_t coord = (index / final_stride) % final_shape[i];
        size_t coord_a = (shape_a[i] == 1) ? 0 : coord;
        size_t coord_b = (shape_b[i] == 1) ? 0 : coord;

        index_a += coord_a * stride_a;
        index_b += coord_b * stride_b;

        final_stride *= final_shape[i];
        stride_a *= shape_a[i];
        stride_b *= shape_b[i];
    }

    result[index] = a[index_a] * b[index_b];
}


__global__ void DivKernel(const float* a, const float* b, float* result,
    size_t result_size, size_t rank, const size_t* shape_a,
    const size_t* shape_b, const size_t* final_shape) {

    size_t index = (size_t)(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= result_size) { return; }

    size_t index_a = 0;
    size_t index_b = 0;

    size_t final_stride = 1;
    size_t stride_a = 1;
    size_t stride_b = 1;

    for (int i = (int)(rank) - 1; i >= 0; i--) {
        size_t coord = (index / final_stride) % final_shape[i];
        size_t coord_a = (shape_a[i] == 1) ? 0 : coord;
        size_t coord_b = (shape_b[i] == 1) ? 0 : coord;

        index_a += coord_a * stride_a;
        index_b += coord_b * stride_b;

        final_stride *= final_shape[i];
        stride_a *= shape_a[i];
        stride_b *= shape_b[i];
    }

    float divisor = b[index_b];

    if (divisor == 0.0f) {
        result[index] = 0.0f;
    } else {
        result[index] = a[index_a] / divisor;
    }
}



Tensor CUDABackend::Add(const Tensor& a, const Tensor& b) const {
    if (a.GetDevice() != Device::CUDA || b.GetDevice() != Device::CUDA) {
        throw std::runtime_error("CUDA Add requires CUDA tensors");
    }

    std::vector<size_t> s1 = a.GetShape();
    std::vector<size_t> s2 = b.GetShape();

    s1 = Tensor::AlignTensors(s1, s2);
    s2 = Tensor::AlignTensors(s2, s1);

    std::vector<size_t> final_shape = Tensor::GetFinalShape(s1, s2);
    size_t final_size = Tensor::GetFinalSize(final_shape);

    Tensor result(final_shape, Device::CUDA);

    size_t* d_shape_a = nullptr;
    size_t* d_shape_b = nullptr;
    size_t* d_final_shape = nullptr;

    cudaMalloc(&d_shape_a, s1.size() * sizeof(size_t));
    cudaMalloc(&d_shape_b, s2.size() * sizeof(size_t));

    cudaMalloc(&d_final_shape, final_shape.size() * sizeof(size_t));
    cudaMemcpy(d_shape_a, s1.data(), 
        s1.size() * sizeof(size_t), cudaMemcpyHostToDevice);

    cudaMemcpy(d_shape_b, s2.data(),
        s2.size() * sizeof(size_t), cudaMemcpyHostToDevice);

    cudaMemcpy(d_final_shape, final_shape.data(),
        final_shape.size() * sizeof(size_t), cudaMemcpyHostToDevice);

    const size_t threads = 256;
    const size_t blocks = (final_size + threads - 1) / threads;

    AddKernel<<<blocks, threads>>>(a.Data(), b.Data(), result.Data(),
        final_size, final_shape.size(), d_shape_a, d_shape_b, d_final_shape);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        cudaFree(d_shape_a);
        cudaFree(d_shape_b);
        cudaFree(d_final_shape);

        throw std::runtime_error(cudaGetErrorString(error));
    }

    cudaDeviceSynchronize();

    cudaFree(d_shape_a);
    cudaFree(d_shape_b);
    cudaFree(d_final_shape);

    return result;
}

Tensor CUDABackend::Sub(const Tensor& a, const Tensor& b) const {
    if (a.GetDevice() != Device::CUDA || b.GetDevice() != Device::CUDA) {
        throw std::runtime_error("CUDA Sub requires CUDA tensors");
    }

    std::vector<size_t> s1 = a.GetShape();
    std::vector<size_t> s2 = b.GetShape();

    s1 = Tensor::AlignTensors(s1, s2);
    s2 = Tensor::AlignTensors(s2, s1);

    std::vector<size_t> final_shape = Tensor::GetFinalShape(s1, s2);
    size_t final_size = Tensor::GetFinalSize(final_shape);
    Tensor result(final_shape, Device::CUDA);

    size_t* d_shape_a = nullptr;
    size_t* d_shape_b = nullptr;
    size_t* d_final_shape = nullptr;

    cudaMalloc(&d_shape_a, s1.size() * sizeof(size_t));
    cudaMalloc(&d_shape_b, s2.size() * sizeof(size_t));
    cudaMalloc(&d_final_shape, final_shape.size() * sizeof(size_t));

    cudaMemcpy(d_shape_a, s1.data(),
        s1.size() * sizeof(size_t), cudaMemcpyHostToDevice);

    cudaMemcpy(d_shape_b, s2.data(),
        s2.size() * sizeof(size_t), cudaMemcpyHostToDevice);

    cudaMemcpy(d_final_shape, final_shape.data(),
        final_shape.size() * sizeof(size_t), cudaMemcpyHostToDevice);

    const size_t threads = 256;
    const size_t blocks = (final_size + threads - 1) / threads;

    SubKernel<<<blocks, threads>>>(a.Data(), b.Data(), result.Data(),
        final_size, final_shape.size(), d_shape_a, d_shape_b, d_final_shape);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        cudaFree(d_shape_a);
        cudaFree(d_shape_b);
        cudaFree(d_final_shape);

        throw std::runtime_error(cudaGetErrorString(error));
    }

    cudaDeviceSynchronize();

    cudaFree(d_shape_a);
    cudaFree(d_shape_b);
    cudaFree(d_final_shape);

    return result;
}

Tensor CUDABackend::Mul(const Tensor& a, const Tensor& b) const {
    if (a.GetDevice() != Device::CUDA || b.GetDevice() != Device::CUDA) {
        throw std::runtime_error("CUDA Mul requires CUDA tensors");
    }

    std::vector<size_t> s1 = a.GetShape();
    std::vector<size_t> s2 = b.GetShape();

    s1 = Tensor::AlignTensors(s1, s2);
    s2 = Tensor::AlignTensors(s2, s1);

    std::vector<size_t> final_shape = Tensor::GetFinalShape(s1, s2);
    size_t final_size = Tensor::GetFinalSize(final_shape);
    Tensor result(final_shape, Device::CUDA);

    size_t* d_shape_a = nullptr;
    size_t* d_shape_b = nullptr;
    size_t* d_final_shape = nullptr;

    cudaMalloc(&d_shape_a, s1.size() * sizeof(size_t));
    cudaMalloc( &d_shape_b, s2.size() * sizeof(size_t));
    cudaMalloc(&d_final_shape, final_shape.size() * sizeof(size_t));

    cudaMemcpy(d_shape_a, s1.data(),
        s1.size() * sizeof(size_t), cudaMemcpyHostToDevice);

    cudaMemcpy(d_shape_b, s2.data(),
        s2.size() * sizeof(size_t), cudaMemcpyHostToDevice);

    cudaMemcpy(d_final_shape, final_shape.data(),
        final_shape.size() * sizeof(size_t), cudaMemcpyHostToDevice);

    const size_t threads = 256;
    const size_t blocks = (final_size + threads - 1) / threads;

    MulKernel<<<blocks, threads>>>(a.Data(), b.Data(), result.Data(),
        final_size, final_shape.size(), d_shape_a, d_shape_b, d_final_shape);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        cudaFree(d_shape_a);
        cudaFree(d_shape_b);
        cudaFree(d_final_shape);

        throw std::runtime_error(cudaGetErrorString(error));
    }

    cudaDeviceSynchronize();

    cudaFree(d_shape_a);
    cudaFree(d_shape_b);
    cudaFree(d_final_shape);

    return result;
}

Tensor CUDABackend::Div(const Tensor& a, const Tensor& b) const {
    if (a.GetDevice() != Device::CUDA || b.GetDevice() != Device::CUDA) {
        throw std::runtime_error("CUDA Div requires CUDA tensors");
    }

    std::vector<size_t> s1 = a.GetShape();
    std::vector<size_t> s2 = b.GetShape();

    s1 = Tensor::AlignTensors(s1, s2);
    s2 = Tensor::AlignTensors(s2, s1);

    std::vector<size_t> final_shape = Tensor::GetFinalShape(s1, s2);
    size_t final_size = Tensor::GetFinalSize(final_shape);
    Tensor result(final_shape, Device::CUDA);

    size_t* d_shape_a = nullptr;
    size_t* d_shape_b = nullptr;
    size_t* d_final_shape = nullptr;

    cudaMalloc(&d_shape_a, s1.size() * sizeof(size_t));
    cudaMalloc(&d_shape_b, s2.size() * sizeof(size_t));
    cudaMalloc(&d_final_shape, final_shape.size() * sizeof(size_t));

    cudaMemcpy(d_shape_a, s1.data(),
        s1.size() * sizeof(size_t), cudaMemcpyHostToDevice);

    cudaMemcpy(d_shape_b, s2.data(),
        s2.size() * sizeof(size_t), cudaMemcpyHostToDevice);

    cudaMemcpy(d_final_shape, final_shape.data(),
        final_shape.size() * sizeof(size_t), cudaMemcpyHostToDevice);

    const size_t threads = 256;
    const size_t blocks = (final_size + threads - 1) / threads;

    DivKernel<<<blocks, threads>>>(a.Data(), b.Data(), result.Data(),
        final_size, final_shape.size(), d_shape_a, d_shape_b, d_final_shape);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        cudaFree(d_shape_a);
        cudaFree(d_shape_b);
        cudaFree(d_final_shape);

        throw std::runtime_error(cudaGetErrorString(error));
    }

    cudaDeviceSynchronize();

    cudaFree(d_shape_a);
    cudaFree(d_shape_b);
    cudaFree(d_final_shape);

    return result;
}

__global__ void SumKernel(const float* input, float* output, size_t size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float total = 0.0f;
        for (size_t i = 0; i < size; i++) {
            total += input[i];
        }

        output[0] = total;
    }
}

Tensor CUDABackend::Sum(const Tensor& input) const {
    Tensor result({1, 1}, Device::CUDA);

    SumKernel<<<1, 1>>>(
        input.Data(),
        result.Data(),
        input.GetSize()
    );

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    cudaDeviceSynchronize();
    return result;
}

__global__ void AddScalarKernel(const float* a, float scalar,
        float* result, size_t size) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        result[index] = a[index] + scalar;
    }
}

__global__ void SubScalarKernel(const float* a, float scalar, 
        float* result, size_t size) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        result[index] = a[index] - scalar;
    }
}

__global__ void MulScalarKernel(const float* a, float scalar, 
        float* result, size_t size) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        result[index] = a[index] * scalar;
    }
}

__global__ void DivScalarKernel(const float* a, float scalar, 
        float* result, size_t size) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        result[index] = a[index] / scalar;
    }
}

__global__ void NegKernel(const float* a, 
        float* result, size_t size) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        result[index] = -a[index];
    }
}

Tensor CUDABackend::AddScalar(const Tensor& a, float scalar) const {
    size_t size = a.GetSize();
    Tensor result(a.GetShape(), Device::CUDA);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    AddScalarKernel<<<blocks, threads>>>(
        a.Data(),
        scalar,
        result.Data(),
        size
    );

    cudaDeviceSynchronize();
    return result;
}

Tensor CUDABackend::SubScalar(const Tensor& a, float scalar) const {
    size_t size = a.GetSize();
    Tensor result(a.GetShape(), Device::CUDA);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    SubScalarKernel<<<blocks, threads>>>(
        a.Data(),
        scalar,
        result.Data(),
        size
    );

    cudaDeviceSynchronize();
    return result;
}
Tensor CUDABackend::MulScalar(const Tensor& a, float scalar) const {
    size_t size = a.GetSize();
    Tensor result(a.GetShape(), Device::CUDA);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    MulScalarKernel<<<blocks, threads>>>(
        a.Data(),
        scalar,
        result.Data(),
        size
    );

    cudaDeviceSynchronize();
    return result;
}
Tensor CUDABackend::DivScalar(const Tensor& a, float scalar) const {
    size_t size = a.GetSize();
    Tensor result(a.GetShape(), Device::CUDA);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    DivScalarKernel<<<blocks, threads>>>(
        a.Data(),
        scalar,
        result.Data(),
        size
    );

    cudaDeviceSynchronize();
    return result;
}

Tensor CUDABackend::Neg(const Tensor& a) const {
    size_t size = a.GetSize();
    Tensor result(a.GetShape(), Device::CUDA);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    NegKernel<<<blocks, threads>>>(
        a.Data(),
        result.Data(),
        size
    );

    cudaDeviceSynchronize();
    return result;
}

__global__ void TransposeKernel(const float* a, float* result, 
        size_t rows, size_t cols, size_t outer_size) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t matrix_size = rows * cols;
    size_t total_size = outer_size * matrix_size;

    if (index < total_size) {
        size_t outer = index / matrix_size;
        size_t local_index = index % matrix_size;

        size_t i = local_index / cols;
        size_t j = local_index % cols;

        size_t input_offset = outer * matrix_size;
        size_t output_offset = outer * matrix_size;

        result[output_offset + j * rows + i] = a[input_offset + i * cols + j];
    }
}

Tensor CUDABackend::Transpose(const Tensor& a) const {
    if (a.GetRank() < 2) {
        throw std::runtime_error("CUDABackend::Transpose: rank must be >= 2");
    }

    std::vector<size_t> new_shape = a.GetShape();
    std::swap(new_shape[new_shape.size() - 1], new_shape[new_shape.size() - 2]);

    Tensor result(new_shape, Device::CUDA);

    size_t rows = a.GetShape()[a.GetRank() - 2];
    size_t cols = a.GetShape()[a.GetRank() - 1];

    size_t outer_size = a.GetSize() / (rows * cols);
    size_t total_size = a.GetSize();

    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;

    TransposeKernel<<<blocks, threads>>>(
        a.Data(),
        result.Data(),
        rows,
        cols,
        outer_size
    );

    cudaDeviceSynchronize();
    return result;
}

__global__ void ConcatenateKernel(const float* input, float* output, size_t input_axis,
        size_t output_axis, size_t outer_size, size_t inner_size, size_t output_axis_offset) {

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t input_size = outer_size * input_axis * inner_size;

    if (index < input_size) {
        size_t inner_index = index % inner_size;
        size_t axis_index = (index / inner_size) % input_axis;
        size_t outer_index = index / (input_axis * inner_size);

        size_t output_axis_index = output_axis_offset + axis_index;
        size_t output_index = outer_index * output_axis * inner_size +
            output_axis_index * inner_size + inner_index;

        output[output_index] = input[index];
    }
}

Tensor CUDABackend::Concatenate(const std::vector<Tensor>& tensors, size_t axis) const {
    if (tensors.empty()) {
        throw std::runtime_error( "Tensors must have size > 0");
    }
    size_t rank = tensors[0].GetRank();
    if (axis >= rank) {
        throw std::runtime_error("Concatenate: axis out of bounds");
    }

    std::vector<size_t> final_shape = tensors[0].GetShape();

    for (size_t t = 1; t < tensors.size(); t++) {
        auto shape = tensors[t].GetShape();

        if (shape.size() != rank) {
            throw std::runtime_error("All tensors must have same rank");
        }

        for (size_t j = 0; j < rank; j++) {
            if (j != axis && final_shape[j] != shape[j]) {
                throw std::runtime_error(
                    "All tensors must have same shape except on axis");
            }
        }
    }

    size_t total_dim = 0;

    for (const auto& tensor : tensors) {
        total_dim += tensor.GetShape()[axis];
    }

    final_shape[axis] = total_dim;
    Tensor output(final_shape, Device::CUDA);

    size_t outer_size = 1;
    for (size_t i = 0; i < axis; i++) {
        outer_size *= final_shape[i];
    }

    size_t inner_size = 1;
    for (size_t i = axis + 1; i < rank; i++) {
        inner_size *= final_shape[i];
    }

    size_t offset = 0;

    for (const auto& tensor : tensors) {
        size_t input_axis = tensor.GetShape()[axis];
        size_t input_size = outer_size * input_axis * inner_size;

        int threads = 256;
        int blocks = (input_size + threads - 1) / threads;

        ConcatenateKernel<<<blocks, threads>>>(
            tensor.Data(),
            output.Data(),
            input_axis,
            total_dim,
            outer_size,
            inner_size,
            offset
        );

        offset += input_axis;
    }

    cudaDeviceSynchronize();
    return output;
}

__global__ void SplitKernel(const float* input, float* output, size_t outer, 
        size_t input_axis_size, size_t offset, size_t output_axis_size, size_t inner) {

    size_t index = (size_t)(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = outer * output_axis_size * inner;

    if (index >= total) { return; }

    size_t inner_index = index % inner;
    size_t temp = index / inner;
    size_t axis_index = temp % output_axis_size;
    size_t outer_index = temp / output_axis_size;

    size_t input_index = outer_index * input_axis_size * inner +
        (offset + axis_index) * inner + inner_index;

    output[index] = input[input_index];
}

std::vector<Tensor> CUDABackend::Split(const Tensor& tensor,
        const std::vector<std::vector<size_t>>& shapes, size_t axis) const {

    const auto& input_shape = tensor.GetShape();

    if (axis >= input_shape.size()) {
        throw std::runtime_error("CUDABackend::Split: axis out of bounds");
    }

    if (shapes.empty()) {
        throw std::runtime_error("CUDABackend::Split: no shapes");
    }

    size_t total_axis = 0;
    for (const auto& shape : shapes) {

        if (shape.size() != input_shape.size()) {
            throw std::runtime_error("CUDABackend::Split: rank mismatch");
        }

        for (size_t i = 0; i < input_shape.size(); i++) {
            if (i != axis && shape[i] != input_shape[i]) {
                throw std::runtime_error("CUDABackend::Split: shape mismatch");
            }
        }

        total_axis += shape[axis];
    }

    if (total_axis != input_shape[axis]) {
        throw std::runtime_error("CUDABackend::Split: split sizes do not match input");
    }

    const size_t outer = std::accumulate(input_shape.begin(),
            input_shape.begin() + axis, size_t{1}, std::multiplies<size_t>());

    const size_t inner = std::accumulate(input_shape.begin() + axis + 1,
            input_shape.end(), size_t{1}, std::multiplies<size_t>());

    std::vector<Tensor> result;
    result.reserve(shapes.size());
    size_t offset = 0;
    constexpr int threads = 256;

    for (const auto& shape : shapes) {
        const size_t axis_size = shape[axis];
        Tensor output(shape, Device::CUDA);

        const size_t total = outer * axis_size * inner;
        const int blocks = (int)((total + threads - 1) / threads);

        SplitKernel<<<blocks, threads>>>(
            tensor.Data(),
            output.Data(),
            outer,
            input_shape[axis],
            offset,
            axis_size,
            inner
        );

        cudaError_t error = cudaGetLastError();

        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string("CUDABackend::Split kernel failed: ") 
                + cudaGetErrorString(error));
        }

        cudaDeviceSynchronize();
        result.push_back(std::move(output));
        offset += axis_size;
    }

    return result;
}

__global__ void SumAxisKernel(const float* input, float* output, size_t outer_size, 
        size_t axis_size, size_t inner_size) {

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t output_size = outer_size * inner_size;

    if (index < output_size) {
        size_t outer_index = index / inner_size;
        size_t inner_index = index % inner_size;

        float sum = 0.0f;

        for (size_t axis_index = 0; axis_index < axis_size; axis_index++) {
            size_t input_index = outer_index * axis_size * inner_size +
                axis_index * inner_size + inner_index;

            sum += input[input_index];
        }

        output[index] = sum;
    }
}

Tensor CUDABackend::SumAxis(const Tensor& a, int axis) const {
    size_t rank = a.GetRank();

    if (axis < 0 || axis >= (int)(rank)) {
        throw std::runtime_error("CUDABackend::SumAxis: axis out of bounds");
    }

    std::vector<size_t> result_shape = a.GetShape();
    result_shape.erase(result_shape.begin() + axis);
    size_t outer_size = 1;

    for (size_t i = 0; i < (size_t)(axis); i++) {
        outer_size *= a.GetShape()[i];
    }

    size_t axis_size = a.GetShape()[axis];
    size_t inner_size = 1;

    for (size_t i = (size_t)(axis) + 1; i < rank; i++) {
        inner_size *= a.GetShape()[i];
    }

    Tensor result(result_shape, Device::CUDA);

    size_t output_size = outer_size * inner_size;

    int threads = 256;
    int blocks = (output_size + threads - 1) / threads;

    SumAxisKernel<<<blocks, threads>>>(
        a.Data(),
        result.Data(),
        outer_size,
        axis_size,
        inner_size
    );

    cudaDeviceSynchronize();
    return result;
}

__global__ void MeanKernel(const float* input, float* output, size_t outer_size,
        size_t axis_size, size_t inner_size) {

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t output_size = outer_size * inner_size;

    if (index < output_size) {
        size_t outer_index = index / inner_size;
        size_t inner_index = index % inner_size;
        float sum = 0.0f;

        for (size_t axis_index = 0; axis_index < axis_size; axis_index++) {
            size_t input_index = outer_index * axis_size * inner_size +
                axis_index * inner_size + inner_index;

            sum += input[input_index];
        }

        output[index] = sum / (float)(axis_size);
    }
}

Tensor CUDABackend::Mean(const Tensor& a, int axis) const {
    size_t rank = a.GetRank();
    if (axis < 0 || axis >= (int)(rank)) {
        throw std::runtime_error("CUDABackend::Mean: axis out of bounds");
    }

    std::vector<size_t> result_shape = a.GetShape();
    result_shape.erase(result_shape.begin() + axis);

    size_t outer_size = 1;
    for (size_t i = 0; i < (size_t)(axis); i++) {
        outer_size *= a.GetShape()[i];
    }

    size_t axis_size = a.GetShape()[axis];
    size_t inner_size = 1;

    for (size_t i = (size_t)(axis) + 1; i < rank; i++) {
        inner_size *= a.GetShape()[i];
    }

    Tensor result(result_shape, Device::CUDA);
    size_t output_size = outer_size * inner_size;

    int threads = 256;
    int blocks = (output_size + threads - 1) / threads;

    MeanKernel<<<blocks, threads>>>(
        a.Data(),
        result.Data(),
        outer_size,
        axis_size,
        inner_size
    );

    cudaDeviceSynchronize();
    return result;
}

Tensor CUDABackend::MatMul2D(const Tensor& A, const Tensor& B) const {
    const auto& shapeA = A.GetShape();
    const auto& shapeB = B.GetShape();

    if (shapeA.size() != 2 || shapeB.size() != 2) {
        throw std::runtime_error("MatMul2D: tensors must be 2D");
    }

    size_t M = shapeA[0];
    size_t K = shapeA[1];
    size_t K2 = shapeB[0];
    size_t N = shapeB[1];

    if (K != K2) {
        throw std::runtime_error("MatMul2D: incompatible matrix dimensions");
    }

    Tensor result({M, N}, Device::CUDA);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasStatus_t status = cublasSgemm(
        handle_,
        CUBLAS_OP_N,
        CUBLAS_OP_N,

        (int)(N),
        (int)(M),
        (int)(K),

        &alpha,
        B.Data(),
        (int)(N),

        A.Data(),
        (int)(K),

        &beta,
        result.Data(),
        (int)(N)
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("CUDABackend::MatMul2D: cublasSgemm failed");
    }

    cudaDeviceSynchronize();
    return result;
}

Tensor CUDABackend::MatMul3D2D(const Tensor& A, const Tensor& B) const {
    const auto& shapeA = A.GetShape();
    const auto& shapeB = B.GetShape();

    if (shapeA.size() != 3 || shapeB.size() != 2) {
        throw std::runtime_error("MatMul3D2D: invalid ranks");
    }

    size_t batch = shapeA[0];
    size_t M = shapeA[1];
    size_t K = shapeA[2];

    size_t K2 = shapeB[0];
    size_t N = shapeB[1];

    if (K != K2) {
        throw std::runtime_error("MatMul3D2D: incompatible matrix dimensions");
    }

    Tensor result({batch, M, N}, Device::CUDA);

    size_t strideA = M * K;
    size_t strideB = 0;
    size_t strideC = M * N;

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasStatus_t status = cublasSgemmStridedBatched(
        handle_,
        CUBLAS_OP_N,
        CUBLAS_OP_N,

        (int)(N),
        (int)(M),
        (int)(K),

        &alpha,

        B.Data(),
        (int)(N),
        (long long)(strideB),

        A.Data(),
        (int)(K),
        (long long)(strideA),

        &beta,
        result.Data(),

        (int)(N),
        (long long)(strideC),
        (int)(batch)
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            "CUDABackend::MatMul3D2D: cublasSgemmStridedBatched failed");
    }

    cudaDeviceSynchronize();
    return result;
}

Tensor CUDABackend::MatMulBatched(const Tensor& A, const Tensor& B) const {
    const auto& shapeA = A.GetShape();
    const auto& shapeB = B.GetShape();

    if (shapeA.size() != 3 || shapeB.size() != 3) {
        throw std::runtime_error("MatMulBatched: invalid ranks");
    }

    size_t batch = shapeA[0];
    size_t M = shapeA[1];
    size_t K = shapeA[2];

    size_t batch2 = shapeB[0];
    size_t K2 = shapeB[1];
    size_t N = shapeB[2];

    if (batch != batch2) {
        throw std::runtime_error("MatMulBatched: incompatible batch dimensions");
    }
    if (K != K2) {
        throw std::runtime_error("MatMulBatched: incompatible matrix dimensions" );
    }

    Tensor result({batch, M, N}, Device::CUDA);

    size_t strideA = M * K;
    size_t strideB = K * N;
    size_t strideC = M * N;

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasStatus_t status = cublasSgemmStridedBatched(
        handle_,
        CUBLAS_OP_N,
        CUBLAS_OP_N,

        (int)(N),
        (int)(M),
        (int)(K),

        &alpha,

        B.Data(),
        (int)(N),
        (long long)(strideB),

        A.Data(),
        (int)(K),
        (long long)(strideA),

        &beta,
        result.Data(),
        (int)(N),
        (long long)(strideC),
        (int)(batch)
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            "CUDABackend::MatMulBatched: cublasSgemmStridedBatched failed");
    }

    cudaDeviceSynchronize();
    return result;
}

Tensor CUDABackend::MatMul(const Tensor& a, const Tensor& b) const {
    const auto& shapeA = a.GetShape();
    const auto& shapeB = b.GetShape();

    if (shapeA.size() < 2 || shapeB.size() < 2) {
        throw std::runtime_error("MatMul: tensors must have rank >= 2");
    }

    if (shapeA.size() == 2 && shapeB.size() == 2) {
        return MatMul2D(a, b);
    }

    if (shapeA.size() == 3 && shapeB.size() == 2) {
        return MatMul3D2D(a, b);
    }

    if (shapeA.size() == 3 && shapeB.size() == 3) {
        return MatMulBatched(a, b);
    }

    throw std::runtime_error("MatMul: unsupported tensor ranks");
}
__global__ void ScalarDivKernel(float scalar, const float* input, 
        float* output, size_t size) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        float divisor = input[index];
        output[index] = (divisor == 0.0f) ? 0.0f : scalar / divisor;
    }
}

Tensor CUDABackend::ScalarDiv(float scalar, const Tensor& t) const {
    Tensor result(t.GetShape(), Device::CUDA);
    size_t size = t.GetSize();

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    ScalarDivKernel<<<blocks, threads>>>(
        scalar,
        t.Data(),
        result.Data(),
        size
    );

    cudaDeviceSynchronize();
    return result;
}

__global__ void EmbeddingKernel(const float* embeddings, const float* indices,
    float* output, size_t indices_count, size_t embedding_dim, size_t vocab_size) {

    size_t index = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
    size_t total = indices_count * embedding_dim;
    if (index >= total) { return; }

    size_t token = index / embedding_dim;
    size_t feature = index % embedding_dim;
    size_t embedding_index = (size_t)(indices[token]);

    if (embedding_index >= vocab_size) { return; }
    output[index] = embeddings[embedding_index * embedding_dim + feature];
}

Tensor CUDABackend::Embedding(const Tensor& embeddings,
        const Tensor& indices) const {
    const auto& embedding_shape = embeddings.GetShape();

    if (embedding_shape.size() != 2) {
        throw std::runtime_error(
            "CUDABackend::Embedding: embeddings must be 2D");
    }

    if (embeddings.GetDevice() != Device::CUDA 
        || indices.GetDevice() != Device::CUDA) {
        throw std::runtime_error(
            "CUDABackend::Embedding: tensors must be CUDA");
    }

    const size_t vocab_size = embedding_shape[0];
    const size_t embedding_dim = embedding_shape[1];

    std::vector<size_t> result_shape = indices.GetShape();
    result_shape.push_back(embedding_dim);
    Tensor result(result_shape, Device::CUDA);

    const size_t total = indices.GetSize() * embedding_dim;
    constexpr int threads = 256;
    const int blocks = (int)((total + threads - 1) / threads);

    EmbeddingKernel<<<blocks, threads>>>(
        embeddings.Data(),
        indices.Data(),
        result.Data(),
        indices.GetSize(),
        embedding_dim,
        vocab_size
    );

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDABackend::Embedding kernel failed: ") 
            + cudaGetErrorString(error)
        );
    }

    cudaDeviceSynchronize();
    return result;
}

__global__ void EmbeddingBackwardKernel(const float* indices, 
        const float* grad_output, float* grad_embeddings, 
        size_t indices_count, size_t embedding_dim, size_t vocab_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= indices_count) { return; }

    size_t embedding_index = (size_t)(indices[idx]);
    if (embedding_index >= vocab_size) { return; }

    for (size_t j = 0; j < embedding_dim; j++) {
        atomicAdd(
            &grad_embeddings[embedding_index * embedding_dim + j],
            grad_output[idx * embedding_dim + j]
        );
    }
}

Tensor CUDABackend::EmbeddingBackward(const Tensor& embeddings,
        const Tensor& indices, const Tensor& grad_output) const {

    const auto& embedding_shape = embeddings.GetShape();

    if (embedding_shape.size() != 2) {
        throw std::runtime_error(
            "CUDABackend::EmbeddingBackward: embeddings must be 2D"
        );
    }

    const size_t vocab_size = embedding_shape[0];
    const size_t embedding_dim = embedding_shape[1];

    std::vector<size_t> expected_shape = indices.GetShape();
    expected_shape.push_back(embedding_dim);

    if (grad_output.GetShape() != expected_shape) {
        throw std::runtime_error(
            "CUDABackend::EmbeddingBackward: grad_output shape mismatch"
        );
    }

    Tensor grad({vocab_size, embedding_dim}, Device::CUDA);

    cudaMemset(grad.Data(), 0, grad.GetSize() * sizeof(float));

    const size_t indices_count = indices.GetSize();
    constexpr size_t threads = 256;
    const size_t blocks = (indices_count + threads - 1) / threads;

    EmbeddingBackwardKernel<<<blocks, threads>>>(
        indices.Data(),
        grad_output.Data(),
        grad.Data(),
        indices_count,
        embedding_dim,
        vocab_size
    );

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("EmbeddingBackward kernel failed: ") +
            cudaGetErrorString(error)
        );
    }

    return grad;
}

__global__ void GeluKernel(const float* input, float* output, size_t size) {
    size_t i = (size_t)( blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= size) { return; }
    const float pi = 3.14159265358979323846f;
    const float sqrt_2_pi = sqrtf(2.0f / pi);
    const float a =  0.044715f;
    const float x = input[i];
    const float x3 = x * x * x;
    output[i] = 0.5f * x * ( 1.0f + tanhf(sqrt_2_pi * (x + a * x3)));
}

__global__ void GeluBackwardKernel(const float* input, 
        const float* grad_output, float* grad_input, size_t size) {

    size_t i = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= size) { return; }

    const float pi = 3.14159265358979323846f;
    const float sqrt_2_pi = sqrtf(2.0f / pi);
    const float a = 0.044715f;
    const float x = input[i];
    const float x2 = x * x;
    const float x3 = x2 * x;
    const float arg = sqrt_2_pi * (x + a * x3);
    const float tanh_val = tanhf(arg);
    const float sech_sq = 1.0f - tanh_val * tanh_val;
    const float derivative = 0.5f * (1.0f + tanh_val) + 0.5f
             * x * sech_sq * sqrt_2_pi * (1.0f + 3.0f * a * x2);

    grad_input[i] = grad_output[i] * derivative;
}

Tensor CUDABackend::Gelu(const Tensor& a) const {
    Tensor result(a.GetShape(), Device::CUDA);

    constexpr int threads = 256;
    const int blocks = (int)((a.GetSize() + threads - 1) / threads);

    GeluKernel<<<blocks, threads>>>(a.Data(), result.Data(), a.GetSize());
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(
            "CUDABackend::Gelu kernel failed: ") 
            + cudaGetErrorString(error));
    }

    cudaDeviceSynchronize();
    return result;
}

Tensor CUDABackend::GeluBackward(const Tensor& input, const Tensor& grad_output) const {
    if (input.GetShape() != grad_output.GetShape()) {
        throw std::runtime_error("CUDABackend::GeluBackward: shape mismatch");
    }

    Tensor result(input.GetShape(), Device::CUDA);

    constexpr int threads = 256;
    const int blocks = (int)((input.GetSize() + threads - 1) / threads);

    GeluBackwardKernel<<<blocks, threads>>>(
        input.Data(),
        grad_output.Data(),
        result.Data(),
        input.GetSize()
    );

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(
            "CUDABackend::GeluBackward kernel failed: ") 
            + cudaGetErrorString(error)
        );
    }

    cudaDeviceSynchronize();
    return result;
}

__global__ void ReLUKernel(const float* input, float* output, size_t size) {
    size_t i = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= size) { return; }
    output[i] = fmaxf(0.0f, input[i]);
}

__global__ void ReLUBackwardKernel(const float* input, const float* grad_output,
         float* grad_input, size_t size) {
    size_t i = (size_t)( blockIdx.x * blockDim.x + threadIdx.x );

    if (i >= size) { return; }
    grad_input[i] = input[i] > 0.0f ? grad_output[i] : 0.0f;
}

Tensor CUDABackend::ReLU(const Tensor& a) const {
    Tensor result(a.GetShape(), Device::CUDA);
    constexpr int threads = 256;
    const int blocks = (int)((a.GetSize() + threads - 1) / threads);

    ReLUKernel<<<blocks, threads>>>(
        a.Data(),
        result.Data(),
        a.GetSize()
    );

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(
             "CUDABackend::ReLU kernel failed: ") 
             + cudaGetErrorString(error)
        );
    }

    cudaDeviceSynchronize();
    return result;
}

Tensor CUDABackend::ReLUBackward(const Tensor& input, const Tensor& grad_output) const {
    if (input.GetShape() != grad_output.GetShape()) {
        throw std::runtime_error("CUDABackend::ReLUBackward: shape mismatch");
    }

    Tensor result(input.GetShape(), Device::CUDA);
    constexpr int threads = 256;
    const int blocks = (int)((input.GetSize() + threads - 1) / threads);

    ReLUBackwardKernel<<<blocks, threads>>>(
        input.Data(),
        grad_output.Data(),
        result.Data(),
        input.GetSize()
    );

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(
            "CUDABackend::ReLUBackward kernel failed: ") 
            + cudaGetErrorString(error)
        );
    }

    cudaDeviceSynchronize();
    return result;
}

__global__ void RMSNormForwardKernel(const float* input, const float* gamma,
    float* output, float* x_norm, float* rms, size_t batch, size_t seq_len,
    size_t embed_dim) {

    size_t row = (size_t)(blockIdx.x);
    if (row >= batch * seq_len) { return; }

    const size_t offset = row * embed_dim;
    float mean_square = 0.0f;

    for (size_t i = 0; i < embed_dim; i++) {
        float x = input[offset + i];
        mean_square += x * x;
    }

    mean_square /= (float)(embed_dim);
    float current_rms = sqrtf(mean_square + 1e-6f);
    rms[row] = current_rms;

    for (size_t i = 0; i < embed_dim; i++) {
        float x = input[offset + i];
        float normalized = x / current_rms;

        x_norm[offset + i] = normalized;
        output[offset + i] = normalized * gamma[i];
    }
}

__global__ void RMSNormBackwardKernel(const float* input, const float* gamma,
    const float* x_norm, const float* rms, const float* grad_output,
    float* grad_x, float* grad_gamma, size_t batch, size_t seq_len,
    size_t embed_dim) {
    size_t row = (size_t)(blockIdx.x);

    if (row >= batch * seq_len) { return; }
    const size_t offset = row * embed_dim;
    const float current_rms = rms[row];

    float mean_term = 0.0f;

    for (size_t i = 0; i < embed_dim; ++i) {
        const float grad_y = grad_output[offset + i];
        const float current_gamma = gamma[i];
        const float normalized = x_norm[offset + i];

        mean_term += grad_y * current_gamma * normalized;
    }

    mean_term /= (float)(embed_dim);

    for (size_t i = 0; i < embed_dim; ++i) {
        const float grad_y = grad_output[offset + i];
        const float current_gamma = gamma[i];
        const float normalized = x_norm[offset + i];

        const float grad_normalized = grad_y * current_gamma;

        grad_x[offset + i] = 
            (grad_normalized - normalized * mean_term) / current_rms;

        atomicAdd(&grad_gamma[i], grad_y * normalized);
    }
}

std::vector<Tensor> CUDABackend::RMSNormForward(const Tensor& input, 
        const Tensor& gamma) const {

    const std::vector<size_t>& input_shape = input.GetShape();

    if (input_shape.size() != 3) {
        throw std::runtime_error(
            "CUDABackend::RMSNormForward: "
            "input must be 3D"
        );
    }

    if (input.GetDevice() != Device::CUDA) {
        throw std::runtime_error(
            "CUDABackend::RMSNormForward: "
            "input must be CUDA tensor"
        );
    }

    if (gamma.GetDevice() != Device::CUDA) {
        throw std::runtime_error(
            "CUDABackend::RMSNormForward: "
            "gamma must be CUDA tensor"
        );
    }

    const size_t batch = input_shape[0];
    const size_t seq_len = input_shape[1];
    const size_t embed_dim = input_shape[2];

    const std::vector<size_t>& gamma_shape = gamma.GetShape();

    if (gamma_shape.size() != 1 || gamma_shape[0] != embed_dim) {
        throw std::runtime_error(
            "CUDABackend::RMSNormForward: "
            "invalid gamma shape"
        );
    }

    if (input.GetSize() != batch * seq_len * embed_dim) {
        throw std::runtime_error(
            "CUDABackend::RMSNormForward: "
            "invalid input size"
        );
    }

    if (gamma.GetSize() != embed_dim) {
        throw std::runtime_error(
            "CUDABackend::RMSNormForward: "
            "invalid gamma size"
        );
    }

    Tensor output(input_shape, Device::CUDA);
    Tensor x_norm(input_shape, Device::CUDA);
    Tensor rms({batch, seq_len}, Device::CUDA);

    const size_t rows = batch * seq_len;
    const size_t threads = 1;
    const size_t blocks = rows;

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(
                "CUDABackend::RMSNormForward: "
                "CUDA already in error state: "
            ) + cudaGetErrorString(error)
        );
    }

    RMSNormForwardKernel<<<blocks, threads>>>(
        input.Data(),
        gamma.Data(),
        output.Data(),
        x_norm.Data(),
        rms.Data(),
        batch,
        seq_len,
        embed_dim
    );

    error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(
                "CUDABackend::RMSNormForward: "
                "kernel launch failed: "
            ) + cudaGetErrorString(error)
        );
    }

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(
                "CUDABackend::RMSNormForward: "
                "kernel execution failed: "
            ) + cudaGetErrorString(error)
        );
    }

    return { std::move(output), std::move(x_norm), std::move(rms) };
}

std::vector<Tensor> CUDABackend::RMSNormBackward(const Tensor& input,
    const Tensor& gamma, const Tensor& x_norm, const Tensor& rms,
    const Tensor& grad_output) const {

    const std::vector<size_t>& input_shape = input.GetShape();

    if (input_shape.size() != 3) {
        throw std::runtime_error(
            "CUDABackend::RMSNormBackward: "
            "input must be 3D"
        );
    }

    if (input.GetDevice() != Device::CUDA || gamma.GetDevice() != Device::CUDA 
        || x_norm.GetDevice() != Device::CUDA || rms.GetDevice() != Device::CUDA
        || grad_output.GetDevice() != Device::CUDA) {

        throw std::runtime_error(
            "CUDABackend::RMSNormBackward: "
            "all tensors must be CUDA tensors"
        );
    }

    const size_t batch = input_shape[0];
    const size_t seq_len = input_shape[1];
    const size_t embed_dim = input_shape[2];

    if (x_norm.GetShape() != input_shape) {
        throw std::runtime_error(
            "CUDABackend::RMSNormBackward: "
            "invalid x_norm shape"
        );
    }

    if (rms.GetShape() != std::vector<size_t>{batch, seq_len}) {
        throw std::runtime_error(
            "CUDABackend::RMSNormBackward: "
            "invalid rms shape"
        );
    }

    if (grad_output.GetShape() != input_shape) {
        throw std::runtime_error(
            "CUDABackend::RMSNormBackward: "
            "invalid grad_output shape"
        );
    }

    if (gamma.GetShape() != std::vector<size_t>{embed_dim}) {
        throw std::runtime_error(
            "CUDABackend::RMSNormBackward: "
            "invalid gamma shape"
        );
    }

    Tensor grad_x(input_shape, Device::CUDA);
    Tensor grad_gamma({embed_dim}, 0.0f, Device::CUDA);

    const size_t rows = batch * seq_len;
    const size_t threads = 1;
    const size_t blocks = rows;

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(
                "CUDABackend::RMSNormBackward: "
                "CUDA already in error state: "
            ) + cudaGetErrorString(error)
        );
    }

    RMSNormBackwardKernel<<<blocks, threads>>>(
        input.Data(),
        gamma.Data(),
        x_norm.Data(),
        rms.Data(),
        grad_output.Data(),
        grad_x.Data(),
        grad_gamma.Data(),
        batch,
        seq_len,
        embed_dim
    );

    error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(
                "CUDABackend::RMSNormBackward: "
                "kernel launch failed: "
            ) + cudaGetErrorString(error)
        );
    }

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(
                "CUDABackend::RMSNormBackward: "
                "kernel execution failed: "
            ) + cudaGetErrorString(error)
        );
    }

    return { std::move(grad_x), std::move(grad_gamma) };
}

__global__ void RoPEForwardKernel(const float* input, float* output, size_t batch,
        size_t seq_len, size_t head_dim, size_t start_pos) {

    size_t index = (size_t)(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = batch * seq_len * head_dim;
    if (index >= total) { return; }

    const size_t dim = index % head_dim;
    if (dim % 2 != 0) { return; }
    const size_t pos = (index / head_dim) % seq_len;
    const size_t actual_pos = start_pos + pos;
    const float theta = (float)(actual_pos) 
        * powf(10000.0f, -(float)(dim) / head_dim);

    const float cos_theta = cosf(theta);
    const float sin_theta = sinf(theta);
    const float x = input[index];
    const float y = input[index + 1];
    output[index] = x * cos_theta - y * sin_theta;
    output[index + 1] = x * sin_theta + y * cos_theta;
}

__global__ void RoPEBackwardKernel(const float* input, float* output, size_t batch,
        size_t seq_len, size_t head_dim, size_t start_pos) {

    size_t index = (size_t)(blockIdx.x) * blockDim.x + threadIdx.x;

    const size_t total =  batch * seq_len * head_dim;
    if (index >= total) { return; }
    const size_t dim = index % head_dim;

    if (dim % 2 != 0) { return; }
    const size_t pos = (index / head_dim) % seq_len;
    const size_t actual_pos = start_pos + pos;
    const float theta = (float)(actual_pos) 
        * powf( 10000.0f, -(float)(dim) / head_dim);

    const float cos_theta = cosf(theta);
    const float sin_theta = sinf(theta);
    const float x = input[index];
    const float y = input[index + 1];
    output[index] = x * cos_theta + y * sin_theta;
    output[index + 1] = -x * sin_theta + y * cos_theta;
}

Tensor CUDABackend::RoPE(const Tensor& tensor, size_t start_pos) const {
    const std::vector<size_t>& shape = tensor.GetShape();

    if (shape.size() != 3) {
        throw std::runtime_error("CUDABackend::RoPE: tensor must be 3D");
    }

    const size_t batch = shape[0];
    const size_t seq_len = shape[1];
    const size_t head_dim = shape[2];

    if (head_dim % 2 != 0) {
        throw std::runtime_error("CUDABackend::RoPE: head_dim must be even");
    }

    Tensor result(shape, Device::CUDA);

    const size_t total = batch * seq_len * head_dim;
    const size_t threads = 256;
    const size_t blocks = (total + threads - 1) / threads;

    RoPEForwardKernel<<<blocks, threads>>>(
        tensor.Data(),
        result.Data(),
        batch,
        seq_len,
        head_dim,
        start_pos
    );

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(
            "CUDABackend::RoPE: kernel launch failed: ") 
            + cudaGetErrorString(error));
    }

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(
            "CUDABackend::RoPE: kernel execution failed: ") 
            + cudaGetErrorString(error));
    }

    return result;
}

Tensor CUDABackend::RoPEBackward(const Tensor& tensor, size_t start_pos) const {
    const std::vector<size_t>& shape = tensor.GetShape();
    if (shape.size() != 3) {
        throw std::runtime_error(
            "CUDABackend::RoPEBackward: tensor must be 3D");
    }

    const size_t batch = shape[0];
    const size_t seq_len = shape[1];
    const size_t head_dim = shape[2];

    if (head_dim % 2 != 0) {
        throw std::runtime_error(
            "CUDABackend::RoPEBackward: head_dim must be even");
    }

    Tensor result(shape, Device::CUDA);

    const size_t total = batch * seq_len * head_dim;
    const size_t threads = 256;
    const size_t blocks = (total + threads - 1) / threads;

    RoPEBackwardKernel<<<blocks, threads>>>(
        tensor.Data(),
        result.Data(),
        batch,
        seq_len,
        head_dim,
        start_pos
    );

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(
            "CUDABackend::RoPEBackward: kernel launch failed: ") 
            + cudaGetErrorString(error)
        );
    }

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(
            "CUDABackend::RoPEBackward: kernel execution failed: ") 
            + cudaGetErrorString(error));
    }

    return result;
}

__global__ void SigmoidForwardKernel(const float* input, float* output, size_t size) {
    size_t index = (size_t)(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size) { return; }

    const float x = input[index];
    output[index] = 1.0f / (1.0f + expf(-x));
}

__global__ void SigmoidBackwardKernel(const float* input, 
        const float* grad_output, float* grad_input, size_t size) {

    size_t index = (size_t)(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size) { return; }

    const float x = input[index];
    const float s = 1.0f / (1.0f + expf(-x));
    grad_input[index] = grad_output[index] * s * (1.0f - s);
}

Tensor CUDABackend::Sigmoid(const Tensor& input) const {
    Tensor result(input.GetShape(), Device::CUDA);

    const size_t size = input.GetSize();
    const size_t threads = 256;
    const size_t blocks = (size + threads - 1) / threads;

    SigmoidForwardKernel<<<blocks, threads>>>(input.Data(), result.Data(), size);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(
            "CUDABackend::Sigmoid: kernel launch failed: ") 
            + cudaGetErrorString(error));
    }

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(
            "CUDABackend::Sigmoid: kernel execution failed: ") 
            + cudaGetErrorString(error));
    }

    return result;
}

Tensor CUDABackend::SigmoidBackward(const Tensor& input, 
        const Tensor& grad_output) const {
    if (input.GetShape() != grad_output.GetShape()) {
        throw std::runtime_error(
            "CUDABackend::SigmoidBackward: shape mismatch");
    }

    Tensor result(input.GetShape(), Device::CUDA);

    const size_t size = input.GetSize();
    const size_t threads = 256;
    const size_t blocks = (size + threads - 1) / threads;

    SigmoidBackwardKernel<<<blocks, threads>>>(
        input.Data(),
        grad_output.Data(),
        result.Data(),
        size
    );

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(
            "CUDABackend::SigmoidBackward: kernel launch failed: ")
            + cudaGetErrorString(error));
    }

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(
            "CUDABackend::SigmoidBackward: kernel execution failed: ") 
            + cudaGetErrorString(error));
    }

    return result;
}

__global__ void SoftmaxForwardKernel(const float* input, float* output,
        size_t last_dim, size_t total_vectors) {

    const size_t vector_index = (size_t)(blockIdx.x);
    if (vector_index >= total_vectors) { return; }

    const size_t base = vector_index * last_dim;
    float max_value = -INFINITY;
    for (size_t i = 0; i < last_dim; i++) {
        max_value = fmaxf(max_value, input[base + i]);
    }

    float sum = 0.0f;

    for (size_t i = 0; i < last_dim; i++) {
        const float value = expf(input[base + i] - max_value);

        output[base + i] = value;
        sum += value;
    }

    for (size_t i = 0; i < last_dim; i++) {
        output[base + i] /= sum;
    }
}

__global__ void SoftmaxBackwardKernel(const float* output, const float* grad_output,
        float* grad_input, size_t last_dim, size_t total_vectors) {

    const size_t vector_index = (size_t)(blockIdx.x);
    if (vector_index >= total_vectors) { return; }

    const size_t base = vector_index * last_dim;
    float sum = 0.0f;

    for (size_t i = 0; i < last_dim; i++) {
        sum += grad_output[base + i] * output[base + i];
    }

    for (size_t i = 0; i < last_dim; i++) {
        grad_input[base + i] = output[base + i] * (grad_output[base + i] - sum);
    }
}

Tensor CUDABackend::Softmax(const Tensor& input) const {
    const std::vector<size_t>& shape = input.GetShape();

    if (shape.empty()) {
        throw std::runtime_error(
            "CUDABackend::Softmax: input must have at least one dimension");
    }

    const size_t last_dim = shape.back();
    size_t total_vectors = 1;

    for (size_t i = 0; i + 1 < shape.size(); i++) {
        total_vectors *= shape[i];
    }

    Tensor result(shape, Device::CUDA);
    const size_t blocks = total_vectors;
    const size_t threads = 1;

    SoftmaxForwardKernel<<<blocks, threads>>>(
        input.Data(),
        result.Data(),
        last_dim,
        total_vectors
    );

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(
            "CUDABackend::Softmax: kernel launch failed: ") 
            + cudaGetErrorString(error));
    }

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(
            "CUDABackend::Softmax: kernel execution failed: ") 
            + cudaGetErrorString(error));
    }

    return result;
}

Tensor CUDABackend::SoftmaxBackward(const Tensor& output, 
        const Tensor& grad_output) const {
    const std::vector<size_t>& shape = output.GetShape();

    if (shape.empty()) {
        throw std::runtime_error(
            "CUDABackend::SoftmaxBackward: output must have at least one dimension");
    }

    if (grad_output.GetShape() != shape) {
        throw std::runtime_error(
            "CUDABackend::SoftmaxBackward: shape mismatch");
    }

    const size_t last_dim = shape.back();
    const size_t total_size = output.GetSize();
    const size_t total_vectors = total_size / last_dim;

    Tensor grad_input(shape, Device::CUDA);
    const size_t blocks = total_vectors;
    const size_t threads = 1;

    SoftmaxBackwardKernel<<<blocks, threads>>>(
        output.Data(),
        grad_output.Data(),
        grad_input.Data(),
        last_dim,
        total_vectors
    );

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(
            "CUDABackend::SoftmaxBackward: kernel launch failed: ") 
            + cudaGetErrorString(error));
    }

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(
            "CUDABackend::SoftmaxBackward: kernel execution failed: ") 
            + cudaGetErrorString(error)
        );
    }

    return grad_input;
}

__global__ void ReduceBroadcastKernel(const float* grad, float* result, size_t grad_size,
        size_t grad_rank, size_t target_rank, const size_t* grad_shape, const size_t* target_shape) {

    size_t index = (size_t)(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= grad_size) { return; }

    size_t temp = index;
    size_t grad_coord[8];

    for (int axis = (int)(grad_rank) - 1; axis >= 0; axis--) {
        grad_coord[axis] = temp % grad_shape[axis];
        temp /= grad_shape[axis];
    }

    size_t target_index = 0;
    size_t target_stride = 1;

    for (int axis = (int)(target_rank) - 1; axis >= 0; axis--) {
        const size_t grad_axis = grad_rank - target_rank + axis;
        size_t coord;

        if (target_shape[axis] == 1) {
            coord = 0;
        } else {
            coord = grad_coord[grad_axis];
        }

        target_index += coord * target_stride;
        target_stride *= target_shape[axis];
    }

    atomicAdd(&result[target_index], grad[index]);
}

Tensor CUDABackend::ReduceBroadcast(const Tensor& grad, 
        const std::vector<size_t>& target_shape) const {

    const std::vector<size_t>& grad_shape = grad.GetShape();

    if (grad.GetDevice() != Device::CUDA) {
        throw std::runtime_error("ReduceBroadcast: grad must be CUDA tensor");
    }

    if (target_shape.size() > grad_shape.size()) {
        throw std::runtime_error(
            "ReduceBroadcast: target rank cannot be greater than grad rank"
        );
    }

    const size_t grad_rank = grad_shape.size();
    const size_t target_rank = target_shape.size();

    if (grad_rank > 8 || target_rank > 8) {
        throw std::runtime_error(
            "ReduceBroadcast: rank greater than 8 is not supported"
        );
    }

    for (size_t i = 0; i < target_rank; ++i) {
        const size_t grad_axis = grad_rank - target_rank + i;
        const size_t grad_dim = grad_shape[grad_axis];
        const size_t target_dim = target_shape[i];

        if (target_dim != 1 && target_dim != grad_dim) {
            throw std::runtime_error("ReduceBroadcast: incompatible shapes");
        }
    }

    Tensor result(target_shape, 0.0f, Device::CUDA);

    cudaError_t error = cudaMemset(
        result.Data(), 0, result.GetSize() * sizeof(float));

    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(
            "ReduceBroadcast: cudaMemset failed: "
            ) + cudaGetErrorString(error)
        );
    }

    size_t* d_grad_shape = nullptr;
    size_t* d_target_shape = nullptr;

    error = cudaMalloc(&d_grad_shape, grad_rank * sizeof(size_t));

    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(
            "ReduceBroadcast: cudaMalloc grad_shape failed: "
            ) + cudaGetErrorString(error)
        );
    }

    error = cudaMalloc(&d_target_shape, target_rank * sizeof(size_t));

    if (error != cudaSuccess) {
        cudaFree(d_grad_shape);

        throw std::runtime_error(std::string(
            "ReduceBroadcast: cudaMalloc target_shape failed: "
            ) + cudaGetErrorString(error)
        );
    }

    error = cudaMemcpy(
        d_grad_shape,
        grad_shape.data(),
        grad_rank * sizeof(size_t),
        cudaMemcpyHostToDevice
    );

    if (error != cudaSuccess) {
        cudaFree(d_grad_shape);
        cudaFree(d_target_shape);

        throw std::runtime_error(std::string(
            "ReduceBroadcast: cudaMemcpy grad_shape failed: "
            ) + cudaGetErrorString(error)
        );
    }

    error = cudaMemcpy(d_target_shape, target_shape.data(),
        target_rank * sizeof(size_t), cudaMemcpyHostToDevice);

    if (error != cudaSuccess) {
        cudaFree(d_grad_shape);
        cudaFree(d_target_shape);

        throw std::runtime_error(std::string(
            "ReduceBroadcast: cudaMemcpy target_shape failed: "
            ) + cudaGetErrorString(error)
        );
    }

    const size_t threads = 256;
    const size_t blocks = (grad.GetSize() + threads - 1) / threads;

    ReduceBroadcastKernel<<<blocks, threads>>>(
        grad.Data(),
        result.Data(),
        grad.GetSize(),
        grad_rank,
        target_rank,
        d_grad_shape,
        d_target_shape
    );

    error = cudaGetLastError();

    if (error != cudaSuccess) {
        cudaFree(d_grad_shape);
        cudaFree(d_target_shape);

        throw std::runtime_error(std::string(
            "ReduceBroadcast: kernel launch failed: "
            ) + cudaGetErrorString(error)
        );
    }

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        cudaFree(d_grad_shape);
        cudaFree(d_target_shape);

        throw std::runtime_error(std::string(
            "ReduceBroadcast: kernel execution failed: "
            ) + cudaGetErrorString(error)
        );
    }

    cudaFree(d_grad_shape);
    cudaFree(d_target_shape);

    return result;
}

__global__ void TanhKernel(const float* input, float* output, size_t size) {
    size_t index = (size_t)(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size) { return; }
    output[index] = tanhf(input[index]);
}

__global__ void TanhBackwardKernel(const float* input, const float* grad_output,
        float* grad_input, size_t size) {

    size_t index = (size_t)(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size) { return; }

    float t = tanhf(input[index]);
    grad_input[index] = grad_output[index] * (1.0f - t * t);
}

Tensor CUDABackend::Tanh(const Tensor& input) const {
    Tensor result(input.GetShape(), Device::CUDA);

    const size_t threads = 256;
    const size_t blocks = (input.GetSize() + threads - 1) / threads;

    TanhKernel<<<blocks, threads>>>(
        input.Data(),
        result.Data(),
        input.GetSize()
    );

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    cudaDeviceSynchronize();
    return result;
}

Tensor CUDABackend::TanhBackward(const Tensor& input, 
        const Tensor& grad_output) const {
    if (input.GetShape() != grad_output.GetShape()) {
        throw std::runtime_error("TanhBackward: shape mismatch");
    }

    Tensor result(input.GetShape(), Device::CUDA);

    const size_t threads = 256;
    const size_t blocks = (input.GetSize() + threads - 1) / threads;

    TanhBackwardKernel<<<blocks, threads>>>(
        input.Data(),
        grad_output.Data(),
        result.Data(),
        input.GetSize()
    );

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    cudaDeviceSynchronize();
    return result;
}

__global__ void CreateCausalMaskKernel(float* mask, size_t query_len, 
        size_t key_len, size_t query_start) {
    size_t index = (size_t)(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total_size = query_len * key_len;
    if (index >= total_size) { return; }

    size_t q = index / key_len;
    size_t k = index % key_len;

    size_t absolute_q = query_start + q;

    if (k <= absolute_q) {
        mask[index] = 0.0f;
    } else {
        mask[index] = -1e9f;
    }
}

Tensor CUDABackend::CreateCausalMask(size_t query_len, size_t key_len, size_t query_start) const {
    if (query_start + query_len > key_len) {
        throw std::runtime_error("CreateCausalMask: invalid query range");
    }

    Tensor result({query_len, key_len}, Device::CUDA);

    size_t total_size = query_len * key_len;
    const size_t threads = 256;
    const size_t blocks = (total_size + threads - 1) / threads;

    CreateCausalMaskKernel<<<blocks, threads>>>(
        result.Data(),
        query_len,
        key_len,
        query_start
    );

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("CreateCausalMask kernel failed: ")
            + cudaGetErrorString(error)
        );
    }

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("CreateCausalMask execution failed: ")
            + cudaGetErrorString(error)
        );
    }

    return result;
}

void CheckCUDA(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(message) + ": " + cudaGetErrorString(error)
        );
    }
}

__global__ void CrossEntropyForwardKernel(const float* logits,
        const float* targets, float* losses, size_t batch, 
        size_t seq_len, size_t vocab_size) {

    size_t position = (size_t)(blockIdx.x);
    size_t total_positions = batch * seq_len;

    if (position >= total_positions) { return; }

    size_t b = position / seq_len;
    size_t s = position % seq_len;

    size_t offset = (b * seq_len + s) * vocab_size;
    int target = (int)(targets[b * seq_len + s]);

    float max_val = logits[offset];

    for (size_t i = 1; i < vocab_size; i++) {
        max_val = fmaxf(max_val, logits[offset + i]);
    }

    float sum_exp = 0.0f;

    for (size_t i = 0; i < vocab_size; i++) {
        sum_exp += expf(logits[offset + i] - max_val);
    }

    float target_exp = expf(logits[offset + (size_t)(target)] - max_val);
    float probability = target_exp / sum_exp;
    probability = fmaxf(probability, 1e-12f);
    losses[position] = -logf(probability);
}


__global__ void CrossEntropyBackwardKernel(const float* logits,
    const float* targets, float* grad, size_t batch, size_t seq_len,
    size_t vocab_size) {

    size_t index = (size_t)(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = batch * seq_len * vocab_size;

    if (index >= total) { return; }

    size_t vocab_index = index % vocab_size;
    size_t position = index / vocab_size;
    int target = (int)(targets[position]);

    if (target < 0 || (size_t)(target) >= vocab_size) {
        grad[index] = 0.0f;
        return;
    }

    size_t offset = position * vocab_size;
    float max_val = logits[offset];

    for (size_t i = 1; i < vocab_size; i++) {
        max_val = fmaxf(max_val, logits[offset + i]);
    }

    float sum_exp = 0.0f;

    for (size_t i = 0; i < vocab_size; i++) {
        sum_exp += expf(logits[offset + i] - max_val);
    }

    float probability = expf(logits[offset + vocab_index] - max_val) / sum_exp;
    float grad_scale = 1.0f / (float)(batch * seq_len);
    grad[index] = probability * grad_scale;

    if ((int)(vocab_index) == target) {
        grad[index] -= grad_scale;
    }
}


Tensor CUDABackend::CrossEntropy(const Tensor& logits, const Tensor& targets) const {
    const auto& logits_shape = logits.GetShape();
    const auto& targets_shape = targets.GetShape();

    if (logits.GetRank() != 3) {
        throw std::runtime_error(
            "CrossEntropy: logits must be 3D "
            "[batch, seq_len, vocab_size]"
        );
    }

    if (targets.GetRank() != 2) {
        throw std::runtime_error(
            "CrossEntropy: targets must be 2D "
            "[batch, seq_len]"
        );
    }

    size_t batch = logits_shape[0];
    size_t seq_len = logits_shape[1];
    size_t vocab_size = logits_shape[2];

    if (targets_shape[0] != batch || targets_shape[1] != seq_len) {
        throw std::runtime_error(
            "CrossEntropy: logits and targets "
            "shapes must match"
        );
    }

    if (logits.GetDevice() != Device::CUDA || targets.GetDevice() != Device::CUDA) {
        throw std::runtime_error(
            "CUDABackend::CrossEntropy: "
            "tensors must be on CUDA"
        );
    }

    size_t positions = batch * seq_len;
    float* losses = nullptr;

    CheckCUDA(cudaMalloc(&losses, positions * sizeof(float)),
        "CrossEntropy: cudaMalloc failed"
    );

    const int threads = 256;
    const int blocks = (int)((positions + threads - 1) / threads);

    CrossEntropyForwardKernel<<<positions, 1>>>(
        logits.Data(),
        targets.Data(),
        losses,
        batch,
        seq_len,
        vocab_size
    );

    CheckCUDA(cudaGetLastError(), "CrossEntropyForwardKernel launch failed");
    CheckCUDA(cudaDeviceSynchronize(), "CrossEntropyForwardKernel failed");

    float* total_loss = nullptr;

    CheckCUDA(cudaMalloc(&total_loss, sizeof(float)),
        "CrossEntropy: cudaMalloc total_loss failed"
    );

    CheckCUDA(cudaMemset(total_loss, 0, sizeof(float)),
        "CrossEntropy: cudaMemset failed"
    );

    float* host_losses = new float[positions];

    CheckCUDA(cudaMemcpy(host_losses, losses, positions * sizeof(float), 
        cudaMemcpyDeviceToHost), "CrossEntropy: D2H copy failed"
    );

    float sum = 0.0f;

    for (size_t i = 0; i < positions; i++) {
        sum += host_losses[i];
    }

    delete[] host_losses;

    cudaFree(losses);
    cudaFree(total_loss);
    float loss = sum / (float)(positions);

    return Tensor({1, 1}, loss, Device::CUDA);
}


Tensor CUDABackend::CrossEntropyBackward(const Tensor& logits, const Tensor& targets) const {
    const auto& logits_shape = logits.GetShape();
    const auto& targets_shape = targets.GetShape();

    if (logits.GetRank() != 3) {
        throw std::runtime_error(
            "CrossEntropyBackward: logits must be 3D "
            "[batch, seq_len, vocab_size]"
        );
    }

    if (targets.GetRank() != 2) {
        throw std::runtime_error(
            "CrossEntropyBackward: targets must be 2D "
            "[batch, seq_len]"
        );
    }

    size_t batch = logits_shape[0];
    size_t seq_len = logits_shape[1];
    size_t vocab_size = logits_shape[2];

    if (targets_shape[0] != batch || targets_shape[1] != seq_len) {
        throw std::runtime_error(
            "CrossEntropyBackward: logits and targets "
            "shapes must match"
        );
    }

    if (logits.GetDevice() != Device::CUDA || targets.GetDevice() != Device::CUDA) {
        throw std::runtime_error(
            "CUDABackend::CrossEntropyBackward: "
            "tensors must be on CUDA"
        );
    }

    Tensor grad(logits_shape, Device::CUDA);

    size_t total = batch * seq_len * vocab_size;

    const int threads = 256;
    const int blocks = (int)((total + threads - 1) / threads);

    CrossEntropyBackwardKernel<<<blocks, threads>>>(
        logits.Data(),
        targets.Data(),
        grad.Data(),
        batch,
        seq_len,
        vocab_size
    );

    CheckCUDA(cudaGetLastError(), "CrossEntropyBackwardKernel launch failed");
    CheckCUDA(cudaDeviceSynchronize(), "CrossEntropyBackwardKernel failed");

    return grad;
}

size_t CUDABackend::ArgMax(const Tensor& tensor) const {
    if (tensor.GetSize() == 0) {
        throw std::runtime_error(
            "CUDABackend::ArgMax: tensor is empty"
        );
    }

    std::vector<float> host_data(tensor.GetSize());

    cudaError_t error = cudaMemcpy(host_data.data(), tensor.Data(),
        tensor.GetSize() * sizeof(float), cudaMemcpyDeviceToHost);

    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDABackend::ArgMax cudaMemcpy failed: ") +
            cudaGetErrorString(error)
        );
    }

    size_t max_index = 0;
    float max_value = host_data[0];

    for (size_t i = 1; i < host_data.size(); i++) {
        if (host_data[i] > max_value) {
            max_value = host_data[i];
            max_index = i;
        }
    }

    return max_index;
}

Tensor CUDABackend::Reshape(const Tensor& input, 
        const std::vector<size_t>& new_shape) const {
    size_t old_size = input.GetSize();
    size_t new_size = 1;

    for (size_t dimension : new_shape) {
        new_size *= dimension;
    }

    if (old_size != new_size) {
        throw std::runtime_error("Reshape: incompatible tensor sizes");
    }

    Tensor result(new_shape, Device::CUDA);

    cudaMemcpy(result.data_, input.data_, old_size * sizeof(float),
        cudaMemcpyDeviceToDevice);

    return result;
}

__global__ void AdamWKernel(float* parameter, float* m, float* v, const float* gradient,
    size_t size, float lr, float beta1, float beta2, float eps, float weight_decay,
    float bias_correction1, float bias_correction2) {

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size) { return; }
    float g = gradient[i];

    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

    float m_hat = m[i] / bias_correction1;
    float v_hat = v[i] / bias_correction2;

    parameter[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) +
        weight_decay * parameter[i]);
}

void CUDABackend::AdamW(Tensor& parameter, Tensor& m, Tensor& v, const Tensor& gradient,
    float lr, float beta1, float beta2, float eps, float weight_decay, size_t step) {
        
    if (parameter.GetDevice() != Device::CUDA || m.GetDevice() != Device::CUDA ||
        v.GetDevice() != Device::CUDA || gradient.GetDevice() != Device::CUDA) {

        throw std::runtime_error("CUDABackend::AdamW: all tensors must be on CUDA");
    }

    if (parameter.GetSize() != m.GetSize() ||
        parameter.GetSize() != v.GetSize() ||
        parameter.GetSize() != gradient.GetSize()) {

        throw std::runtime_error("CUDABackend::AdamW: tensor sizes must match");
    }

    if (step == 0) {
        throw std::runtime_error("CUDABackend::AdamW: step must be greater than 0");
    }

    float bias_correction1 = 1.0f - std::pow(beta1, (float)(step));
    float bias_correction2 = 1.0f - std::pow(beta2, (float)(step));

    const size_t threads = 256;
    const size_t blocks = (parameter.GetSize() + threads - 1) / threads;

    AdamWKernel<<<blocks, threads>>>(
        parameter.Data(),
        m.Data(),
        v.Data(),
        gradient.Data(),
        parameter.GetSize(),
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        bias_correction1,
        bias_correction2
    );

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDABackend::AdamW: kernel launch failed: ") +
            cudaGetErrorString(error)
        );
    }
}