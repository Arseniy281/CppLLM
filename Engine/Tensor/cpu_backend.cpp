#include "tensor.h"
#include "cpu_backend.h"
#include <climits>
#include <cstring>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h> 
#endif

static void CheckDimension(size_t value) {
    if (value > (size_t)(INT_MAX)) {
        throw std::runtime_error("MatMul dimension is too large for Accelerate");
    }
}

Tensor CPUBackend::MatMul2DAccelerate(const Tensor& A, const Tensor& B) {
    const auto& shapeA = A.GetShape();
    const auto& shapeB = B.GetShape();

    if (shapeA.size() != 2 || shapeB.size() != 2) {
        throw std::runtime_error("MatMul2DAccelerate: tensors must be 2D");
    }

    size_t M = shapeA[0];
    size_t K = shapeA[1];

    size_t K2 = shapeB[0];
    size_t N = shapeB[1];

    if (K != K2) {
        throw std::runtime_error("MatMul: incompatible matrix dimensions");
    }

    CheckDimension(M);
    CheckDimension(K);
    CheckDimension(N);

    Tensor result({M, N});

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,

        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K),

        1.0f,
        A.Data(),
        static_cast<int>(K),

        B.Data(),
        static_cast<int>(N),
        0.0f,

        result.Data(),
        static_cast<int>(N)
    );

    return result;
}

Tensor CPUBackend::MatMul3D2DAccelerate(const Tensor& A, const Tensor& B) {
    const auto& shapeA = A.GetShape();
    const auto& shapeB = B.GetShape();

    if (shapeA.size() != 3 || shapeB.size() != 2) {
        throw std::runtime_error("MatMul3D2DAccelerate: invalid ranks");
    }

    size_t batch = shapeA[0];
    size_t M = shapeA[1];
    size_t K = shapeA[2];

    size_t K2 = shapeB[0];
    size_t N = shapeB[1];

    if (K != K2) {
        throw std::runtime_error("MatMul: incompatible matrix dimensions");
    }

    CheckDimension(M);
    CheckDimension(K);
    CheckDimension(N);

    Tensor result({batch, M, N});

    const float* A_data = A.Data();
    const float* B_data = B.Data();
    float* C_data = result.Data();

    size_t A_batch_size = M * K;
    size_t C_batch_size = M * N;

    for (size_t b = 0; b < batch; b++) {
        const float* A_batch = A_data + b * A_batch_size;
        float* C_batch = C_data + b * C_batch_size;

        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,

            static_cast<int>(M),
            static_cast<int>(N),
            static_cast<int>(K),

            1.0f,
            A_batch,
            static_cast<int>(K),

            B_data,
            static_cast<int>(N),

            0.0f,
            C_batch,
            static_cast<int>(N)
        );
    }

    return result;
}

Tensor CPUBackend::MatMulBatchedFallback(const Tensor& A, const Tensor& B) {
    const auto& shapeA = A.GetShape();
    const auto& shapeB = B.GetShape();

    if (shapeA.size() < 2 || shapeB.size() < 2) {
        throw std::runtime_error("MatMul: tensors must have rank >= 2");
    }

    size_t rankA = shapeA.size();
    size_t rankB = shapeB.size();

    size_t M = shapeA[rankA - 2];
    size_t K = shapeA[rankA - 1];

    size_t K2 = shapeB[rankB - 2];
    size_t N = shapeB[rankB - 1];

    if (K != K2) {
        throw std::runtime_error("MatMul: incompatible matrix dimensions");
    }

    size_t batchRank = std::max(rankA, rankB) - 2;

    std::vector<size_t> resultShape(batchRank);

    for (size_t i = 0; i < batchRank; ++i) {
        size_t offsetA = batchRank - (rankA - 2);
        size_t offsetB = batchRank - (rankB - 2);

        size_t dimA = (i < offsetA) ? 1 : shapeA[i - offsetA];
        size_t dimB = (i < offsetB) ? 1 : shapeB[i - offsetB];

        if (dimA != dimB && dimA != 1 && dimB != 1) {
            throw std::runtime_error("MatMul: incompatible batch dimensions");
        }

        resultShape[i] = std::max(dimA, dimB);
    }

    resultShape.push_back(M);
    resultShape.push_back(N);

    Tensor result(resultShape);
    size_t totalBatch = 1;

    for (size_t i = 0; i < batchRank; i++) {
        totalBatch *= resultShape[i];
    }

    size_t C_matrix_size = M * N;

    const float* A_data = A.Data();
    const float* B_data = B.Data();
    float* C_data = result.Data();

    std::vector<size_t> stridesA(rankA, 1);
    std::vector<size_t> stridesB(rankB, 1);

    for (int i = (int)(rankA) - 2; i >= 0; i--) {
        stridesA[i] = stridesA[i + 1] * shapeA[i + 1];
    }

    for (int i = (int)(rankB) - 2; i >= 0; i--) {
        stridesB[i] = stridesB[i + 1] * shapeB[i + 1];
    }

    std::vector<size_t> resultIndices(batchRank);

    for (size_t batchIndex = 0; batchIndex < totalBatch; batchIndex++) {
        size_t tmp = batchIndex;

        for (int i = (int)(batchRank) - 1; i >= 0; i--) {
            resultIndices[i] = tmp % resultShape[i];
            tmp /= resultShape[i];
        }

        size_t offsetA = 0;
        size_t offsetB = 0;

        for (size_t i = 0; i < batchRank; i++) {
            size_t offsetDimA = batchRank - (rankA - 2);
            size_t offsetDimB = batchRank - (rankB - 2);

            if (i >= offsetDimA) {
                size_t dimA = shapeA[i - offsetDimA];

                if (dimA != 1) {
                    offsetA += resultIndices[i] * stridesA[i - offsetDimA];
                }
            }

            if (i >= offsetDimB) {
                size_t dimB = shapeB[i - offsetDimB];
                if (dimB != 1) {
                    offsetB += resultIndices[i] * stridesB[i - offsetDimB];
                }
            }
        }

        const float* A_matrix = A_data + offsetA;
        const float* B_matrix = B_data + offsetB;
        float* C_matrix = C_data + batchIndex * C_matrix_size;

        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,

            static_cast<int>(M),
            static_cast<int>(N),
            static_cast<int>(K),

            1.0f,
            A_matrix,
            static_cast<int>(K),

            B_matrix,
            static_cast<int>(N),

            0.0f,
            C_matrix,
            static_cast<int>(N)
        );
    }

    return result;
}

Tensor CPUBackend::Add(const Tensor& a, const Tensor& b) const {
    if (a.shape_ == b.shape_) {
        Tensor result(a.shape_);

        for (size_t i = 0; i < a.size_; i++) {
            result.data_[i] = a.data_[i] + b.data_[i];
        }

        return result;
    }

    std::vector<size_t> s1 = a.shape_;
    std::vector<size_t> s2 = b.GetShape();

    s1 = Tensor::AlignTensors(s1, s2);
    s2 = Tensor::AlignTensors(s2, s1);

    const float* data_1 = a.Data();
    const float* data_2 = b.Data();

    std::vector<size_t> final_shape = Tensor::GetFinalShape(s1, s2);
    size_t final_size = Tensor::GetFinalSize(final_shape);

    std::vector<float> final_data(final_size);

    for (size_t i = 0; i < final_size; i++) {
        final_data[i] =
            data_1[Tensor::BroadcastIndex(s1, final_shape, i)] +
            data_2[Tensor::BroadcastIndex(s2, final_shape, i)];
    }

    return Tensor(final_shape, final_data);
}

Tensor CPUBackend::Sub(const Tensor& a, const Tensor& b) const {
    if (a.shape_ == b.shape_) {
        Tensor result(a.shape_);

        for (size_t i = 0; i < a.size_; i++) {
            result.data_[i] = a.data_[i] - b.data_[i];
        }

        return result;
    }

    std::vector<size_t> s1 = a.shape_;
    std::vector<size_t> s2 = b.GetShape();
    s1 = Tensor::AlignTensors(s1, s2);
    s2 = (Tensor::AlignTensors(s2, s1));

    const float* data_1 = a.Data();
    const float* data_2 = b.Data();

    std::vector<size_t> final_shape = Tensor::GetFinalShape(s1, s2);
    size_t final_size = Tensor::GetFinalSize(final_shape);

    std::vector<float> final_data(final_size);
    for (size_t i = 0; i < final_size; i++) {
        final_data[i] = data_1[Tensor::BroadcastIndex(s1, final_shape, i)]
                        - data_2[Tensor::BroadcastIndex(s2, final_shape, i)];
    }

    Tensor result(final_shape, final_data);
    return result;
}

Tensor CPUBackend::Mul(const Tensor& a, const Tensor& b) const {
    if (a.shape_ == b.shape_) {
        Tensor result(a.shape_);

        for (size_t i = 0; i < a.size_; i++) {
            result.data_[i] = a.data_[i] * b.data_[i];
        }

        return result;
    }

    std::vector<size_t> s1 = a.shape_;
    std::vector<size_t> s2 = b.GetShape();
    s1 = Tensor::AlignTensors(s1, s2);
    s2 = (Tensor::AlignTensors(s2, s1));

    const float* data_1 = a.Data();
    const float* data_2 = b.Data();

    std::vector<size_t> final_shape = Tensor::GetFinalShape(s1, s2);
    size_t final_size = Tensor::GetFinalSize(final_shape);

    std::vector<float> final_data(final_size);
    for (size_t i = 0; i < final_size; i++) {
        final_data[i] = data_1[Tensor::BroadcastIndex(s1, final_shape, i)]
                        * data_2[Tensor::BroadcastIndex(s2, final_shape, i)];
    }

    Tensor result(final_shape, final_data);
    return result;
}

Tensor CPUBackend::Div(const Tensor& a, const Tensor& b) const {
    if (a.shape_ == b.shape_) {
        Tensor result(a.shape_);

        for (size_t i = 0; i < a.size_; i++) {
            const float divisor = b.data_[i];
            result.data_[i] = (divisor == 0.0f) ? 0.0f : a.data_[i] / divisor;
        }

        return result;
    }

    std::vector<size_t> s1 = a.shape_;
    std::vector<size_t> s2 = b.GetShape();
    s1 = Tensor::AlignTensors(s1, s2);
    s2 = (Tensor::AlignTensors(s2, s1));

    const float* data_1 = a.Data();
    const float* data_2 = b.Data();

    std::vector<size_t> final_shape = Tensor::GetFinalShape(s1, s2);
    size_t final_size = Tensor::GetFinalSize(final_shape);

    std::vector<float> final_data(final_size);
    for (size_t i = 0; i < final_size; i++) {
        size_t idx1 = Tensor::BroadcastIndex(s1, final_shape, i);
        size_t idx2 = Tensor::BroadcastIndex(s2, final_shape, i);

        float divisor = data_2[idx2];
        if (divisor == 0.0f) {
            final_data[i] = 0.0f;
        } else {
            final_data[i] = data_1[idx1] / divisor;
        }
    }

    Tensor result(final_shape, final_data);
    return result;
}

Tensor CPUBackend::Sum(const Tensor& input) const {
    float total = 0.0f;
    for (size_t i = 0; i < input.GetSize(); i++) {
        total += input.at(i);
    }

    Tensor result({1, 1}, total);

    return result;
}

Tensor CPUBackend::AddScalar(const Tensor& a, float scalar) const {
    Tensor tensor(a.shape_);
    for (size_t i = 0; i < a.size_; i++) {
        tensor.data_[i] = a.data_[i] + scalar;
    }
    return tensor;
}

Tensor CPUBackend::SubScalar(const Tensor& a, float scalar) const {
    Tensor tensor(a.shape_);
    for (size_t i = 0; i < a.size_; i++) {
        tensor.data_[i] = a.data_[i] - scalar;
    }
    return tensor;
}

Tensor CPUBackend::MulScalar(const Tensor& a, float scalar) const {
    Tensor tensor(a.shape_);
    for (size_t i = 0; i < a.size_; i++) {
        tensor.data_[i] = a.data_[i] * scalar;
    }
    return tensor;
}

Tensor CPUBackend::DivScalar(const Tensor& a, float scalar) const {
    if (scalar == 0.0f) {
        throw std::runtime_error("Division by zero");
    }

    Tensor tensor(a.shape_);

    for (size_t i = 0; i < a.size_; i++) {
        tensor.data_[i] = a.data_[i] / scalar;
    }

    return tensor;
}

Tensor CPUBackend::Neg(const Tensor& a) const {
    Tensor tensor(a.shape_);
    for (size_t i = 0; i < a.size_; i++) {
        tensor.data_[i] = -a.data_[i];
    }
    return tensor;
}

Tensor CPUBackend::Transpose(const Tensor& a) const {
    if (a.rank_ < 2) {
        throw std::runtime_error("Transpose requires rank >= 2");
    }

    std::vector<size_t> new_shape = a.shape_;
    std::swap(new_shape[a.rank_ - 1], new_shape[a.rank_ - 2]);

    Tensor result(new_shape);

    size_t rows = a.shape_[a.rank_ - 2];
    size_t cols = a.shape_[a.rank_ - 1];

    size_t outer_size = a.size_ / (rows * cols);

    for (size_t outer = 0; outer < outer_size; outer++) {
        size_t input_offset = outer * rows * cols;
        size_t output_offset = outer * rows * cols;

        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.at(output_offset + j * rows + i) 
                    = a.data_[input_offset + i * cols + j];
            }
        }
    }

    return result;
}

Tensor CPUBackend::Concatenate(const std::vector<Tensor>& tensors, size_t axis) const {
    if (tensors.empty()) {
        throw std::runtime_error("Tensors must have size > 0");
    }

    if (axis >= tensors[0].GetRank()) {
        throw std::runtime_error("Concatenate: axis out of bounds");
    }
    
    std::vector<size_t> final_shape = tensors[0].GetShape();
    for (size_t t = 1; t < tensors.size(); t++) {
        auto shape = tensors[t].GetShape();
        if (shape.size() != final_shape.size()) {
            throw std::runtime_error("All tensors must have same rank");
        }
        for (size_t j = 0; j < final_shape.size(); j++) {
            if (j != axis && final_shape[j] != shape[j]) {
                throw std::runtime_error(
                    "All tensors must have same shape except on axis");
            }
        }
    }
    
    size_t total_dim = 0;
    for (const auto& t : tensors) {
        total_dim += t.GetShape()[axis];
    }
    final_shape[axis] = total_dim;
    
    Tensor output(final_shape);
    float* result_data = output.Data();
    size_t offset = 0;
    
    for (const auto& tensor : tensors) {
        const float* data = tensor.Data();
        size_t tensor_size = tensor.GetSize();
        
        for (size_t i = 0; i < tensor_size; i++) {
            std::vector<size_t> coords 
                = Tensor::IndexToCoord(i, tensor.GetShape());
            coords[axis] += offset;
            size_t idx = Tensor::CoordToIndex(coords, final_shape);
            result_data[idx] = data[i];
        }
        offset += tensor.GetShape()[axis];
    }
    
    return output;
}

std::vector<Tensor> CPUBackend::Split(const Tensor& tensor, 
        const std::vector<std::vector<size_t>>& shapes, size_t axis) const {

    const auto& input_shape = tensor.GetShape();

    if (axis >= input_shape.size()) {
        throw std::runtime_error( "CPUBackend::Split: axis out of bounds");
    }
    if (shapes.empty()) {
        throw std::runtime_error("CPUBackend::Split: no shapes");
    }

    size_t total_axis = 0;

    for (const auto& shape : shapes) {
        if (shape.size() != input_shape.size()) {
            throw std::runtime_error("CPUBackend::Split: rank mismatch");
        }

        for (size_t i = 0; i < input_shape.size(); ++i) {
            if (i != axis && shape[i] != input_shape[i]) {
                throw std::runtime_error("CPUBackend::Split: shape mismatch");
            }
        }

        total_axis += shape[axis];
    }

    if (total_axis != input_shape[axis]) {
        throw std::runtime_error(
            "CPUBackend::Split: split sizes do not match input");
    }

    std::vector<Tensor> result;
    result.reserve(shapes.size());

    size_t offset = 0;

    const size_t outer = std::accumulate(input_shape.begin(),
            input_shape.begin() + axis, size_t{1}, std::multiplies<size_t>());

    const size_t inner = std::accumulate(input_shape.begin() + axis + 1,
            input_shape.end(), size_t{1}, std::multiplies<size_t>());

    for (const auto& shape : shapes) {
        const size_t axis_size = shape[axis];
        Tensor output(shape, 0.0f);

        const float* src = tensor.Data();
        float* dst = output.Data();

        for (size_t o = 0; o < outer; o++) {
            const size_t src_offset = o * input_shape[axis] * inner + offset * inner;
            const size_t count = axis_size * inner;

            std::memcpy(dst + o * count, src + src_offset,
                count * sizeof(float));
        }

        result.push_back(std::move(output));
        offset += axis_size;
    }

    return result;
}


Tensor CPUBackend::SumAxis(const Tensor& a, int axis) const {
    if (axis < 0 || axis >= (int)(a.rank_)) {
        throw std::runtime_error("Axis is out of bounds");
    }
    std::vector<size_t> result_shape = a.shape_;
    result_shape.erase(result_shape.begin() + axis);
    Tensor result(result_shape, 0.0f);
    for (size_t i = 0; i < a.size_; i++) {
        std::vector<size_t> coord = Tensor::IndexToCoord(i, a.shape_);
        std::vector<size_t> result_coord = coord;
        result_coord.erase(result_coord.begin() + axis);
        result.at(result_coord) += a.data_[i];
    }
    return result;
}

Tensor CPUBackend::Mean(const Tensor& a, int axis) const {
    if (axis < 0 || axis >= (int)(a.rank_)) {
        throw std::runtime_error("Axis is out of bounds");
    }

    size_t dim = a.shape_[axis];

    Tensor result = SumAxis(a, axis);
    result /= (float)(dim);

    return result;
}

Tensor CPUBackend::MatMul(const Tensor& a, const Tensor& b) const {

    const auto& shapeA = a.GetShape();
    const auto& shapeB = b.GetShape();

    if (shapeA.size() < 2 || shapeB.size() < 2) {
        throw std::runtime_error(
            "MatMul: tensors must have rank >= 2"
        );
    }

    if (shapeA.size() == 2 && shapeB.size() == 2) {
        return MatMul2DAccelerate(a, b);
    }

    if (shapeA.size() == 3 && shapeB.size() == 2) {
        return MatMul3D2DAccelerate(a, b);
    }

    return MatMulBatchedFallback(a, b);
}

Tensor CPUBackend::ScalarDiv(float scalar, const Tensor& t) const {
    Tensor result(t.GetShape());

    for (size_t i = 0; i < t.GetSize(); i++) {
        const float divisor = t.Data()[i];
        result.Data()[i] =(divisor == 0.0f) ? 0.0f : scalar / divisor;
    }

    return result;
}

Tensor CPUBackend::Embedding(const Tensor& embeddings,
        const Tensor& indices) const {

    const auto& embedding_shape = embeddings.GetShape();
    if (embedding_shape.size() != 2) {
        throw std::runtime_error("CPUBackend::Embedding: embeddings must be 2D");
    }

    const size_t vocab_size = embedding_shape[0];
    const size_t embedding_dim = embedding_shape[1];

    std::vector<size_t> result_shape = indices.GetShape();
    result_shape.push_back(embedding_dim);
    Tensor result(result_shape, 0.0f);

    for (size_t i = 0; i < indices.GetSize(); i++) {
        const size_t index = (size_t)(indices.Data()[i]);

        if (index >= vocab_size) {
            throw std::runtime_error(
                "CPUBackend::Embedding: index out of range");
        }

        const float* source = embeddings.Data() + index * embedding_dim;
        float* destination = result.Data() + i * embedding_dim;

        std::copy(source, source + embedding_dim, destination);
    }

    return result;
}

Tensor CPUBackend::EmbeddingBackward(const Tensor& embeddings, 
        const Tensor& indices, const Tensor& grad_output) const {

    const auto& embedding_shape = embeddings.GetShape();

    if (embedding_shape.size() != 2) {
        throw std::runtime_error(
            "CPUBackend::EmbeddingBackward: embeddings must be 2D"
        );
    }

    const size_t vocab_size = embedding_shape[0];
    const size_t embedding_dim = embedding_shape[1];

    const auto& indices_shape = indices.GetShape();
    const auto& grad_shape = grad_output.GetShape();

    std::vector<size_t> expected_grad_shape = indices_shape;
    expected_grad_shape.push_back(embedding_dim);

    if (grad_shape != expected_grad_shape) {
        throw std::runtime_error(
            "CPUBackend::EmbeddingBackward: grad_output shape mismatch"
        );
    }

    Tensor grad({vocab_size, embedding_dim}, 0.0f);

    const float* indices_data = indices.Data();
    const float* grad_output_data = grad_output.Data();
    float* grad_data = grad.Data();

    const size_t indices_count = indices.GetSize();

    for (size_t i = 0; i < indices_count; i++) {
        const size_t index = (size_t)(indices_data[i]);

        if (index >= vocab_size) {
            throw std::runtime_error(
                "CPUBackend::EmbeddingBackward: index out of range"
            );
        }

        float* destination = grad_data + index * embedding_dim;
        const float* source = grad_output_data + i * embedding_dim;

        for (size_t j = 0; j < embedding_dim; j++) {
            destination[j] += source[j];
        }
    }

    return grad;
}

Tensor CPUBackend::Gelu(const Tensor& a) const {
    Tensor result(a.GetShape(), 0.0f);
    const float pi = 3.14159265358979323846f;
    const float sqrt_2_pi = std::sqrt(2.0f / pi);
    const float coeff = 0.044715f;

    for (size_t i = 0; i < a.GetSize(); i++) {
        const float x = a.Data()[i];
        const float x3 = x * x * x;
        result.Data()[i] = 0.5f * x * (1.0f + std::tanh(sqrt_2_pi * (x + coeff * x3)));
    }

    return result;
}

Tensor CPUBackend::GeluBackward(const Tensor& input,
        const Tensor& grad_output) const {
    if (input.GetShape() != grad_output.GetShape()) {
        throw std::runtime_error("CPUBackend::GeluBackward: shape mismatch");
    }

    Tensor result(input.GetShape(), 0.0f);
    const float pi = 3.14159265358979323846f;
    const float sqrt_2_pi = std::sqrt(2.0f / pi);
    const float a = 0.044715f;

    for (size_t i = 0; i < input.GetSize(); i++) {
        const float x = input.Data()[i];
        const float x2 = x * x;
        const float x3 = x2 * x;
        const float arg = sqrt_2_pi * (x + a * x3);
        const float tanh_val = std::tanh(arg);
        const float sech_sq = 1.0f - tanh_val * tanh_val;
        const float derivative = 0.5f * (1.0f + tanh_val) + 
                0.5f * x * sech_sq * sqrt_2_pi * (1.0f + 3.0f * a * x2);

        result.Data()[i] = grad_output.Data()[i] * derivative;
    }

    return result;
}

Tensor CPUBackend::ReLU(const Tensor& a) const {
    Tensor result(a.GetShape(), 0.0f);
    for (size_t i = 0; i < a.GetSize(); i++) {
        result.Data()[i] = std::max(0.0f, a.Data()[i]);
    }

    return result;
}

Tensor CPUBackend::ReLUBackward(const Tensor& input,
        const Tensor& grad_output) const {
    if (input.GetShape() != grad_output.GetShape()) {
        throw std::runtime_error("CPUBackend::ReLUBackward: shape mismatch");
    }

    Tensor result(input.GetShape(), 0.0f);

    for (size_t i = 0; i < input.GetSize(); i++) {
        result.Data()[i] = input.Data()[i] > 0.0f ? grad_output.Data()[i] : 0.0f;
    }

    return result;
}

std::vector<Tensor> CPUBackend::RMSNormForward(const Tensor& input, const Tensor& gamma) const {
    const std::vector<size_t>& shape = input.GetShape();

    if (shape.size() != 3) {
        throw std::runtime_error("CPUBackend::RMSNormForward: expected 3D input");
    }

    const size_t batch = shape[0];
    const size_t seq_len = shape[1];
    const size_t embed_dim = shape[2];

    if (gamma.GetShape() != std::vector<size_t>{embed_dim}) {
        throw std::runtime_error("CPUBackend::RMSNormForward: invalid gamma shape");
    }

    Tensor result(shape, 0.0f);
    Tensor x_norm(shape, 0.0f);
    Tensor rms({batch, seq_len, 1}, 0.0f);

    const float* input_data = input.Data();
    const float* gamma_data = gamma.Data();
    float* result_data = result.Data();
    float* x_norm_data = x_norm.Data();
    float* rms_data = rms.Data();

    for (size_t b = 0; b < batch; b++) {

        for (size_t pos = 0; pos < seq_len; pos++) {
            const size_t offset = (b * seq_len + pos) * embed_dim;
            float mean_square = 0.0f;

            for (size_t i = 0; i < embed_dim; i++) {
                const float x = input_data[offset + i];
                mean_square += x * x;
            }

            mean_square /= static_cast<float>(embed_dim);
            const float current_rms = std::sqrt(mean_square + 1e-6f);
            rms_data[b * seq_len + pos] = current_rms;

            for (size_t i = 0; i < embed_dim; i++) {
                const float x = input_data[offset + i];
                const float normalized = x / current_rms;

                x_norm_data[offset + i] = normalized;
                result_data[ offset + i] = normalized * gamma_data[i];
            }
        }
    }

    return {std::move(result), std::move(x_norm), std::move(rms)};
}

std::vector<Tensor> CPUBackend::RMSNormBackward(const Tensor& input, const Tensor& gamma,
        const Tensor& x_norm, const Tensor& rms, const Tensor& grad_output) const {
    const std::vector<size_t>& shape = input.GetShape();

    if (shape.size() != 3) {
        throw std::runtime_error("CPUBackend::RMSNormBackward: expected 3D input");
    }
    if (grad_output.GetShape() != shape) {
        throw std::runtime_error("CPUBackend::RMSNormBackward: invalid gradient shape");
    }

    const size_t batch = shape[0];
    const size_t seq_len = shape[1];
    const size_t embed_dim = shape[2];

    Tensor grad_gamma({embed_dim}, 0.0f);
    Tensor grad_x(shape, 0.0f);

    const float* grad_output_data = grad_output.Data();
    const float* x_norm_data = x_norm.Data();
    const float* rms_data = rms.Data();
    const float* gamma_data = gamma.Data();
    float* grad_gamma_data = grad_gamma.Data();
    float* grad_x_data = grad_x.Data();

    for (size_t b = 0; b < batch; b++) {
        for (size_t pos = 0; pos < seq_len; pos++) {
            const size_t offset = (b * seq_len + pos) * embed_dim;
            for (size_t i = 0; i < embed_dim; i++) {
                grad_gamma_data[i] += grad_output_data[offset + i] 
                                            * x_norm_data[offset + i];
            }
        }
    }

    for (size_t b = 0; b < batch; b++) {
        for (size_t pos = 0; pos < seq_len; pos++) {
            const size_t offset = (b * seq_len + pos) * embed_dim;
            const size_t rms_offset = b * seq_len + pos;
            const float current_rms = rms_data[rms_offset];
            float mean_term = 0.0f;

            for (size_t i = 0; i < embed_dim; i++) {
                const float grad_y = grad_output_data[offset + i];
                const float gamma_value = gamma_data[i];
                const float normalized = x_norm_data[offset + i];
                mean_term += grad_y * gamma_value * normalized;
            }

            mean_term /= (float)(embed_dim);
            const float inv_rms = 1.0f / current_rms;
            for (size_t i = 0; i < embed_dim; i++) {
                const float grad_y = grad_output_data[offset + i];
                const float gamma_value = gamma_data[i];
                const float normalized = x_norm_data[offset + i];
                grad_x_data[offset + i] = (grad_y * gamma_value - normalized * mean_term) * inv_rms;
            }
        }
    }

    return {std::move(grad_x), std::move(grad_gamma)};
}

Tensor CPUBackend::RoPE(const Tensor& tensor, size_t start_pos) const {
    const std::vector<size_t>& shape = tensor.GetShape();

    if (shape.size() != 3) {
        throw std::runtime_error("CPUBackend::RoPE: tensor must be 3D");
    }

    const size_t batch = shape[0];
    const size_t seq_len = shape[1];
    const size_t head_dim = shape[2];

    if (head_dim % 2 != 0) {
        throw std::runtime_error("CPUBackend::RoPE: head_dim must be even");
    }

    Tensor result(shape, 0.0f);

    const float* input = tensor.Data();
    float* output = result.Data();

    for (size_t pos = 0; pos < seq_len; pos++) {
        const size_t actual_pos = start_pos + pos;
        const size_t pos_offset = pos * head_dim;

        for (size_t i = 0; i < head_dim; i += 2) {
            const float theta = (float)(actual_pos)
                 * std::pow(10000.0f, -(float)(i) / head_dim);

            const float cos_theta = std::cos(theta);
            const float sin_theta = std::sin(theta);

            for (size_t b = 0; b < batch; b++) {
                const size_t index = b * seq_len * head_dim + pos_offset + i;
                const float x = input[index];
                const float y = input[index + 1];
                output[index] = x * cos_theta - y * sin_theta;
                output[index + 1] = x * sin_theta + y * cos_theta;
            }
        }
    }

    return result;
}

Tensor CPUBackend::RoPEBackward(const Tensor& tensor, size_t start_pos) const {
    const std::vector<size_t>& shape = tensor.GetShape();
    if (shape.size() != 3) {
        throw std::runtime_error("CPUBackend::RoPEBackward: tensor must be 3D");
    }

    const size_t batch = shape[0];
    const size_t seq_len = shape[1];
    const size_t head_dim = shape[2];

    if (head_dim % 2 != 0) {
        throw std::runtime_error("CPUBackend::RoPEBackward: head_dim must be even");
    }

    Tensor result(shape, 0.0f);

    const float* input = tensor.Data();
    float* output = result.Data();

    for (size_t pos = 0; pos < seq_len; pos++) {
        const size_t actual_pos = start_pos + pos;
        const size_t pos_offset = pos * head_dim;

        for (size_t i = 0; i < head_dim; i += 2) {
            const float theta = (float)(actual_pos) 
                * std::pow(10000.0f, -(float)(i) / head_dim);

            const float cos_theta = std::cos(theta);
            const float sin_theta = std::sin(theta);

            for (size_t b = 0; b < batch; b++) {
                const size_t index = b * seq_len * head_dim + pos_offset + i;
                const float x = input[index];
                const float y = input[index + 1];
                output[index] = x * cos_theta + y * sin_theta;
                output[index + 1] = -x * sin_theta + y * cos_theta;
            }
        }
    }

    return result;
}

Tensor CPUBackend::Sigmoid(const Tensor& input) const {
    Tensor result(input.GetShape(), 0.0f);

    const float* input_data = input.Data();
    float* output_data = result.Data();

    for (size_t i = 0; i < input.GetSize(); i++) {
        output_data[i] = 1.0f / (1.0f + std::exp(-input_data[i]));
    }

    return result;
}

Tensor CPUBackend::SigmoidBackward(const Tensor& input, 
        const Tensor& grad_output) const {
    if (input.GetShape() != grad_output.GetShape()) {
        throw std::runtime_error("CPUBackend::SigmoidBackward: shape mismatch");
    }

    Tensor result(input.GetShape(), 0.0f);

    const float* input_data = input.Data();
    const float* grad_data = grad_output.Data();
    float* output_data = result.Data();

    for (size_t i = 0; i < input.GetSize(); i++) {
        const float s = 1.0f / (1.0f + std::exp(-input_data[i]));
        output_data[i] = grad_data[i] * s * (1.0f - s);
    }

    return result;
}

Tensor CPUBackend::Softmax(const Tensor& input) const {
    const std::vector<size_t>& shape = input.GetShape();

    if (shape.empty()) {
        throw std::runtime_error("CPUBackend::Softmax: input must have at least one dimension");
    }

    const size_t last_dim = shape.back();
    size_t total_vectors = 1;

    for (size_t i = 0; i + 1 < shape.size(); i++) {
        total_vectors *= shape[i];
    }

    Tensor result(shape, 0.0f);
    const float* input_data = input.Data();
    float* output_data = result.Data();

    for (size_t i = 0; i < total_vectors; i++) {
        const size_t base = i * last_dim;
        float max_value = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < last_dim; j++) {
            max_value = std::max(max_value, input_data[base + j]);
        }

        float sum = 0.0f;

        for (size_t j = 0; j < last_dim; j++) {
            const float value = std::exp(input_data[base + j] - max_value);
            output_data[base + j] = value;
            sum += value;
        }

        for (size_t j = 0; j < last_dim; j++) {
            output_data[base + j] /= sum;
        }
    }

    return result;
}

Tensor CPUBackend::SoftmaxBackward(const Tensor& output,
        const Tensor& grad_output) const {
    const std::vector<size_t>& shape = output.GetShape();

    if (shape.empty()) {
        throw std::runtime_error(
            "CPUBackend::SoftmaxBackward: output must have at least one dimension");
    }

    if (grad_output.GetShape() != shape) {
        throw std::runtime_error("CPUBackend::SoftmaxBackward: shape mismatch");
    }

    const size_t embed_dim = shape.back();
    const size_t total_size = output.GetSize();
    const size_t num_vectors = total_size / embed_dim;
    Tensor grad_input(shape, 0.0f);

    const float* output_data = output.Data();
    const float* grad_data = grad_output.Data();
    float* grad_input_data = grad_input.Data();

    for (size_t vec_idx = 0; vec_idx < num_vectors; vec_idx++) {
        const size_t base = vec_idx * embed_dim;
        float sum = 0.0f;

        for (size_t d = 0; d < embed_dim;d++) {
            const size_t index = base + d;
            sum += grad_data[index] * output_data[index];
        }

        for (size_t d = 0; d < embed_dim; d++) {
            const size_t index = base + d;
            grad_input_data[index] = output_data[index] 
                                * (grad_data[index] - sum);
        }
    }

    return grad_input;
}

Tensor CPUBackend::ReduceBroadcast(const Tensor& grad,
        const std::vector<size_t>& target_shape) const {
    const std::vector<size_t>& grad_shape = grad.GetShape();

    if (target_shape.size() > grad_shape.size()) {
        throw std::runtime_error(
            "ReduceBroadcast: target rank cannot be greater than grad rank");
    }

    const size_t grad_rank = grad_shape.size();
    const size_t target_rank = target_shape.size();

    for (size_t i = 0; i < target_rank; i++) {
        const size_t grad_axis = grad_rank - target_rank + i;

        const size_t grad_dim = grad_shape[grad_axis];
        const size_t target_dim = target_shape[i];

        if (target_dim != 1 && target_dim != grad_dim) {
            throw std::runtime_error("ReduceBroadcast: incompatible shapes");
        }
    }

    Tensor result(target_shape, 0.0f);

    for (size_t grad_index = 0; grad_index < grad.GetSize(); grad_index++) {
        std::vector<size_t> grad_coord 
                = Tensor::IndexToCoord(grad_index, grad_shape);

        std::vector<size_t> target_coord(target_rank);

        for (size_t i = 0; i < target_rank; i++) {
            const size_t grad_axis = grad_rank - target_rank + i;

            if (target_shape[i] == 1) {
                target_coord[i] = 0;
            } else {
                target_coord[i] = grad_coord[grad_axis];
            }
        }

        size_t target_index = Tensor::CoordToIndex(target_coord, target_shape);
        result.at(target_index) += grad.at(grad_index);
    }

    return result;
}

Tensor CPUBackend::Tanh(const Tensor& input) const {
    Tensor result(input.GetShape(), UninitializedTag{});

    for (size_t i = 0; i < input.GetSize(); i++) {
        result.at(i) = std::tanh(input.at(i));
    }

    return result;
}

Tensor CPUBackend::TanhBackward(const Tensor& input,
        const Tensor& grad_output) const {
    if (input.GetShape() != grad_output.GetShape()) {
        throw std::runtime_error("TanhBackward: shape mismatch");
    }

    Tensor result(input.GetShape(), UninitializedTag{});

    for (size_t i = 0; i < input.GetSize(); i++) {
        float t = std::tanh(input.at(i));
        result.at(i) = grad_output.at(i) * (1.0f - t * t);
    }
    return result;
}

Tensor CPUBackend::CreateCausalMask(size_t query_len,
        size_t key_len, size_t query_start) const {
    if (query_start + query_len > key_len) {
        throw std::runtime_error("CreateCausalMask: invalid query range");
    }

    Tensor mask({query_len, key_len}, 0.0f);
    const float neg_inf = -1e9f;

    for (size_t q = 0; q < query_len; q++) {
        const size_t absolute_q = query_start + q;

        for (size_t k = 0; k < key_len; k++) {
            if (k > absolute_q) {
                mask.at({q, k}) = neg_inf;
            }
        }
    }

    return mask;
}

Tensor CPUBackend::CrossEntropy(const Tensor& logits,
        const Tensor& targets) const {
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

    const size_t batch = logits_shape[0];
    const size_t seq_len = logits_shape[1];
    const size_t vocab_size = logits_shape[2];

    if (targets_shape[0] != batch || targets_shape[1] != seq_len) {
        throw std::runtime_error(
            "CrossEntropy: logits and targets shapes must match"
        );
    }

    if (logits.GetDevice() != Device::CPU ||
        targets.GetDevice() != Device::CPU) {
        throw std::runtime_error(
            "CPUBackend::CrossEntropy: tensors must be on CPU"
        );
    }

    float total_loss = 0.0f;

    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {

            int target = (int)(targets.at({b, s}));

            if (target < 0 || target >= (int)(vocab_size)) {
                throw std::runtime_error("CrossEntropy: target out of bounds");
            }

            float max_val = logits.at({b, s, 0});
            for (size_t i = 1; i < vocab_size; i++) {
                max_val = std::max(max_val, logits.at({b, s, i}));
            }

            float sum_exp = 0.0f;
            for (size_t i = 0; i < vocab_size; i++) {
                sum_exp += std::exp(logits.at({b, s, i}) - max_val);
            }

            float target_exp = std::exp(
                logits.at({b, s, (size_t)(target)}) - max_val);
            float probability = target_exp / sum_exp;
            probability = std::max(probability, 1e-12f);

            total_loss -= std::log(probability);
        }
    }

    const float loss = total_loss / (float)(batch * seq_len);
    return Tensor({1, 1}, loss);
}


Tensor CPUBackend::CrossEntropyBackward(const Tensor& logits,
        const Tensor& targets) const {
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

    const size_t batch = logits_shape[0];
    const size_t seq_len = logits_shape[1];
    const size_t vocab_size = logits_shape[2];

    if (targets_shape[0] != batch ||
        targets_shape[1] != seq_len) {
        throw std::runtime_error(
            "CrossEntropyBackward: logits and targets shapes must match"
        );
    }

    if (logits.GetDevice() != Device::CPU ||
        targets.GetDevice() != Device::CPU) {
        throw std::runtime_error(
            "CPUBackend::CrossEntropyBackward: "
            "tensors must be on CPU"
        );
    }

    Tensor grad(logits_shape, 0.0f);
    const float grad_scale = 1.0f / (float)(batch * seq_len);

    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            int target = (int)(targets.at({b, s}));

            if (target < 0 || target >= (int)(vocab_size)) {
                throw std::runtime_error("CrossEntropyBackward: target out of bounds");
            }

            float max_val = logits.at({b, s, 0});
            for (size_t i = 1; i < vocab_size; i++) {
                max_val = std::max(max_val, logits.at({b, s, i}));
            }

            float sum_exp = 0.0f;
            for (size_t i = 0; i < vocab_size; i++) {
                sum_exp += std::exp(logits.at({b, s, i}) - max_val);
            }

            for (size_t i = 0; i < vocab_size; i++) {
                float probability = std::exp(
                    logits.at({b, s, i}) - max_val) / sum_exp;
                grad.at({b, s, i}) = probability * grad_scale;
            }

            grad.at({b, s, (size_t)(target)}) -= grad_scale;
        }
    }

    return grad;
}

size_t CPUBackend::ArgMax(const Tensor& tensor) const {
    if (tensor.GetSize() == 0) {
        throw std::runtime_error(
            "CPUBackend::ArgMax: tensor is empty"
        );
    }

    const float* data = tensor.Data();

    size_t max_index = 0;
    float max_value = data[0];

    for (size_t i = 1; i < tensor.GetSize(); ++i) {
        if (data[i] > max_value) {
            max_value = data[i];
            max_index = i;
        }
    }

    return max_index;
}

Tensor CPUBackend::Reshape(const Tensor& input,
        const std::vector<size_t>& new_shape) const {
    size_t old_size = input.GetSize();
    size_t new_size = 1;

    for (size_t dimension : new_shape) {
        new_size *= dimension;
    }

    if (old_size != new_size) {
        throw std::runtime_error("Reshape: incompatible tensor sizes");
    }

    Tensor result(new_shape);

    for (size_t i = 0; i < old_size; ++i) {
        result.data_[i] = input.data_[i];
    }

    return result;
}

void CPUBackend::AdamW(Tensor& parameter, Tensor& m, Tensor& v, const Tensor& gradient,
    float lr, float beta1, float beta2, float eps, float weight_decay, size_t step) {

    float bias_correction1 = 1.0f - std::pow(beta1, (float)(step));
    float bias_correction2 = 1.0f - std::pow(beta2, (float)(step));

    for (size_t i = 0; i < parameter.GetSize(); ++i) {
        float g = gradient.at(i);

        m.at(i) = beta1 * m.at(i) + (1.0f - beta1) * g;
        v.at(i) = beta2 * v.at(i) + (1.0f - beta2) * g * g;
        float m_hat = m.at(i) / bias_correction1;
        float v_hat = v.at(i) / bias_correction2;

        parameter.at(i) -= lr * (m_hat / (std::sqrt(v_hat) + eps) +
            weight_decay * parameter.at(i)
        );
    }
}