cmake_minimum_required(VERSION 3.10)

project(All_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

# ============================================================
# Apple Accelerate
# ============================================================

if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
endif()


# ============================================================
# Helper function
# ============================================================

function(setup_target target)

    target_include_directories(${target} PRIVATE
        .
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    if(APPLE)
        target_link_libraries(${target} PRIVATE
            ${ACCELERATE_FRAMEWORK}
        )
    else()
        target_link_libraries(${target} PRIVATE
            openblas
        )
    endif()

    target_link_libraries(${target} PRIVATE
        CUDA::cudart
        CUDA::cublas
    )

endfunction()


# ============================================================
# Attention test
# ============================================================

add_executable(attention_test

    ../../Engine/Tensor/tensor.cpp
    ../../Engine/Tensor/tensor_ops.cpp
    ../../Engine/Tensor/tensor_print.cpp

    ../../Engine/Tensor/backend.cpp
    ../../Engine/Tensor/cpu_backend.cpp
    ../../Engine/Tensor/matmul.cpp
    ../../Engine/Tensor/cuda_backend.cu

    # Autograd
    ../../Engine/Autograd/add_op.cpp
    ../../Engine/Autograd/mul_op.cpp
    ../../Engine/Autograd/sub_op.cpp
    ../../Engine/Autograd/square_op.cpp
    ../../Engine/Autograd/sum_op.cpp
    ../../Engine/Autograd/relu.cpp
    ../../Engine/Autograd/sigmoid.cpp
    ../../Engine/Autograd/tanh_op.cpp
    ../../Engine/Autograd/gelu.cpp
    ../../Engine/Autograd/mul_scalar_op.cpp
    ../../Engine/Autograd/concatenate_op.cpp
    ../../Engine/Autograd/transpose_op.cpp
    ../../Engine/Autograd/rope_op.cpp
    ../../Engine/Autograd/softmax_op.cpp
    ../../Engine/Autograd/reshape_op.cpp

    # Attention
    ../../Engine/Attention/attention.cpp
    ../../Engine/Attention/softmax.cpp
    ../../Engine/Attention/rope.cpp

    # Layers
    ../../Engine/Layers/linear_layer.cpp
    ../../Engine/Layers/feed_forward.cpp
    ../../Engine/Layers/embedding_layer.cpp
    ../../Engine/Layers/language_model.cpp
    ../../Engine/Layers/ce_loss.cpp


    ../../Engine/Autograd/rms_norm_op.cpp
    ../../Engine/Autograd/embedding_op.cpp

    # Normalization
    ../../Engine/Normalization/rms_norm.cpp

    # Transformer
    ../../Engine/Transformer/transformer_block.cpp
    ../../Engine/Transformer/transformer.cpp

    # Tokenizer
    ../../Engine/Tokenizer/bpe_tokenizer.cpp
    ../../Engine/Tokenizer/trie.cpp

    attention_test.cpp
)

setup_target(attention_test)


# ============================================================
# Autograd test
# ============================================================

add_executable(autograd_test

    ../../Engine/Tensor/backend.cpp
    ../../Engine/Tensor/cpu_backend.cpp
    ../../Engine/Tensor/matmul.cpp

    ../../Engine/Tensor/tensor.cpp
    ../../Engine/Tensor/tensor_ops.cpp
    ../../Engine/Tensor/tensor_print.cpp

    ../../Engine/Autograd/add_op.cpp
    ../../Engine/Autograd/mul_op.cpp
    ../../Engine/Autograd/square_op.cpp
    ../../Engine/Autograd/sub_op.cpp
    ../../Engine/Autograd/sum_op.cpp

    autograd_test.cpp
)

setup_target(autograd_test)


# ============================================================
# XOR test
# ============================================================

add_executable(xor_test

    ../../Engine/Tensor/backend.cpp
    ../../Engine/Tensor/cpu_backend.cpp
    ../../Engine/Tensor/matmul.cpp

    ../../Engine/Tensor/tensor.cpp
    ../../Engine/Tensor/tensor_ops.cpp
    ../../Engine/Tensor/tensor_print.cpp

    ../../Engine/Autograd/add_op.cpp
    ../../Engine/Autograd/mul_op.cpp
    ../../Engine/Autograd/square_op.cpp
    ../../Engine/Autograd/sub_op.cpp
    ../../Engine/Autograd/sum_op.cpp
    ../../Engine/Autograd/relu.cpp
    ../../Engine/Autograd/sigmoid.cpp
    ../../Engine/Autograd/tanh_op.cpp

    ../../Engine/Layers/mse_loss.cpp
    ../../Engine/Layers/linear_layer.cpp

    xor_test.cpp
)

setup_target(xor_test)


# ============================================================
# Log test
# ============================================================

add_executable(log_test

    ../../Engine/Tensor/backend.cpp
    ../../Engine/Tensor/cpu_backend.cpp
    ../../Engine/Tensor/matmul.cpp

    ../../Engine/Tensor/tensor.cpp
    ../../Engine/Tensor/tensor_ops.cpp
    ../../Engine/Tensor/tensor_print.cpp

    ../../Engine/Autograd/add_op.cpp
    ../../Engine/Autograd/mul_op.cpp
    ../../Engine/Autograd/square_op.cpp
    ../../Engine/Autograd/sub_op.cpp
    ../../Engine/Autograd/sum_op.cpp
    ../../Engine/Autograd/relu.cpp
    ../../Engine/Autograd/sigmoid.cpp
    ../../Engine/Autograd/tanh_op.cpp

    ../../Engine/Layers/mse_loss.cpp
    ../../Engine/Layers/linear_layer.cpp

    log_test.cpp
)

setup_target(log_test)


# ============================================================
# Feed Forward test
# ============================================================

add_executable(ffn_test

    ../../Engine/Tensor/backend.cpp
    ../../Engine/Tensor/cpu_backend.cpp
    ../../Engine/Tensor/matmul.cpp

    ../../Engine/Tensor/tensor.cpp
    ../../Engine/Tensor/tensor_ops.cpp
    ../../Engine/Tensor/tensor_print.cpp

    ../../Engine/Autograd/add_op.cpp
    ../../Engine/Autograd/mul_op.cpp
    ../../Engine/Autograd/square_op.cpp
    ../../Engine/Autograd/sub_op.cpp
    ../../Engine/Autograd/sum_op.cpp
    ../../Engine/Autograd/relu.cpp
    ../../Engine/Autograd/sigmoid.cpp
    ../../Engine/Autograd/tanh_op.cpp
    ../../Engine/Autograd/gelu.cpp

    ../../Engine/Layers/mse_loss.cpp
    ../../Engine/Layers/linear_layer.cpp
    ../../Engine/Layers/feed_forward.cpp

    ffn_test.cpp
)

setup_target(ffn_test)


# ============================================================
# Language Model test
# ============================================================

add_executable(lm_test

    ../../Engine/Tensor/backend.cpp
    ../../Engine/Tensor/cpu_backend.cpp
    ../../Engine/Tensor/matmul.cpp

    ../../Engine/Tensor/tensor.cpp
    ../../Engine/Tensor/tensor_ops.cpp
    ../../Engine/Tensor/tensor_print.cpp
    ../../Engine/Autograd/rms_norm_op.cpp
    ../../Engine/Layers/embedding_layer.cpp
    ../../Engine/Autograd/embedding_op.cpp

    # Autograd
    ../../Engine/Autograd/add_op.cpp
    ../../Engine/Autograd/mul_op.cpp
    ../../Engine/Autograd/sub_op.cpp
    ../../Engine/Autograd/square_op.cpp
    ../../Engine/Autograd/sum_op.cpp
    ../../Engine/Autograd/relu.cpp
    ../../Engine/Autograd/sigmoid.cpp
    ../../Engine/Autograd/tanh_op.cpp
    ../../Engine/Autograd/gelu.cpp
    ../../Engine/Autograd/mul_scalar_op.cpp
    ../../Engine/Autograd/concatenate_op.cpp
    ../../Engine/Autograd/transpose_op.cpp
    ../../Engine/Autograd/rope_op.cpp
    ../../Engine/Autograd/softmax_op.cpp

    # Attention
    ../../Engine/Attention/softmax.cpp
    ../../Engine/Attention/attention.cpp
    ../../Engine/Attention/rope.cpp

    # Layers
    ../../Engine/Layers/embedding_layer.cpp
    ../../Engine/Layers/linear_layer.cpp
    ../../Engine/Layers/feed_forward.cpp
    ../../Engine/Layers/mse_loss.cpp
    ../../Engine/Layers/language_model.cpp

    # Normalization
    ../../Engine/Normalization/rms_norm.cpp

    # Transformer
    ../../Engine/Transformer/transformer.cpp
    ../../Engine/Transformer/transformer_block.cpp

    language_model_test.cpp
)

setup_target(lm_test)


# ============================================================
# Save / Load test
# ============================================================

add_executable(sl_test

    ../../Engine/Tensor/backend.cpp
    ../../Engine/Tensor/cpu_backend.cpp
    ../../Engine/Tensor/matmul.cpp

    ../../Engine/Autograd/rms_norm_op.cpp
    ../../Engine/Autograd/embedding_op.cpp

    ../../Engine/Tensor/tensor.cpp
    ../../Engine/Tensor/tensor_ops.cpp
    ../../Engine/Tensor/tensor_print.cpp

    # Autograd
    ../../Engine/Autograd/add_op.cpp
    ../../Engine/Autograd/mul_op.cpp
    ../../Engine/Autograd/sub_op.cpp
    ../../Engine/Autograd/square_op.cpp
    ../../Engine/Autograd/sum_op.cpp
    ../../Engine/Autograd/relu.cpp
    ../../Engine/Autograd/sigmoid.cpp
    ../../Engine/Autograd/tanh_op.cpp
    ../../Engine/Autograd/gelu.cpp
    ../../Engine/Autograd/mul_scalar_op.cpp
    ../../Engine/Autograd/concatenate_op.cpp
    ../../Engine/Autograd/transpose_op.cpp
    ../../Engine/Autograd/rope_op.cpp
    ../../Engine/Autograd/softmax_op.cpp

    # Attention
    ../../Engine/Attention/softmax.cpp
    ../../Engine/Attention/attention.cpp
    ../../Engine/Attention/rope.cpp

    # Layers
    ../../Engine/Layers/embedding_layer.cpp
    ../../Engine/Layers/linear_layer.cpp
    ../../Engine/Layers/feed_forward.cpp
    ../../Engine/Layers/mse_loss.cpp
    ../../Engine/Layers/language_model.cpp

    # Normalization
    ../../Engine/Normalization/rms_norm.cpp

    # Transformer
    ../../Engine/Transformer/transformer.cpp
    ../../Engine/Transformer/transformer_block.cpp

    save_load_test.cpp
)

setup_target(sl_test)


# ============================================================
# Matmul test
# ============================================================

add_executable(matmul_test

    ../../Engine/Tensor/backend.cpp
    ../../Engine/Tensor/cpu_backend.cpp
    ../../Engine/Tensor/matmul.cpp

    ../../Engine/Tensor/tensor.cpp
    ../../Engine/Tensor/tensor_ops.cpp
    ../../Engine/Tensor/tensor_print.cpp

    matmul_test.cpp
)

setup_target(matmul_test)


# ============================================================
# Normalization test
# ============================================================

add_executable(normalization_test

    ../../Engine/Tensor/tensor.cpp
    ../../Engine/Tensor/tensor_ops.cpp
    ../../Engine/Tensor/tensor_print.cpp
    ../../Engine/Autograd/rms_norm_op.cpp
    ../../Engine/Tensor/backend.cpp
    ../../Engine/Tensor/cpu_backend.cpp
    ../../Engine/Tensor/matmul.cpp

    ../../Engine/Normalization/rms_norm.cpp

    rms_test.cpp
)

setup_target(normalization_test)


# ============================================================
# Tensor test
# ============================================================

add_executable(tensor_test

    ../../Engine/Tensor/tensor.cpp
    ../../Engine/Tensor/tensor_ops.cpp
    ../../Engine/Tensor/tensor_print.cpp
    ../../Engine/Tensor/backend.cpp
    ../../Engine/Tensor/cpu_backend.cpp
    ../../Engine/Tensor/matmul.cpp

    tensor_test.cpp
)

setup_target(tensor_test)


# ============================================================
# Broadcast test
# ============================================================

add_executable(broadcast_test

    ../../Engine/Tensor/tensor.cpp
    ../../Engine/Tensor/tensor_ops.cpp
    ../../Engine/Tensor/tensor_print.cpp
    ../../Engine/Tensor/backend.cpp
    ../../Engine/Tensor/cpu_backend.cpp
    ../../Engine/Tensor/matmul.cpp

    broadcast_test.cpp
)

setup_target(broadcast_test)


# ============================================================
# Tokenizer test
# ============================================================

add_executable(tokenizer_test

    ../../Engine/Tokenizer/bpe_tokenizer.cpp
    ../../Engine/Tokenizer/trie.cpp

    tokenizer_test.cpp
)

setup_target(tokenizer_test)


# ============================================================
# Transformer Block test
# ============================================================

add_executable(transformer_block_test

    ../../Engine/Tensor/tensor.cpp
    ../../Engine/Tensor/tensor_ops.cpp
    ../../Engine/Tensor/tensor_print.cpp

    ../../Engine/Tensor/backend.cpp
    ../../Engine/Tensor/cpu_backend.cpp
    ../../Engine/Tensor/matmul.cpp

    ../../Engine/Autograd/rms_norm_op.cpp

    # Autograd
    ../../Engine/Autograd/add_op.cpp
    ../../Engine/Autograd/mul_op.cpp
    ../../Engine/Autograd/sub_op.cpp
    ../../Engine/Autograd/square_op.cpp
    ../../Engine/Autograd/sum_op.cpp
    ../../Engine/Autograd/relu.cpp
    ../../Engine/Autograd/sigmoid.cpp
    ../../Engine/Autograd/tanh_op.cpp
    ../../Engine/Autograd/gelu.cpp
    ../../Engine/Autograd/mul_scalar_op.cpp
    ../../Engine/Autograd/concatenate_op.cpp
    ../../Engine/Autograd/transpose_op.cpp
    ../../Engine/Autograd/rope_op.cpp
    ../../Engine/Autograd/softmax_op.cpp

    # Attention
    ../../Engine/Attention/attention.cpp
    ../../Engine/Attention/softmax.cpp
    ../../Engine/Attention/rope.cpp

    # Layers
    ../../Engine/Layers/linear_layer.cpp
    ../../Engine/Layers/feed_forward.cpp

    # Normalization
    ../../Engine/Normalization/rms_norm.cpp

    # Transformer
    ../../Engine/Transformer/transformer_block.cpp

    transformer_block_test.cpp
)

setup_target(transformer_block_test)


# ============================================================
# Transformer test
# ============================================================

add_executable(transformer_test

    ../../Engine/Tensor/tensor.cpp
    ../../Engine/Tensor/tensor_ops.cpp
    ../../Engine/Tensor/tensor_print.cpp

    ../../Engine/Tensor/backend.cpp
    ../../Engine/Tensor/cpu_backend.cpp
    ../../Engine/Tensor/matmul.cpp

    # Autograd
    ../../Engine/Autograd/add_op.cpp
    ../../Engine/Autograd/mul_op.cpp
    ../../Engine/Autograd/sub_op.cpp
    ../../Engine/Autograd/square_op.cpp
    ../../Engine/Autograd/sum_op.cpp
    ../../Engine/Autograd/relu.cpp
    ../../Engine/Autograd/sigmoid.cpp
    ../../Engine/Autograd/tanh_op.cpp
    ../../Engine/Autograd/gelu.cpp
    ../../Engine/Autograd/mul_scalar_op.cpp
    ../../Engine/Autograd/concatenate_op.cpp
    ../../Engine/Autograd/transpose_op.cpp
    ../../Engine/Autograd/rope_op.cpp
    ../../Engine/Autograd/softmax_op.cpp

    ../../Engine/Autograd/rms_norm_op.cpp

    # Attention
    ../../Engine/Attention/attention.cpp
    ../../Engine/Attention/softmax.cpp
    ../../Engine/Attention/rope.cpp

    # Layers
    ../../Engine/Layers/linear_layer.cpp
    ../../Engine/Layers/feed_forward.cpp

    # Normalization
    ../../Engine/Normalization/rms_norm.cpp

    # Transformer
    ../../Engine/Transformer/transformer_block.cpp
    ../../Engine/Transformer/transformer.cpp

    transformer_test.cpp
)

setup_target(transformer_test)


# ============================================================
# Final model test
# ============================================================

add_executable(final_model_test

    ../../Engine/Tensor/tensor.cpp
    ../../Engine/Tensor/tensor_ops.cpp
    ../../Engine/Tensor/tensor_print.cpp

    ../../Engine/Tensor/backend.cpp
    ../../Engine/Tensor/cpu_backend.cpp
    ../../Engine/Tensor/matmul.cpp
    ../../Engine/Tensor/cuda_backend.cu

    # Autograd
    ../../Engine/Autograd/add_op.cpp
    ../../Engine/Autograd/mul_op.cpp
    ../../Engine/Autograd/sub_op.cpp
    ../../Engine/Autograd/square_op.cpp
    ../../Engine/Autograd/sum_op.cpp
    ../../Engine/Autograd/relu.cpp
    ../../Engine/Autograd/sigmoid.cpp
    ../../Engine/Autograd/tanh_op.cpp
    ../../Engine/Autograd/gelu.cpp
    ../../Engine/Autograd/mul_scalar_op.cpp
    ../../Engine/Autograd/concatenate_op.cpp
    ../../Engine/Autograd/transpose_op.cpp
    ../../Engine/Autograd/rope_op.cpp
    ../../Engine/Autograd/softmax_op.cpp
    ../../Engine/Autograd/reshape_op.cpp

    # Attention
    ../../Engine/Attention/attention.cpp
    ../../Engine/Attention/softmax.cpp
    ../../Engine/Attention/rope.cpp

    # Layers
    ../../Engine/Layers/linear_layer.cpp
    ../../Engine/Layers/feed_forward.cpp
    ../../Engine/Layers/embedding_layer.cpp
    ../../Engine/Layers/language_model.cpp
    ../../Engine/Layers/ce_loss.cpp


    ../../Engine/Autograd/rms_norm_op.cpp
    ../../Engine/Autograd/embedding_op.cpp

    # Normalization
    ../../Engine/Normalization/rms_norm.cpp

    # Transformer
    ../../Engine/Transformer/transformer_block.cpp
    ../../Engine/Transformer/transformer.cpp

    # Tokenizer
    ../../Engine/Tokenizer/bpe_tokenizer.cpp
    ../../Engine/Tokenizer/trie.cpp

    final_model_test.cpp
)

setup_target(final_model_test)


# ============================================================
# General tests
# ============================================================

add_executable(tests

    ../../Engine/Tensor/tensor.cpp
    ../../Engine/Tensor/tensor_ops.cpp
    ../../Engine/Tensor/tensor_print.cpp

    ../../Engine/Tensor/backend.cpp
    ../../Engine/Tensor/cpu_backend.cpp
    ../../Engine/Tensor/matmul.cpp
    ../../Engine/Tensor/cuda_backend.cu

    # Autograd
    ../../Engine/Autograd/add_op.cpp
    ../../Engine/Autograd/mul_op.cpp
    ../../Engine/Autograd/sub_op.cpp
    ../../Engine/Autograd/square_op.cpp
    ../../Engine/Autograd/sum_op.cpp
    ../../Engine/Autograd/relu.cpp
    ../../Engine/Autograd/sigmoid.cpp
    ../../Engine/Autograd/tanh_op.cpp
    ../../Engine/Autograd/gelu.cpp
    ../../Engine/Autograd/mul_scalar_op.cpp
    ../../Engine/Autograd/concatenate_op.cpp
    ../../Engine/Autograd/transpose_op.cpp
    ../../Engine/Autograd/rope_op.cpp
    ../../Engine/Autograd/softmax_op.cpp
    ../../Engine/Autograd/reshape_op.cpp

    # Attention
    ../../Engine/Attention/attention.cpp
    ../../Engine/Attention/softmax.cpp
    ../../Engine/Attention/rope.cpp

    # Layers
    ../../Engine/Layers/linear_layer.cpp
    ../../Engine/Layers/feed_forward.cpp
    ../../Engine/Layers/embedding_layer.cpp
    ../../Engine/Layers/language_model.cpp
    ../../Engine/Layers/ce_loss.cpp


    ../../Engine/Autograd/rms_norm_op.cpp
    ../../Engine/Autograd/embedding_op.cpp

    # Normalization
    ../../Engine/Normalization/rms_norm.cpp

    # Transformer
    ../../Engine/Transformer/transformer_block.cpp
    ../../Engine/Transformer/transformer.cpp

    # Tokenizer
    ../../Engine/Tokenizer/bpe_tokenizer.cpp
    ../../Engine/Tokenizer/trie.cpp

    tests.cpp
)

setup_target(tests)


# ============================================================
# Text test
# ============================================================

add_executable(text_test

    ../../Engine/Tensor/tensor.cpp
    ../../Engine/Tensor/tensor_ops.cpp
    ../../Engine/Tensor/tensor_print.cpp

    ../../Engine/Tensor/backend.cpp
    ../../Engine/Tensor/cpu_backend.cpp
    ../../Engine/Tensor/matmul.cpp

    # Autograd
    ../../Engine/Autograd/add_op.cpp
    ../../Engine/Autograd/mul_op.cpp
    ../../Engine/Autograd/sub_op.cpp
    ../../Engine/Autograd/square_op.cpp
    ../../Engine/Autograd/sum_op.cpp
    ../../Engine/Autograd/relu.cpp
    ../../Engine/Autograd/sigmoid.cpp
    ../../Engine/Autograd/tanh_op.cpp
    ../../Engine/Autograd/gelu.cpp
    ../../Engine/Autograd/mul_scalar_op.cpp
    ../../Engine/Autograd/concatenate_op.cpp
    ../../Engine/Autograd/transpose_op.cpp
    ../../Engine/Autograd/rope_op.cpp
    ../../Engine/Autograd/softmax_op.cpp

    ../../Engine/Autograd/rms_norm_op.cpp
    ../../Engine/Autograd/embedding_op.cpp

    # Attention
    ../../Engine/Attention/attention.cpp
    ../../Engine/Attention/softmax.cpp
    ../../Engine/Attention/rope.cpp

    # Layers
    ../../Engine/Layers/linear_layer.cpp
    ../../Engine/Layers/feed_forward.cpp
    ../../Engine/Layers/embedding_layer.cpp
    ../../Engine/Layers/language_model.cpp

    # Normalization
    ../../Engine/Normalization/rms_norm.cpp

    # Transformer
    ../../Engine/Transformer/transformer_block.cpp
    ../../Engine/Transformer/transformer.cpp

    # Tokenizer
    ../../Engine/Tokenizer/bpe_tokenizer.cpp
    ../../Engine/Tokenizer/trie.cpp

    text_test.cpp
)

setup_target(text_test)