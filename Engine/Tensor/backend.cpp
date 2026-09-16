#include "backend.h"
#include "cpu_backend.h"

#if ENABLE_CUDA
#include "cuda_backend.h"
#endif

Backend& GetBackend(Device device) {
    switch (device) {
        case Device::CPU: {
            static CPUBackend cpu_backend;
            return cpu_backend;
        }
        
        case Device::CUDA: {

#if ENABLE_CUDA

            static CUDABackend cuda_backend;
            return cuda_backend;

#else

            throw std::runtime_error(
                "CUDA backend requested, "
                "but CUDA support is disabled"
            );

#endif
        }
    }

    throw std::runtime_error("Unknown device");
}