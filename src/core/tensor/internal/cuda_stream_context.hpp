/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <core/export.hpp>
#include <cuda_runtime.h>

namespace lfs::core {

    LFS_CORE_API cudaStream_t getCurrentCUDAStream();
    LFS_CORE_API void setCurrentCUDAStream(cudaStream_t stream);

    /**
     * RAII guard for temporarily setting the current CUDA stream
     * (PyTorch's CUDAStreamGuard pattern)
     *
     * Usage in DataLoader worker:
     *   cudaStream_t worker_stream;
     *   cudaStreamCreate(&worker_stream);
     *   {
     *       CUDAStreamGuard guard(worker_stream);
     *       // All tensor operations in this scope use worker_stream
     *       auto image = load_image();
     *       image = image.to(Device::CUDA);  // Uses worker_stream!
     *       image = preprocess(image);        // Uses worker_stream!
     *   }
     *   // Stream restored to previous value
     */
    class CUDAStreamGuard {
    public:
        explicit CUDAStreamGuard(cudaStream_t stream)
            : prev_stream_(getCurrentCUDAStream()) {
            setCurrentCUDAStream(stream);
        }

        ~CUDAStreamGuard() {
            setCurrentCUDAStream(prev_stream_);
        }

        // Delete copy/move
        CUDAStreamGuard(const CUDAStreamGuard&) = delete;
        CUDAStreamGuard& operator=(const CUDAStreamGuard&) = delete;
        CUDAStreamGuard(CUDAStreamGuard&&) = delete;
        CUDAStreamGuard& operator=(CUDAStreamGuard&&) = delete;

    private:
        cudaStream_t prev_stream_;
    };

    // Resolve explicit stream or fall back to current thread-local stream.
    inline cudaStream_t resolveCUDAStream(cudaStream_t stream = nullptr) {
        return stream ? stream : getCurrentCUDAStream();
    }

    LFS_CORE_API cudaError_t waitForCUDAStream(cudaStream_t consumer_stream, cudaStream_t producer_stream);

    // Sync only the relevant stream when crossing to host-visible API boundaries.
    inline cudaError_t synchronizeCUDAStream(cudaStream_t stream) {
        return cudaStreamSynchronize(stream);
    }

} // namespace lfs::core
