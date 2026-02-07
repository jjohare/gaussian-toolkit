/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/cuda_stream_context.hpp"

namespace lfs::core {

    static thread_local cudaStream_t current_cuda_stream = nullptr;

    cudaStream_t getCurrentCUDAStream() {
        return current_cuda_stream;
    }

    void setCurrentCUDAStream(cudaStream_t stream) {
        current_cuda_stream = stream;
    }

    namespace {
        struct ThreadLocalEvent {
            cudaEvent_t event = nullptr;
            ~ThreadLocalEvent() {
                if (event)
                    cudaEventDestroy(event);
            }
        };

        cudaError_t stream_is_nonblocking(cudaStream_t stream, bool* out) {
            if (stream == nullptr) {
                *out = false;
                return cudaSuccess;
            }
            unsigned int flags = 0;
            cudaError_t err = cudaStreamGetFlags(stream, &flags);
            if (err != cudaSuccess)
                return err;
            *out = (flags & cudaStreamNonBlocking) != 0;
            return cudaSuccess;
        }
    } // namespace

    cudaError_t waitForCUDAStream(cudaStream_t consumer_stream, cudaStream_t producer_stream) {
        if (producer_stream == consumer_stream)
            return cudaSuccess;

        if (consumer_stream == nullptr || producer_stream == nullptr) {
            bool consumer_nonblocking = false;
            bool producer_nonblocking = false;
            cudaError_t err = stream_is_nonblocking(consumer_stream, &consumer_nonblocking);
            if (err != cudaSuccess)
                return err;
            err = stream_is_nonblocking(producer_stream, &producer_nonblocking);
            if (err != cudaSuccess)
                return err;
            if (!consumer_nonblocking && !producer_nonblocking)
                return cudaSuccess;
        }

        static thread_local ThreadLocalEvent tls_event;
        if (!tls_event.event) {
            cudaError_t err = cudaEventCreateWithFlags(&tls_event.event, cudaEventDisableTiming);
            if (err != cudaSuccess)
                return err;
        }

        cudaError_t err = cudaEventRecord(tls_event.event, producer_stream);
        if (err == cudaSuccess)
            err = cudaStreamWaitEvent(consumer_stream, tls_event.event, 0);
        return err;
    }

} // namespace lfs::core
