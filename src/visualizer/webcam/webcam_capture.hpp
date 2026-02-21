/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#ifdef __linux__

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace lfs::io {

    enum class CaptureFormat { YUYV,
                               MJPEG };

    struct WebcamDeviceInfo {
        std::string path;
        std::string name;
        int index = -1;
    };

    class WebcamCapture {
    public:
        WebcamCapture() = default;
        ~WebcamCapture();

        WebcamCapture(const WebcamCapture&) = delete;
        WebcamCapture& operator=(const WebcamCapture&) = delete;

        bool open(const std::string& device_path, int width, int height);
        void close();

        bool isCapturing() const { return capturing_.load(); }
        bool getLatestFrame(const uint8_t*& data, int& out_width, int& out_height);
        bool hasNewFrame() const { return new_frame_available_.load(); }

        int frameWidth() const { return frame_width_; }
        int frameHeight() const { return frame_height_; }
        CaptureFormat format() const { return format_; }

        static std::vector<WebcamDeviceInfo> enumerateDevices();

    private:
        void captureThread();
        static void convertYUYVtoRGB(const uint8_t* yuyv, uint8_t* rgb, int width, int height);
        static bool decodeMJPEG(const uint8_t* data, size_t size,
                                uint8_t* dst, int width, int height);

        int fd_ = -1;
        std::atomic<bool> capturing_{false};
        std::thread capture_thread_;
        CaptureFormat format_ = CaptureFormat::YUYV;

        // Double-buffered RGB frames
        std::vector<uint8_t> frame_buffers_[2];
        std::atomic<int> front_buffer_{0};
        std::atomic<bool> new_frame_available_{false};

        int frame_width_ = 0;
        int frame_height_ = 0;

        // v4l2 MMAP buffers
        struct MMapBuffer {
            void* start = nullptr;
            size_t length = 0;
        };
        static constexpr int NUM_BUFFERS = 4;
        MMapBuffer mmap_buffers_[NUM_BUFFERS] = {};
        int num_mmap_buffers_ = 0;
    };

} // namespace lfs::io

#endif // __linux__
