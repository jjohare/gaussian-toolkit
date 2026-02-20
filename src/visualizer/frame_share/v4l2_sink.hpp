/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#ifdef __linux__

#include "frame_share/frame_share_sink.hpp"
#include <cstdint>
#include <vector>

namespace lfs::vis {

    class V4L2Sink : public IFrameShareSink {
    public:
        V4L2Sink() = default;
        ~V4L2Sink() override;

        bool start(const std::string& name, int width, int height) override;
        void stop() override;
        void sendFrame(unsigned int gl_texture_id, int width, int height) override;
        bool isActive() const override { return fd_ >= 0; }

    private:
        int findLoopbackDevice() const;
        void createPBOs(int width, int height);
        void destroyPBOs();

        int fd_ = -1;
        int dev_width_ = 0;
        int dev_height_ = 0;

        unsigned int pbos_[2] = {0, 0};
        int pbo_index_ = 0;
        bool pbo_initialized_ = false;
        int pbo_width_ = 0;
        int pbo_height_ = 0;

        std::vector<uint8_t> flip_buffer_;
    };

} // namespace lfs::vis

#endif // __linux__
