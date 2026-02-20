/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#ifdef __linux__

#include "frame_share/frame_share_sink.hpp"
#include <cstdint>

namespace lfs::vis::shm {
    struct ShmHeader;
}

namespace lfs::vis {

    class ShmSink : public IFrameShareSink {
    public:
        ShmSink() = default;
        ~ShmSink() override;

        bool start(const std::string& name, int width, int height) override;
        void stop() override;
        void sendFrame(unsigned int gl_texture_id, int width, int height) override;
        bool isActive() const override { return header_ != nullptr; }

    private:
        void createPBOs(int width, int height);
        void destroyPBOs();

        shm::ShmHeader* header_ = nullptr;
        int shm_fd_ = -1;
        size_t shm_size_ = 0;
        std::string shm_name_;
        int max_width_ = 0;
        int max_height_ = 0;

        unsigned int pbos_[2] = {0, 0};
        int pbo_index_ = 0;
        bool pbo_initialized_ = false;
        int pbo_width_ = 0;
        int pbo_height_ = 0;

        int last_written_slot_ = -1;
        uint64_t frame_counter_ = 0;
    };

} // namespace lfs::vis

#endif // __linux__
