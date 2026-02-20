/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#ifdef LFS_SPOUT_ENABLED

#include "frame_share/frame_share_sink.hpp"
#include <SpoutGL/SpoutSender.h>

namespace lfs::vis {

    class SpoutSink : public IFrameShareSink {
    public:
        SpoutSink() = default;
        ~SpoutSink() override;

        bool start(const std::string& name, int width, int height) override;
        void stop() override;
        void sendFrame(unsigned int gl_texture_id, int width, int height) override;
        bool isActive() const override { return active_; }

    private:
        SpoutSender sender_;
        bool active_ = false;
    };

} // namespace lfs::vis

#endif // LFS_SPOUT_ENABLED
