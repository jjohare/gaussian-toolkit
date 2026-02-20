/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#ifdef LFS_SPOUT_ENABLED

#include "frame_share/spout_sink.hpp"
#include "core/logger.hpp"

#include <glad/glad.h>

namespace lfs::vis {

    SpoutSink::~SpoutSink() {
        stop();
    }

    bool SpoutSink::start(const std::string& name, int /*width*/, int /*height*/) {
        stop();
        sender_.SetSenderName(name.c_str());
        active_ = true;
        LOG_INFO("Frame share: Spout sender '{}' created", name);
        return true;
    }

    void SpoutSink::stop() {
        if (active_) {
            sender_.ReleaseSender();
            active_ = false;
        }
    }

    void SpoutSink::sendFrame(unsigned int gl_texture_id, int width, int height) {
        if (!active_)
            return;
        sender_.SendTexture(gl_texture_id, GL_TEXTURE_2D, width, height);
    }

} // namespace lfs::vis

#endif // LFS_SPOUT_ENABLED
