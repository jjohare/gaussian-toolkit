/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "frame_share/frame_share_manager.hpp"
#include "core/logger.hpp"
#include "frame_share/frame_share_sink.hpp"
#include <glad/glad.h>

#ifdef LFS_SPOUT_ENABLED
#include "frame_share/spout_sink.hpp"
#endif

#if defined(__linux__)
#include "frame_share/shm_sink.hpp"
#include "frame_share/v4l2_sink.hpp"
#endif

namespace lfs::vis {

    FrameShareManager::FrameShareManager() {
#ifdef LFS_SPOUT_ENABLED
        backend_ = FrameShareBackend::Spout;
#elif defined(__linux__)
        backend_ = FrameShareBackend::SharedMemory;
#endif
        status_message_ = "Inactive";
        next_retry_time_ = std::chrono::steady_clock::now();
    }

    FrameShareManager::~FrameShareManager() {
        destroySink();
        destroyCaptureTexture();
    }

    void FrameShareManager::setBackend(FrameShareBackend backend) {
        if (backend_ == backend)
            return;

        const bool was_enabled = enabled_;
        if (was_enabled)
            setEnabled(false);

        backend_ = backend;

        if (was_enabled)
            setEnabled(true);
    }

    void FrameShareManager::setEnabled(bool enabled) {
        if (enabled_ == enabled)
            return;
        enabled_ = enabled;

        if (enabled_) {
            next_retry_time_ = std::chrono::steady_clock::now();
            createSink(last_width_, last_height_);
        } else {
            destroySink();
            status_message_ = "Inactive";
        }
    }

    bool FrameShareManager::isActive() const {
        return sink_ && sink_->isActive();
    }

    void FrameShareManager::setSenderName(const std::string& name) {
        if (sender_name_ == name)
            return;

        const bool was_enabled = enabled_;
        if (was_enabled)
            setEnabled(false);

        sender_name_ = name;

        if (was_enabled)
            setEnabled(true);
    }

    void FrameShareManager::onFrame(unsigned int gl_texture_id, int width, int height) {
        if (!enabled_ || gl_texture_id == 0 || width <= 0 || height <= 0)
            return;

        last_width_ = width;
        last_height_ = height;

        if (!sink_ || !sink_->isActive()) {
            const auto now = std::chrono::steady_clock::now();
            if (now < next_retry_time_) {
                return;
            }
            createSink(width, height);
            if (!sink_ || !sink_->isActive()) {
                return;
            }
        }

        if (width != sink_width_ || height != sink_height_) {
            LOG_INFO("Frame share: resolution changed to {}x{}, restarting sink", width, height);
            createSink(width, height);
            if (!sink_ || !sink_->isActive())
                return;
        }

        sink_->sendFrame(gl_texture_id, width, height);
    }

    void FrameShareManager::onFrameFromViewport(int viewport_x, int viewport_y, int width, int height) {
        if (!enabled_ || width <= 0 || height <= 0)
            return;

        const auto now = std::chrono::steady_clock::now();
        if (!sink_ && now < next_retry_time_) {
            return;
        }

        ensureCaptureTexture(width, height);
        if (capture_texture_ == 0) {
            return;
        }

        GLint prev_read_framebuffer = 0;
        GLint prev_read_buffer = 0;
        GLint double_buffered = GL_TRUE;
        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &prev_read_framebuffer);
        glGetIntegerv(GL_READ_BUFFER, &prev_read_buffer);
        glGetIntegerv(GL_DOUBLEBUFFER, &double_buffered);

        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        glReadBuffer(double_buffered ? GL_BACK : GL_FRONT);
        glBindTexture(GL_TEXTURE_2D, capture_texture_);
        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, viewport_x, viewport_y, width, height);
        glBindTexture(GL_TEXTURE_2D, 0);

        glBindFramebuffer(GL_READ_FRAMEBUFFER, static_cast<GLuint>(prev_read_framebuffer));
        glReadBuffer(static_cast<GLenum>(prev_read_buffer));

        onFrame(capture_texture_, width, height);
    }

    int FrameShareManager::connectedReceivers() const {
        return sink_ ? sink_->connectedReceivers() : 0;
    }

    void FrameShareManager::createSink(int width, int height) {
        destroySink();

        switch (backend_) {
#ifdef LFS_SPOUT_ENABLED
        case FrameShareBackend::Spout:
            sink_ = std::make_unique<SpoutSink>();
            break;
#endif
#if defined(__linux__)
        case FrameShareBackend::SharedMemory:
            sink_ = std::make_unique<ShmSink>();
            break;
        case FrameShareBackend::V4L2:
            sink_ = std::make_unique<V4L2Sink>();
            break;
#endif
        case FrameShareBackend::None:
            status_message_ = "No backend selected";
            next_retry_time_ = std::chrono::steady_clock::now() + std::chrono::seconds(1);
            return;
        }

        if (sink_) {
            if (sink_->start(sender_name_, width, height)) {
                sink_width_ = width;
                sink_height_ = height;
                status_message_ = "Active";
                LOG_INFO("Frame share started: backend={}, name='{}'",
                         static_cast<int>(backend_), sender_name_);
            } else {
                status_message_ = "Failed to start";
                LOG_ERROR("Frame share failed to start");
                sink_.reset();
                sink_width_ = 0;
                sink_height_ = 0;
                next_retry_time_ = std::chrono::steady_clock::now() + std::chrono::seconds(1);
            }
        }
    }

    void FrameShareManager::destroySink() {
        if (sink_) {
            sink_->stop();
            sink_.reset();
        }
        sink_width_ = 0;
        sink_height_ = 0;
    }

    void FrameShareManager::ensureCaptureTexture(int width, int height) {
        if (capture_texture_ == 0) {
            glGenTextures(1, &capture_texture_);
            if (capture_texture_ == 0) {
                LOG_ERROR("Frame share: failed to create capture texture");
                return;
            }

            glBindTexture(GL_TEXTURE_2D, capture_texture_);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        } else {
            glBindTexture(GL_TEXTURE_2D, capture_texture_);
        }

        if (capture_width_ != width || capture_height_ != height) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            capture_width_ = width;
            capture_height_ = height;
        }

        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void FrameShareManager::destroyCaptureTexture() {
        if (capture_texture_ != 0) {
            glDeleteTextures(1, &capture_texture_);
            capture_texture_ = 0;
        }
        capture_width_ = 0;
        capture_height_ = 0;
    }

} // namespace lfs::vis
