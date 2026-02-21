/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#ifdef __linux__

#include <glad/glad.h>

#include "core/logger.hpp"
#include "frame_share/v4l2_sink.hpp"

#include <cassert>
#include <cstring>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

namespace lfs::vis {

    static constexpr int V4L2_CHANNELS_RGB = 3;

    V4L2Sink::~V4L2Sink() {
        stop();
    }

    int V4L2Sink::findLoopbackDevice() const {
        for (int i = 0; i < 64; ++i) {
            const std::string path = "/dev/video" + std::to_string(i);
            int fd = open(path.c_str(), O_WRONLY | O_NONBLOCK);
            if (fd < 0)
                continue;

            struct v4l2_capability cap {};
            if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == 0) {
                const std::string driver(reinterpret_cast<const char*>(cap.driver));
                if (driver.find("v4l2 loopback") != std::string::npos ||
                    driver.find("v4l2loopback") != std::string::npos) {
                    close(fd);
                    return i;
                }
            }
            close(fd);
        }
        return -1;
    }

    bool V4L2Sink::start(const std::string& /*name*/, int width, int height) {
        stop();

        const int dev_idx = findLoopbackDevice();
        if (dev_idx < 0) {
            LOG_ERROR("Frame share: no v4l2loopback device found");
            return false;
        }

        const std::string path = "/dev/video" + std::to_string(dev_idx);
        fd_ = open(path.c_str(), O_WRONLY);
        if (fd_ < 0) {
            LOG_ERROR("Frame share: failed to open {}: {}", path, strerror(errno));
            return false;
        }

        dev_width_ = (width > 0) ? width : 1920;
        dev_height_ = (height > 0) ? height : 1080;

        struct v4l2_format fmt {};
        fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
        fmt.fmt.pix.width = static_cast<uint32_t>(dev_width_);
        fmt.fmt.pix.height = static_cast<uint32_t>(dev_height_);
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
        fmt.fmt.pix.sizeimage = static_cast<uint32_t>(dev_width_ * dev_height_ * V4L2_CHANNELS_RGB);
        fmt.fmt.pix.field = V4L2_FIELD_NONE;

        if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
            LOG_ERROR("Frame share: VIDIOC_S_FMT failed: {}", strerror(errno));
            close(fd_);
            fd_ = -1;
            return false;
        }

        dev_width_ = static_cast<int>(fmt.fmt.pix.width);
        dev_height_ = static_cast<int>(fmt.fmt.pix.height);

        LOG_INFO("Frame share: v4l2 opened {} ({}x{})", path, dev_width_, dev_height_);
        return true;
    }

    void V4L2Sink::stop() {
        destroyPBOs();
        flip_buffer_.clear();

        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
    }

    void V4L2Sink::sendFrame(unsigned int gl_texture_id, int width, int height) {
        assert(fd_ >= 0);
        assert(width > 0 && height > 0);

        if (width != pbo_width_ || height != pbo_height_) {
            createPBOs(width, height);
        }

        const size_t rgb_size = static_cast<size_t>(width) * height * V4L2_CHANNELS_RGB;

        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos_[pbo_index_]);
        glBindTexture(GL_TEXTURE_2D, gl_texture_id);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        const int read_pbo = 1 - pbo_index_;
        if (pbo_initialized_) {
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos_[read_pbo]);
            auto* src = static_cast<const uint8_t*>(
                glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, static_cast<GLsizeiptr>(rgb_size), GL_MAP_READ_BIT));

            if (src) {
                const size_t row_bytes = static_cast<size_t>(width) * V4L2_CHANNELS_RGB;
                flip_buffer_.resize(rgb_size);
                for (int y = 0; y < height; ++y) {
                    std::memcpy(flip_buffer_.data() + y * row_bytes, src + (height - 1 - y) * row_bytes, row_bytes);
                }
                glUnmapBuffer(GL_PIXEL_PACK_BUFFER);

                ssize_t written = write(fd_, flip_buffer_.data(), rgb_size);
                if (written < 0) {
                    LOG_ERROR("Frame share: v4l2 write failed: {}", strerror(errno));
                }
            }
        }

        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        pbo_index_ = 1 - pbo_index_;
        pbo_initialized_ = true;
    }

    void V4L2Sink::createPBOs(int width, int height) {
        destroyPBOs();

        const auto buf_size = static_cast<GLsizeiptr>(width) * height * V4L2_CHANNELS_RGB;
        glGenBuffers(2, pbos_);
        for (int i = 0; i < 2; ++i) {
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos_[i]);
            glBufferData(GL_PIXEL_PACK_BUFFER, buf_size, nullptr, GL_STREAM_READ);
        }
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

        pbo_width_ = width;
        pbo_height_ = height;
        pbo_index_ = 0;
        pbo_initialized_ = false;
    }

    void V4L2Sink::destroyPBOs() {
        if (pbos_[0] != 0) {
            glDeleteBuffers(2, pbos_);
            pbos_[0] = pbos_[1] = 0;
        }
        pbo_initialized_ = false;
    }

} // namespace lfs::vis

#endif // __linux__
