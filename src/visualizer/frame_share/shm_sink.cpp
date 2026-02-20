/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#ifdef __linux__

#include <glad/glad.h>

#include "core/logger.hpp"
#include "frame_share/shm_protocol.hpp"
#include "frame_share/shm_sink.hpp"

#include <cassert>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <new>
#include <sys/mman.h>
#include <unistd.h>

namespace lfs::vis {

    using namespace shm;

    static constexpr int MAX_SHM_DIM = 7680; // 8K max

    ShmSink::~ShmSink() {
        stop();
    }

    bool ShmSink::start(const std::string& name, int width, int height) {
        stop();

        max_width_ = (width > 0) ? width : 1920;
        max_height_ = (height > 0) ? height : 1080;
        max_width_ = std::min(max_width_, MAX_SHM_DIM);
        max_height_ = std::min(max_height_, MAX_SHM_DIM);

        shm_name_ = "/" + name;
        for (auto& c : shm_name_) {
            if (c == ' ')
                c = '-';
        }

        shm_size_ = shmTotalSize(max_width_, max_height_);

        shm_fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0666);
        if (shm_fd_ < 0) {
            LOG_ERROR("Frame share: shm_open failed for '{}': {}", shm_name_, strerror(errno));
            return false;
        }

        if (ftruncate(shm_fd_, static_cast<off_t>(shm_size_)) != 0) {
            LOG_ERROR("Frame share: ftruncate failed: {}", strerror(errno));
            close(shm_fd_);
            shm_fd_ = -1;
            shm_unlink(shm_name_.c_str());
            return false;
        }

        void* mapped = mmap(nullptr, shm_size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
        if (mapped == MAP_FAILED) {
            LOG_ERROR("Frame share: mmap failed: {}", strerror(errno));
            close(shm_fd_);
            shm_fd_ = -1;
            shm_unlink(shm_name_.c_str());
            return false;
        }

        header_ = new (mapped) ShmHeader{};
        header_->magic = SHM_MAGIC;
        header_->version = SHM_VERSION;
        header_->max_width = max_width_;
        header_->max_height = max_height_;
        header_->channels = SHM_CHANNELS;
        header_->writer_frame_id.store(0, std::memory_order_release);
        header_->latest_slot.store(0, std::memory_order_release);
        header_->reader_active_slot.store(SHM_NUM_SLOTS, std::memory_order_release);

        for (int i = 0; i < SHM_NUM_SLOTS; ++i) {
            header_->slots[i] = {};
        }

        last_written_slot_ = -1;
        frame_counter_ = 0;

        LOG_INFO("Frame share: shm created '{}' ({}x{}, {:.1f} MB)",
                 shm_name_, max_width_, max_height_, shm_size_ / (1024.0 * 1024.0));
        return true;
    }

    void ShmSink::stop() {
        destroyPBOs();

        if (header_) {
            header_->~ShmHeader();
            munmap(header_, shm_size_);
            header_ = nullptr;
        }
        if (shm_fd_ >= 0) {
            close(shm_fd_);
            shm_fd_ = -1;
        }
        if (!shm_name_.empty()) {
            shm_unlink(shm_name_.c_str());
            shm_name_.clear();
        }
        shm_size_ = 0;
    }

    void ShmSink::sendFrame(unsigned int gl_texture_id, int width, int height) {
        assert(header_);
        assert(width > 0 && height > 0);
        assert(width <= max_width_ && height <= max_height_);

        if (width != pbo_width_ || height != pbo_height_) {
            createPBOs(width, height);
        }

        const size_t data_size = static_cast<size_t>(width) * height * SHM_CHANNELS;

        // Ping-pong PBO: kick readback into current PBO
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos_[pbo_index_]);
        glBindTexture(GL_TEXTURE_2D, gl_texture_id);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        // Map the OTHER PBO (from previous frame) — should be complete
        const int read_pbo = 1 - pbo_index_;
        if (pbo_initialized_) {
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos_[read_pbo]);
            auto* src = static_cast<const uint8_t*>(
                glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, static_cast<GLsizeiptr>(data_size), GL_MAP_READ_BIT));

            if (src) {
                const uint32_t reader_slot = header_->reader_active_slot.load(std::memory_order_acquire);
                int write_slot = (last_written_slot_ + 1) % SHM_NUM_SLOTS;
                if (write_slot == static_cast<int>(reader_slot)) {
                    write_slot = (write_slot + 1) % SHM_NUM_SLOTS;
                }

                uint8_t* dst = shmSlotData(header_, write_slot, max_width_, max_height_);
                const size_t row_bytes = static_cast<size_t>(width) * SHM_CHANNELS;
                for (int y = 0; y < height; ++y) {
                    std::memcpy(dst + y * row_bytes, src + (height - 1 - y) * row_bytes, row_bytes);
                }

                auto now = std::chrono::steady_clock::now();
                auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();

                ++frame_counter_;
                auto& slot = header_->slots[write_slot];
                slot.width = width;
                slot.height = height;
                slot.stride = width * SHM_CHANNELS;
                slot.frame_id = frame_counter_;
                slot.timestamp_ns = static_cast<uint64_t>(ns);

                header_->latest_slot.store(static_cast<uint32_t>(write_slot), std::memory_order_release);
                header_->writer_frame_id.store(frame_counter_, std::memory_order_release);

                last_written_slot_ = write_slot;

                glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
            }
        }

        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

        pbo_index_ = 1 - pbo_index_;
        pbo_initialized_ = true;
    }

    void ShmSink::createPBOs(int width, int height) {
        destroyPBOs();

        const auto buf_size = static_cast<GLsizeiptr>(width) * height * SHM_CHANNELS;
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

    void ShmSink::destroyPBOs() {
        if (pbos_[0] != 0) {
            glDeleteBuffers(2, pbos_);
            pbos_[0] = pbos_[1] = 0;
        }
        pbo_initialized_ = false;
    }

} // namespace lfs::vis

#endif // __linux__
