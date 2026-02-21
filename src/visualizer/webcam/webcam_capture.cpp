/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#ifdef __linux__

#include "webcam_capture.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <csetjmp>
#include <cstring>
#include <fcntl.h>
#include <jpeglib.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <unistd.h>

namespace lfs::io {

    WebcamCapture::~WebcamCapture() {
        close();
    }

    std::vector<WebcamDeviceInfo> WebcamCapture::enumerateDevices() {
        std::vector<WebcamDeviceInfo> devices;

        for (int i = 0; i < 64; ++i) {
            const std::string path = "/dev/video" + std::to_string(i);
            int fd = ::open(path.c_str(), O_RDWR | O_NONBLOCK);
            if (fd < 0)
                continue;

            struct v4l2_capability cap {};
            if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == 0) {
                if (!(cap.device_caps & V4L2_CAP_VIDEO_CAPTURE)) {
                    ::close(fd);
                    continue;
                }

                const std::string driver(reinterpret_cast<const char*>(cap.driver));
                if (driver.find("v4l2 loopback") != std::string::npos ||
                    driver.find("v4l2loopback") != std::string::npos) {
                    ::close(fd);
                    continue;
                }

                WebcamDeviceInfo info;
                info.path = path;
                info.name = reinterpret_cast<const char*>(cap.card);
                info.index = i;
                devices.push_back(std::move(info));
            }
            ::close(fd);
        }

        return devices;
    }

    bool WebcamCapture::open(const std::string& device_path, int width, int height) {
        close();

        fd_ = ::open(device_path.c_str(), O_RDWR);
        if (fd_ < 0) {
            LOG_ERROR("Webcam: failed to open {}: {}", device_path, strerror(errno));
            return false;
        }

        struct v4l2_format fmt {};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = static_cast<uint32_t>(width);
        fmt.fmt.pix.height = static_cast<uint32_t>(height);
        fmt.fmt.pix.field = V4L2_FIELD_NONE;

        // MJPEG first — delivers 30fps at 720p+ over USB 2.0,
        // vs ~5fps for raw YUYV due to bandwidth limits.
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
        if (ioctl(fd_, VIDIOC_S_FMT, &fmt) == 0 &&
            fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_MJPEG) {
            format_ = CaptureFormat::MJPEG;
        } else {
            fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
            if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
                LOG_ERROR("Webcam: failed to set format: {}", strerror(errno));
                ::close(fd_);
                fd_ = -1;
                return false;
            }
            format_ = CaptureFormat::YUYV;
        }

        frame_width_ = static_cast<int>(fmt.fmt.pix.width);
        frame_height_ = static_cast<int>(fmt.fmt.pix.height);
        const char* fmt_name = (format_ == CaptureFormat::MJPEG) ? "MJPEG" : "YUYV";
        LOG_INFO("Webcam: negotiated {}x{} {}", frame_width_, frame_height_, fmt_name);

        // Request max frame rate
        struct v4l2_streamparm parm {};
        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd_, VIDIOC_G_PARM, &parm) == 0 &&
            (parm.parm.capture.capability & V4L2_CAP_TIMEPERFRAME)) {
            parm.parm.capture.timeperframe.numerator = 1;
            parm.parm.capture.timeperframe.denominator = 30;
            ioctl(fd_, VIDIOC_S_PARM, &parm);
            LOG_INFO("Webcam: requested {}/{} fps",
                     parm.parm.capture.timeperframe.denominator,
                     parm.parm.capture.timeperframe.numerator);
        }

        // Setup MMAP buffers
        struct v4l2_requestbuffers req {};
        req.count = NUM_BUFFERS;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;

        if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
            LOG_ERROR("Webcam: VIDIOC_REQBUFS failed: {}", strerror(errno));
            ::close(fd_);
            fd_ = -1;
            return false;
        }

        num_mmap_buffers_ = static_cast<int>(req.count);
        assert(num_mmap_buffers_ <= NUM_BUFFERS);

        for (int i = 0; i < num_mmap_buffers_; ++i) {
            struct v4l2_buffer buf {};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = static_cast<uint32_t>(i);

            if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
                LOG_ERROR("Webcam: VIDIOC_QUERYBUF failed: {}", strerror(errno));
                ::close(fd_);
                fd_ = -1;
                return false;
            }

            mmap_buffers_[i].length = buf.length;
            mmap_buffers_[i].start = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE,
                                          MAP_SHARED, fd_, buf.m.offset);
            if (mmap_buffers_[i].start == MAP_FAILED) {
                LOG_ERROR("Webcam: mmap failed: {}", strerror(errno));
                mmap_buffers_[i].start = nullptr;
                ::close(fd_);
                fd_ = -1;
                return false;
            }
        }

        // Queue all buffers
        for (int i = 0; i < num_mmap_buffers_; ++i) {
            struct v4l2_buffer buf {};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = static_cast<uint32_t>(i);
            if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
                LOG_ERROR("Webcam: VIDIOC_QBUF failed: {}", strerror(errno));
                ::close(fd_);
                fd_ = -1;
                return false;
            }
        }

        // Start streaming
        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
            LOG_ERROR("Webcam: VIDIOC_STREAMON failed: {}", strerror(errno));
            ::close(fd_);
            fd_ = -1;
            return false;
        }

        // Allocate double buffers
        const size_t frame_size = static_cast<size_t>(frame_width_) * frame_height_ * 3;
        frame_buffers_[0].resize(frame_size);
        frame_buffers_[1].resize(frame_size);

        capturing_.store(true);
        capture_thread_ = std::thread(&WebcamCapture::captureThread, this);
        LOG_INFO("Webcam: capture started on {}", device_path);
        return true;
    }

    void WebcamCapture::close() {
        capturing_.store(false);
        if (capture_thread_.joinable())
            capture_thread_.join();

        if (fd_ >= 0) {
            int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            ioctl(fd_, VIDIOC_STREAMOFF, &type);

            for (int i = 0; i < num_mmap_buffers_; ++i) {
                if (mmap_buffers_[i].start && mmap_buffers_[i].start != MAP_FAILED) {
                    munmap(mmap_buffers_[i].start, mmap_buffers_[i].length);
                    mmap_buffers_[i].start = nullptr;
                }
            }
            num_mmap_buffers_ = 0;

            ::close(fd_);
            fd_ = -1;
        }

        frame_width_ = 0;
        frame_height_ = 0;
        new_frame_available_.store(false);
    }

    bool WebcamCapture::getLatestFrame(const uint8_t*& data, int& out_width, int& out_height) {
        if (!new_frame_available_.load())
            return false;

        int front = front_buffer_.load();
        data = frame_buffers_[front].data();
        out_width = frame_width_;
        out_height = frame_height_;
        new_frame_available_.store(false);
        return true;
    }

    void WebcamCapture::convertYUYVtoRGB(const uint8_t* yuyv, uint8_t* rgb, int width, int height) {
        const int num_pixels = width * height;
        for (int i = 0; i < num_pixels / 2; ++i) {
            int y0 = yuyv[i * 4 + 0];
            int u = yuyv[i * 4 + 1] - 128;
            int y1 = yuyv[i * 4 + 2];
            int v = yuyv[i * 4 + 3] - 128;

            int c0 = 298 * (y0 - 16);
            int c1 = 298 * (y1 - 16);
            int d = u;
            int e = v;

            rgb[i * 6 + 0] = static_cast<uint8_t>(std::clamp((c0 + 409 * e + 128) >> 8, 0, 255));
            rgb[i * 6 + 1] = static_cast<uint8_t>(std::clamp((c0 - 100 * d - 208 * e + 128) >> 8, 0, 255));
            rgb[i * 6 + 2] = static_cast<uint8_t>(std::clamp((c0 + 516 * d + 128) >> 8, 0, 255));

            rgb[i * 6 + 3] = static_cast<uint8_t>(std::clamp((c1 + 409 * e + 128) >> 8, 0, 255));
            rgb[i * 6 + 4] = static_cast<uint8_t>(std::clamp((c1 - 100 * d - 208 * e + 128) >> 8, 0, 255));
            rgb[i * 6 + 5] = static_cast<uint8_t>(std::clamp((c1 + 516 * d + 128) >> 8, 0, 255));
        }
    }

    namespace {
        struct JpegErrorMgr {
            jpeg_error_mgr pub;
            std::jmp_buf jmp;
            char msg[JMSG_LENGTH_MAX]{};
        };
        void jpegErrorExit(j_common_ptr cinfo) {
            auto* err = reinterpret_cast<JpegErrorMgr*>(cinfo->err);
            cinfo->err->format_message(cinfo, err->msg);
            std::longjmp(err->jmp, 1);
        }
    } // namespace

    bool WebcamCapture::decodeMJPEG(const uint8_t* data, size_t size,
                                    uint8_t* dst, int width, int height) {
        jpeg_decompress_struct cinfo{};
        JpegErrorMgr jerr{};
        cinfo.err = jpeg_std_error(&jerr.pub);
        jerr.pub.error_exit = jpegErrorExit;

        if (setjmp(jerr.jmp)) {
            LOG_ERROR("Webcam: JPEG decode failed: {}", jerr.msg);
            jpeg_destroy_decompress(&cinfo);
            return false;
        }

        jpeg_create_decompress(&cinfo);
        jpeg_mem_src(&cinfo, data, static_cast<unsigned long>(size));
        jpeg_read_header(&cinfo, TRUE);

        cinfo.out_color_space = JCS_RGB;
        cinfo.dct_method = JDCT_FASTEST;
        jpeg_start_decompress(&cinfo);

        if (static_cast<int>(cinfo.output_width) != width ||
            static_cast<int>(cinfo.output_height) != height) {
            LOG_ERROR("Webcam: JPEG size mismatch: got {}x{}, expected {}x{}",
                      cinfo.output_width, cinfo.output_height, width, height);
            jpeg_destroy_decompress(&cinfo);
            return false;
        }

        const int row_stride = width * 3;
        while (cinfo.output_scanline < cinfo.output_height) {
            uint8_t* row = dst + cinfo.output_scanline * row_stride;
            jpeg_read_scanlines(&cinfo, &row, 1);
        }

        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        return true;
    }

    void WebcamCapture::captureThread() {
        using Clock = std::chrono::steady_clock;
        int frame_count = 0;
        int decode_failures = 0;
        double last_decode_ms = 0.0;
        auto fps_timer = Clock::now();

        while (capturing_.load()) {
            // Periodic stats (runs even when all decodes fail)
            auto now = Clock::now();
            auto elapsed = std::chrono::duration<double>(now - fps_timer).count();
            if (elapsed >= 5.0) {
                double fps = frame_count / elapsed;
                LOG_INFO("Webcam: {:.1f} fps, decode {:.1f} ms ({}), failures: {}",
                         fps, last_decode_ms,
                         format_ == CaptureFormat::MJPEG ? "MJPEG" : "YUYV",
                         decode_failures);
                frame_count = 0;
                decode_failures = 0;
                fps_timer = now;
            }

            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(fd_, &fds);

            struct timeval tv {};
            tv.tv_sec = 0;
            tv.tv_usec = 100000;

            int ret = select(fd_ + 1, &fds, nullptr, nullptr, &tv);
            if (ret <= 0)
                continue;

            struct v4l2_buffer buf {};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;

            if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
                if (errno == EAGAIN)
                    continue;
                LOG_ERROR("Webcam: VIDIOC_DQBUF failed: {}", strerror(errno));
                break;
            }

            assert(buf.index < static_cast<uint32_t>(num_mmap_buffers_));
            auto* raw_data = static_cast<uint8_t*>(mmap_buffers_[buf.index].start);
            int back = 1 - front_buffer_.load();

            auto t0 = Clock::now();
            bool decoded = false;

            if (format_ == CaptureFormat::MJPEG) {
                decoded = decodeMJPEG(raw_data, buf.bytesused,
                                      frame_buffers_[back].data(),
                                      frame_width_, frame_height_);
                if (!decoded)
                    ++decode_failures;
            } else {
                convertYUYVtoRGB(raw_data, frame_buffers_[back].data(),
                                 frame_width_, frame_height_);
                decoded = true;
            }

            last_decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

            if (decoded) {
                front_buffer_.store(back);
                new_frame_available_.store(true);
                ++frame_count;
            }

            if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
                LOG_ERROR("Webcam: VIDIOC_QBUF re-queue failed: {}", strerror(errno));
                break;
            }
        }
    }

} // namespace lfs::io

#endif // __linux__
