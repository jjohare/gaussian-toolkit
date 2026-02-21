/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "webcam/webcam_manager.hpp"
#include "core/logger.hpp"
#include "rendering/billboard_renderer.hpp"

#ifdef __linux__
#include "webcam_capture.hpp"
#endif

#include <cassert>
#include <glm/gtc/matrix_transform.hpp>

namespace lfs::vis {

    WebcamManager::WebcamManager() = default;

    WebcamManager::~WebcamManager() {
        closeCapture();
    }

    void WebcamManager::initialize() {
        renderer_ = std::make_unique<lfs::rendering::BillboardRenderer>();
        auto result = renderer_->initialize();
        if (!result) {
            LOG_ERROR("WebcamManager: failed to initialize billboard renderer: {}", result.error());
            renderer_.reset();
            return;
        }
        initialized_ = true;
    }

    bool WebcamManager::isCapturing() const {
#ifdef __linux__
        return capture_ && capture_->isCapturing();
#else
        return false;
#endif
    }

    void WebcamManager::setEnabled(bool enabled) {
        if (settings_.enabled == enabled)
            return;
        settings_.enabled = enabled;
        if (enabled) {
            openCapture();
        } else {
            closeCapture();
        }
    }

    void WebcamManager::setDevicePath(const std::string& path) {
        if (settings_.device_path == path)
            return;
        settings_.device_path = path;
        if (settings_.enabled) {
            closeCapture();
            openCapture();
        }
    }

    void WebcamManager::setCaptureResolution(int width, int height) {
        if (settings_.capture_width == width && settings_.capture_height == height)
            return;
        settings_.capture_width = width;
        settings_.capture_height = height;
        if (settings_.enabled) {
            closeCapture();
            openCapture();
        }
    }

    void WebcamManager::openCapture() {
#ifdef __linux__
        if (settings_.device_path.empty())
            return;
        capture_ = std::make_unique<lfs::io::WebcamCapture>();
        if (!capture_->open(settings_.device_path, settings_.capture_width, settings_.capture_height)) {
            LOG_ERROR("WebcamManager: failed to open {}", settings_.device_path);
            capture_.reset();
            return;
        }
        settings_.aspect_ratio = static_cast<float>(capture_->frameWidth()) /
                                 static_cast<float>(capture_->frameHeight());
#else
        LOG_WARN("WebcamManager: webcam capture not supported on this platform");
#endif
    }

    void WebcamManager::closeCapture() {
#ifdef __linux__
        if (capture_) {
            capture_->close();
            capture_.reset();
        }
#endif
    }

    std::vector<WebcamManager::DeviceInfo> WebcamManager::enumerateDevices() const {
        std::vector<DeviceInfo> result;
#ifdef __linux__
        auto devices = lfs::io::WebcamCapture::enumerateDevices();
        result.reserve(devices.size());
        for (const auto& d : devices) {
            result.push_back({d.path, d.name});
        }
#endif
        return result;
    }

    void WebcamManager::renderBillboard(const glm::mat4& view, const glm::mat4& projection,
                                        const glm::vec3& camera_pos) {
        if (!initialized_ || !settings_.enabled || !renderer_)
            return;

#ifdef __linux__
        if (!capture_ || !capture_->isCapturing())
            return;

        if (capture_->hasNewFrame()) {
            const uint8_t* data = nullptr;
            int w = 0, h = 0;
            if (capture_->getLatestFrame(data, w, h)) {
                renderer_->uploadFrame(data, w, h);
            }
        }
#else
        return;
#endif

        // Compute model matrix
        glm::mat4 model(1.0f);

        if (settings_.mode == BillboardMode::FaceCamera) {
            // Billboard facing camera
            glm::vec3 to_camera = glm::normalize(camera_pos - settings_.position);
            glm::vec3 world_up(0.0f, 1.0f, 0.0f);
            glm::vec3 right = glm::normalize(glm::cross(world_up, to_camera));
            glm::vec3 up = glm::cross(to_camera, right);

            // Build rotation from axes
            glm::mat4 rotation(1.0f);
            rotation[0] = glm::vec4(right, 0.0f);
            rotation[1] = glm::vec4(up, 0.0f);
            rotation[2] = glm::vec4(to_camera, 0.0f);

            model = glm::translate(glm::mat4(1.0f), settings_.position) *
                    rotation *
                    glm::scale(glm::mat4(1.0f), glm::vec3(settings_.aspect_ratio * settings_.scale,
                                                          settings_.scale, 1.0f));
        } else {
            // Fixed orientation from euler angles
            model = glm::translate(glm::mat4(1.0f), settings_.position);
            model = glm::rotate(model, glm::radians(settings_.rotation_euler.y), glm::vec3(0, 1, 0));
            model = glm::rotate(model, glm::radians(settings_.rotation_euler.x), glm::vec3(1, 0, 0));
            model = glm::rotate(model, glm::radians(settings_.rotation_euler.z), glm::vec3(0, 0, 1));
            model = glm::scale(model, glm::vec3(settings_.aspect_ratio * settings_.scale,
                                                settings_.scale, 1.0f));
        }

        renderer_->render(model, view, projection, settings_.opacity);
    }

} // namespace lfs::vis
