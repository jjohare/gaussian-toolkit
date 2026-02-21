/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <vector>

#ifdef __linux__
namespace lfs::io {
    class WebcamCapture;
    struct WebcamDeviceInfo;
} // namespace lfs::io
#endif

namespace lfs::rendering {
    class BillboardRenderer;
}

namespace lfs::vis {

    enum class BillboardMode : int {
        FaceCamera = 0,
        Fixed = 1,
    };

    struct WebcamSettings {
        bool enabled = false;
        std::string device_path;
        int capture_width = 1280;
        int capture_height = 720;
        glm::vec3 position{0.0f, 1.5f, -3.0f};
        glm::vec3 rotation_euler{0.0f};
        float scale = 1.0f;
        float opacity = 1.0f;
        float aspect_ratio = 16.0f / 9.0f;
        BillboardMode mode = BillboardMode::FaceCamera;
    };

    class LFS_VIS_API WebcamManager {
    public:
        WebcamManager();
        ~WebcamManager();

        WebcamManager(const WebcamManager&) = delete;
        WebcamManager& operator=(const WebcamManager&) = delete;

        void initialize();
        bool isInitialized() const { return initialized_; }

        void setEnabled(bool enabled);
        bool isEnabled() const { return settings_.enabled; }
        bool isCapturing() const;

        void setDevicePath(const std::string& path);
        const std::string& getDevicePath() const { return settings_.device_path; }

        void setCaptureResolution(int width, int height);

        void setPosition(const glm::vec3& pos) { settings_.position = pos; }
        const glm::vec3& getPosition() const { return settings_.position; }

        void setRotationEuler(const glm::vec3& rot) { settings_.rotation_euler = rot; }
        const glm::vec3& getRotationEuler() const { return settings_.rotation_euler; }

        void setScale(float s) { settings_.scale = s; }
        float getScale() const { return settings_.scale; }

        void setOpacity(float o) { settings_.opacity = o; }
        float getOpacity() const { return settings_.opacity; }

        void setMode(BillboardMode mode) { settings_.mode = mode; }
        BillboardMode getMode() const { return settings_.mode; }

        float getAspectRatio() const { return settings_.aspect_ratio; }

        void renderBillboard(const glm::mat4& view, const glm::mat4& projection,
                             const glm::vec3& camera_pos);

        struct DeviceInfo {
            std::string path;
            std::string name;
        };
        std::vector<DeviceInfo> enumerateDevices() const;

    private:
        void openCapture();
        void closeCapture();

        WebcamSettings settings_;
        bool initialized_ = false;

#ifdef __linux__
        std::unique_ptr<lfs::io::WebcamCapture> capture_;
#endif
        std::unique_ptr<lfs::rendering::BillboardRenderer> renderer_;
    };

} // namespace lfs::vis
