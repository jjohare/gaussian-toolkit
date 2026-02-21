/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/webcam_panel.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "gui/string_keys.hpp"
#include "theme/theme.hpp"
#include "webcam/webcam_manager.hpp"
#include <cassert>
#include <imgui.h>

using namespace lichtfeld::Strings;

namespace lfs::vis::gui::panels {

    static constexpr int RESOLUTIONS[][2] = {
        {640, 480},
        {1280, 720},
        {1920, 1080},
    };
    static constexpr const char* RESOLUTION_LABELS[] = {"640x480", "1280x720", "1920x1080"};
    static constexpr int NUM_RESOLUTIONS = 3;

    static constexpr const char* MODE_LABELS[] = {"Face Camera", "Fixed"};

    WebcamPanel::WebcamPanel(WebcamManager* manager)
        : manager_(manager) {
        assert(manager_);
    }

    void WebcamPanel::refreshDeviceList() {
        device_list_.clear();
        auto devices = manager_->enumerateDevices();
        for (const auto& d : devices) {
            device_list_.emplace_back(d.path, d.name);
        }

        // Find current device in list
        selected_device_ = -1;
        const auto& current = manager_->getDevicePath();
        for (int i = 0; i < static_cast<int>(device_list_.size()); ++i) {
            if (device_list_[i].first == current) {
                selected_device_ = i;
                break;
            }
        }
        devices_queried_ = true;
    }

    void WebcamPanel::draw(const PanelDrawContext& /*ctx*/) {
        if (!devices_queried_) {
            refreshDeviceList();
        }

        // Enable checkbox
        bool enabled = manager_->isEnabled();
        if (ImGui::Checkbox(LOC(Webcam::ENABLE), &enabled)) {
            manager_->setEnabled(enabled);
        }

        ImGui::Separator();

        // Device dropdown
        ImGui::TextUnformatted(LOC(Webcam::DEVICE));

        const char* device_preview = (selected_device_ >= 0)
                                         ? device_list_[selected_device_].second.c_str()
                                         : LOC(Webcam::NO_DEVICE);

        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##webcam_device", device_preview)) {
            for (int i = 0; i < static_cast<int>(device_list_.size()); ++i) {
                const bool is_selected = (selected_device_ == i);
                const auto& label = device_list_[i].second + " (" + device_list_[i].first + ")";
                if (ImGui::Selectable(label.c_str(), is_selected)) {
                    selected_device_ = i;
                    manager_->setDevicePath(device_list_[i].first);
                }
                if (is_selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        if (ImGui::SmallButton("Refresh")) {
            refreshDeviceList();
        }

        // Resolution dropdown
        ImGui::Spacing();
        ImGui::TextUnformatted(LOC(Webcam::RESOLUTION));
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##webcam_resolution", &resolution_index_, RESOLUTION_LABELS, NUM_RESOLUTIONS)) {
            manager_->setCaptureResolution(RESOLUTIONS[resolution_index_][0],
                                           RESOLUTIONS[resolution_index_][1]);
        }

        // Mode dropdown
        ImGui::Spacing();
        ImGui::TextUnformatted(LOC(Webcam::MODE));
        ImGui::SetNextItemWidth(-1);
        int mode = static_cast<int>(manager_->getMode());
        if (ImGui::Combo("##webcam_mode", &mode, MODE_LABELS, 2)) {
            manager_->setMode(static_cast<BillboardMode>(mode));
        }

        ImGui::Separator();

        // Position
        ImGui::TextUnformatted(LOC(Webcam::POSITION));
        glm::vec3 pos = manager_->getPosition();
        ImGui::SetNextItemWidth(-1);
        if (ImGui::DragFloat3("##webcam_pos", &pos.x, 0.05f)) {
            manager_->setPosition(pos);
        }

        // Rotation (only in Fixed mode)
        if (manager_->getMode() == BillboardMode::Fixed) {
            ImGui::TextUnformatted(LOC(Webcam::ROTATION));
            glm::vec3 rot = manager_->getRotationEuler();
            ImGui::SetNextItemWidth(-1);
            if (ImGui::DragFloat3("##webcam_rot", &rot.x, 1.0f, -180.0f, 180.0f)) {
                manager_->setRotationEuler(rot);
            }
        }

        // Scale
        ImGui::TextUnformatted(LOC(Webcam::SCALE));
        float scale = manager_->getScale();
        ImGui::SetNextItemWidth(-1);
        if (ImGui::SliderFloat("##webcam_scale", &scale, 0.1f, 10.0f, "%.2f")) {
            manager_->setScale(scale);
        }

        // Opacity
        ImGui::TextUnformatted(LOC(Webcam::OPACITY));
        float opacity = manager_->getOpacity();
        ImGui::SetNextItemWidth(-1);
        if (ImGui::SliderFloat("##webcam_opacity", &opacity, 0.0f, 1.0f, "%.2f")) {
            manager_->setOpacity(opacity);
        }

        // Status
        ImGui::Separator();
        const auto& t = theme();
        ImGui::TextUnformatted(LOC(Webcam::STATUS));
        ImGui::SameLine();

        if (manager_->isCapturing()) {
            ImGui::TextColored(t.palette.success, "%s", LOC(Webcam::CAPTURING));
        } else if (manager_->isEnabled()) {
            ImGui::TextColored(t.palette.error, "%s", LOC(Webcam::DISCONNECTED));
        } else {
            ImGui::TextDisabled("%s", LOC(Webcam::INACTIVE));
        }
    }

} // namespace lfs::vis::gui::panels
