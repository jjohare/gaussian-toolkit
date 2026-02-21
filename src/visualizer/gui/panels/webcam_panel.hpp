/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/panel_registry.hpp"
#include <string>
#include <vector>

namespace lfs::vis {
    class WebcamManager;
}

namespace lfs::vis::gui::panels {

    class WebcamPanel : public IPanel {
    public:
        explicit WebcamPanel(WebcamManager* manager);

        void draw(const PanelDrawContext& ctx) override;

    private:
        void refreshDeviceList();

        WebcamManager* manager_;
        std::vector<std::pair<std::string, std::string>> device_list_; // path, name
        int selected_device_ = -1;
        int resolution_index_ = 1; // default 720p
        bool devices_queried_ = false;
    };

} // namespace lfs::vis::gui::panels
