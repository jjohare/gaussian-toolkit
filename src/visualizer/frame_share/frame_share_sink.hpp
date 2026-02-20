/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <string>

namespace lfs::vis {

    class IFrameShareSink {
    public:
        virtual ~IFrameShareSink() = default;

        virtual bool start(const std::string& name, int width, int height) = 0;
        virtual void stop() = 0;
        virtual void sendFrame(unsigned int gl_texture_id, int width, int height) = 0;
        virtual bool isActive() const = 0;
        virtual int connectedReceivers() const { return -1; }
    };

} // namespace lfs::vis
