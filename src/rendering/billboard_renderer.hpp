/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gl_resources.hpp"
#include "shader_manager.hpp"
#include <glm/glm.hpp>

namespace lfs::rendering {

    class BillboardRenderer {
    public:
        BillboardRenderer() = default;
        ~BillboardRenderer() = default;

        BillboardRenderer(const BillboardRenderer&) = delete;
        BillboardRenderer& operator=(const BillboardRenderer&) = delete;

        Result<void> initialize();
        void uploadFrame(const uint8_t* data, int width, int height);
        void render(const glm::mat4& model, const glm::mat4& view,
                    const glm::mat4& projection, float opacity);

        bool isInitialized() const { return initialized_; }

    private:
        VAO vao_;
        VBO vbo_;
        Texture texture_;
        ManagedShader shader_;
        int tex_width_ = 0;
        int tex_height_ = 0;
        bool initialized_ = false;
    };

} // namespace lfs::rendering
