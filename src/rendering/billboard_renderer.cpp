/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "billboard_renderer.hpp"
#include "core/logger.hpp"
#include <cassert>
#include <glm/gtc/type_ptr.hpp>
#include <span>

namespace lfs::rendering {

    Result<void> BillboardRenderer::initialize() {
        // Unit quad: (-0.5,-0.5,0) to (0.5,0.5,0) with UVs
        float vertices[] = {
            // position          // texcoord
            -0.5f,
            0.5f,
            0.0f,
            0.0f,
            1.0f,
            -0.5f,
            -0.5f,
            0.0f,
            0.0f,
            0.0f,
            0.5f,
            -0.5f,
            0.0f,
            1.0f,
            0.0f,

            -0.5f,
            0.5f,
            0.0f,
            0.0f,
            1.0f,
            0.5f,
            -0.5f,
            0.0f,
            1.0f,
            0.0f,
            0.5f,
            0.5f,
            0.0f,
            1.0f,
            1.0f,
        };

        auto shader_result = load_shader("billboard", "billboard.vert", "billboard.frag", false);
        if (!shader_result) {
            LOG_ERROR("Failed to load billboard shader: {}", shader_result.error().message);
            return std::unexpected("Failed to load billboard shader");
        }
        shader_ = std::move(*shader_result);

        auto vao_result = create_vao();
        if (!vao_result)
            return std::unexpected(vao_result.error());

        auto vbo_result = create_vbo();
        if (!vbo_result)
            return std::unexpected(vbo_result.error());
        vbo_ = std::move(*vbo_result);

        VAOBuilder builder(std::move(*vao_result));
        std::span<const float> vertices_span(vertices, sizeof(vertices) / sizeof(float));

        builder.attachVBO(vbo_, vertices_span, GL_STATIC_DRAW)
            .setAttribute({.index = 0,
                           .size = 3,
                           .type = GL_FLOAT,
                           .normalized = GL_FALSE,
                           .stride = 5 * sizeof(float),
                           .offset = nullptr,
                           .divisor = 0})
            .setAttribute({.index = 1,
                           .size = 2,
                           .type = GL_FLOAT,
                           .normalized = GL_FALSE,
                           .stride = 5 * sizeof(float),
                           .offset = reinterpret_cast<const void*>(3 * sizeof(float)),
                           .divisor = 0});

        vao_ = builder.build();

        // Create texture
        GLuint tex_id;
        glGenTextures(1, &tex_id);
        texture_ = Texture(tex_id);
        glBindTexture(GL_TEXTURE_2D, tex_id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);

        initialized_ = true;
        LOG_INFO("BillboardRenderer initialized");
        return {};
    }

    void BillboardRenderer::uploadFrame(const uint8_t* data, int width, int height) {
        assert(initialized_);
        assert(data);
        assert(width > 0 && height > 0);

        glBindTexture(GL_TEXTURE_2D, texture_.get());
        if (width != tex_width_ || height != tex_height_) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0,
                         GL_RGB, GL_UNSIGNED_BYTE, data);
            tex_width_ = width;
            tex_height_ = height;
        } else {
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                            GL_RGB, GL_UNSIGNED_BYTE, data);
        }
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void BillboardRenderer::render(const glm::mat4& model, const glm::mat4& view,
                                   const glm::mat4& projection, float opacity) {
        assert(initialized_);
        if (tex_width_ == 0 || tex_height_ == 0)
            return;

        glm::mat4 mvp = projection * view * model;

        ShaderScope scope(shader_);
        shader_.set("u_mvp", mvp);
        shader_.set("u_opacity", opacity);
        shader_.set("u_texture", 0);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture_.get());

        VAOBinder vao_bind(vao_);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glBindTexture(GL_TEXTURE_2D, 0);
    }

} // namespace lfs::rendering
