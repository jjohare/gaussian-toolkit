/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "frame_share/shm_protocol.hpp"
#include <obs-module.h>

struct LichtfeldShmSource {
    obs_source_t* source;
    gs_texture_t* texture;
    char shm_name[256];

    lfs::vis::shm::ShmHeader* header;
    int shm_fd;
    size_t shm_size;

    uint64_t last_frame_id;
    int tex_width;
    int tex_height;

    int stale_ticks;
};
