/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <obs-module.h>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-lichtfeld-source", "en-US")

extern struct obs_source_info lichtfeld_shm_source;

bool obs_module_load(void) {
    obs_register_source(&lichtfeld_shm_source);
    return true;
}

void obs_module_unload(void) {}
