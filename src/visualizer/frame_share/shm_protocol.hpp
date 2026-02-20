/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace lfs::vis::shm {

    inline constexpr uint32_t SHM_MAGIC = 0x4C465346; // "LFSF"
    inline constexpr uint32_t SHM_VERSION = 2;
    inline constexpr int SHM_NUM_SLOTS = 3;
    inline constexpr int SHM_CHANNELS = 4; // RGBA
    inline constexpr size_t SHM_HEADER_ALIGNMENT = 4096;

    struct ShmSlot {
        int width;
        int height;
        int stride;
        uint64_t frame_id;
        uint64_t timestamp_ns;
        uint8_t padding_[32];
    };

    static_assert(sizeof(ShmSlot) == 64, "ShmSlot must be 64 bytes for cache alignment");

    struct ShmHeader {
        uint32_t magic;
        uint32_t version;
        int max_width;
        int max_height;
        int channels;
        int reserved_;
        std::atomic<uint64_t> writer_frame_id;
        ShmSlot slots[SHM_NUM_SLOTS];
        std::atomic<uint32_t> latest_slot;
        std::atomic<uint32_t> reader_active_slot;
        uint8_t padding_[SHM_HEADER_ALIGNMENT - (sizeof(uint32_t) * 2 + sizeof(int) * 4 +
                                                 sizeof(std::atomic<uint64_t>) +
                                                 sizeof(ShmSlot) * SHM_NUM_SLOTS +
                                                 sizeof(std::atomic<uint32_t>) * 2)];
    };

    static_assert(sizeof(ShmHeader) == SHM_HEADER_ALIGNMENT, "ShmHeader must be page-aligned");

    inline size_t shmSlotDataSize(int max_width, int max_height) {
        return static_cast<size_t>(max_width) * max_height * SHM_CHANNELS;
    }

    inline size_t shmTotalSize(int max_width, int max_height) {
        return sizeof(ShmHeader) + SHM_NUM_SLOTS * shmSlotDataSize(max_width, max_height);
    }

    inline uint8_t* shmSlotData(ShmHeader* header, int slot_index, int max_width, int max_height) {
        assert(slot_index >= 0 && slot_index < SHM_NUM_SLOTS);
        auto* base = reinterpret_cast<uint8_t*>(header) + sizeof(ShmHeader);
        return base + slot_index * shmSlotDataSize(max_width, max_height);
    }

    inline const uint8_t* shmSlotData(const ShmHeader* header, int slot_index, int max_width, int max_height) {
        assert(slot_index >= 0 && slot_index < SHM_NUM_SLOTS);
        const auto* base = reinterpret_cast<const uint8_t*>(header) + sizeof(ShmHeader);
        return base + slot_index * shmSlotDataSize(max_width, max_height);
    }

} // namespace lfs::vis::shm
