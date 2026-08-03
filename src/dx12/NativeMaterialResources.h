#pragma once

#include "dx12/SponzaScene.h"
#include "gfx.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace ddgi::dx12 {

struct NativeMaterialGpuData {
    std::uint32_t materialIndex{};
    std::uint32_t ntcSlot{};
    GfxTexture diffuse{};
    GfxTexture normal{};
    GfxTexture roughness{};
    GfxTexture metallic{};
};

std::vector<NativeMaterialGpuData> loadNativeMaterials(
    GfxContext context,
    const std::vector<MaterialDrawBatch>& batches,
    const std::filesystem::path& materialRoot);

void destroyNativeMaterials(
    GfxContext context,
    std::vector<NativeMaterialGpuData>& materials);

} // namespace ddgi::dx12
