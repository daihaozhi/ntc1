#pragma once

#include "dx12/SponzaScene.h"
#include "gfx.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace ddgi::dx12 {

struct BcMaterialGpuData {
    std::uint32_t materialIndex{};
    std::uint32_t ntcSlot{};
    GfxTexture diffuse{};
    GfxTexture normal{};
    GfxTexture metallicRoughness{};
};

std::vector<BcMaterialGpuData> loadBcMaterials(
    GfxContext context,
    const std::vector<MaterialDrawBatch>& batches,
    const std::filesystem::path& bcRoot);

void destroyBcMaterials(
    GfxContext context,
    std::vector<BcMaterialGpuData>& materials);

} // namespace ddgi::dx12
