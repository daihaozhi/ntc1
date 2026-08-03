#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace ddgi::dx12 {

struct SceneVertex {
    float position[3]{};
    float normal[3]{};
    float uv[2]{};
};

struct MaterialDrawBatch {
    std::uint32_t materialIndex{};
    std::uint32_t ntcSlot{};
    std::uint32_t firstIndex{};
    std::uint32_t indexCount{};
    bool alphaTest{};
    bool hasNormalMap{};
    float baseColorFactor[4]{1.0f, 1.0f, 1.0f, 1.0f};
    float roughnessFactor{1.0f};
    float metallicFactor{};
    float boundsMin[3]{};
    float boundsMax[3]{};
};

struct SponzaSceneData {
    std::vector<SceneVertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<MaterialDrawBatch> materialBatches;
};

SponzaSceneData loadSponzaSceneByMaterial(
    const std::filesystem::path& gltfPath,
    const std::filesystem::path& ntcRoot);

} // namespace ddgi::dx12
