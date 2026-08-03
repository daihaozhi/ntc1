#include "dx12/SponzaScene.h"

#include "gfx_scene.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ddgi::dx12 {
namespace {

struct PendingBatch {
    std::uint32_t materialIndex{};
    std::uint32_t ntcSlot{};
    bool alphaTest{};
    bool hasNormalMap{};
    glm::vec4 baseColorFactor{1.0f};
    float roughnessFactor{1.0f};
    float metallicFactor{};
    std::vector<std::uint32_t> indices;
    glm::vec3 boundsMin{FLT_MAX};
    glm::vec3 boundsMax{-FLT_MAX};
};

std::filesystem::path materialAssetPath(
    const std::filesystem::path& root,
    std::uint32_t materialIndex) {
    return root / std::format("material_{:02}", materialIndex);
}

bool hasCompleteNtcAsset(
    const std::filesystem::path& root,
    std::uint32_t materialIndex) {
    const std::filesystem::path materialPath =
        materialAssetPath(root, materialIndex);
    return std::filesystem::exists(materialPath / "grid_0.png") &&
           std::filesystem::exists(materialPath / "grid_1.png") &&
           std::filesystem::exists(materialPath / "mlp_weights.bin");
}

} // namespace

SponzaSceneData loadSponzaSceneByMaterial(
    const std::filesystem::path& gltfPath,
    const std::filesystem::path& ntcRoot) {
    if (!std::filesystem::exists(gltfPath)) {
        throw std::runtime_error(
            "Sponza glTF was not found: " + gltfPath.string());
    }
    if (!std::filesystem::exists(ntcRoot)) {
        throw std::runtime_error(
            "Sponza NTC asset root was not found: " + ntcRoot.string());
    }

    GfxScene scene = gfxCreateScene();
    if (!scene) {
        throw std::runtime_error("gfxCreateScene failed");
    }

    try {
        if (gfxSceneImport(scene, gltfPath.string().c_str()) !=
            kGfxResult_NoError) {
            throw std::runtime_error(
                "gfxSceneImport failed for " + gltfPath.string());
        }

        SponzaSceneData output;
        const std::uint32_t materialCount = gfxSceneGetMaterialCount(scene);
        std::vector<PendingBatch> pending(materialCount);
        std::uint32_t nextNtcSlot = 0;
        for (std::uint32_t materialIndex = 0;
             materialIndex < materialCount;
             ++materialIndex) {
            PendingBatch& batch = pending[materialIndex];
            batch.materialIndex = materialIndex;
            if (hasCompleteNtcAsset(ntcRoot, materialIndex)) {
                batch.ntcSlot = nextNtcSlot++;
            } else {
                batch.ntcSlot = UINT32_MAX;
            }
            const GfxRef<GfxMaterial> material =
                gfxSceneGetMaterialHandle(scene, materialIndex);
            batch.alphaTest =
                material &&
                material->alpha_mode == GfxMaterialAlphaMode_Mask;
            if (material) {
                batch.hasNormalMap = static_cast<bool>(material->normal_map);
                batch.baseColorFactor = material->albedo;
                batch.roughnessFactor = material->roughness;
                batch.metallicFactor = material->metallicity;
            }
        }

        const std::uint32_t instanceCount = gfxSceneGetInstanceCount(scene);
        for (std::uint32_t instanceIndex = 0;
             instanceIndex < instanceCount;
             ++instanceIndex) {
            const GfxRef<GfxInstance> instance =
                gfxSceneGetInstanceHandle(scene, instanceIndex);
            if (!instance || !instance->mesh) {
                continue;
            }

            const std::uint32_t materialIndex = instance->material
                ? static_cast<std::uint32_t>(instance->material)
                : 0u;
            if (materialIndex >= pending.size()) {
                throw std::runtime_error(
                    "Sponza instance references an invalid material");
            }

            const GfxMesh& mesh = *instance->mesh;
            const std::uint32_t baseVertex =
                static_cast<std::uint32_t>(output.vertices.size());
            const glm::mat4 transform = instance->transform;
            const glm::mat3 normalTransform =
                glm::transpose(glm::inverse(glm::mat3(transform)));
            PendingBatch& batch = pending[materialIndex];

            output.vertices.reserve(
                output.vertices.size() + mesh.vertices.size());
            for (const GfxVertex& source : mesh.vertices) {
                const glm::vec3 position =
                    glm::vec3(transform * glm::vec4(source.position, 1.0f));
                const glm::vec3 normal =
                    glm::normalize(normalTransform * source.normal);
                output.vertices.push_back({
                    {position.x, position.y, position.z},
                    {normal.x, normal.y, normal.z},
                    {source.uv.x, source.uv.y},
                });
                batch.boundsMin = glm::min(batch.boundsMin, position);
                batch.boundsMax = glm::max(batch.boundsMax, position);
            }

            batch.indices.reserve(
                batch.indices.size() + mesh.indices.size());
            for (std::uint32_t index : mesh.indices) {
                batch.indices.push_back(baseVertex + index);
            }
        }

        for (PendingBatch& pendingBatch : pending) {
            if (pendingBatch.indices.empty()) {
                continue;
            }
            MaterialDrawBatch& batch =
                output.materialBatches.emplace_back();
            batch.materialIndex = pendingBatch.materialIndex;
            batch.ntcSlot = pendingBatch.ntcSlot;
            batch.alphaTest = pendingBatch.alphaTest;
            batch.hasNormalMap = pendingBatch.hasNormalMap;
            std::copy(
                &pendingBatch.baseColorFactor.x,
                &pendingBatch.baseColorFactor.x + 4,
                batch.baseColorFactor);
            batch.roughnessFactor = pendingBatch.roughnessFactor;
            batch.metallicFactor = pendingBatch.metallicFactor;
            batch.firstIndex =
                static_cast<std::uint32_t>(output.indices.size());
            batch.indexCount =
                static_cast<std::uint32_t>(pendingBatch.indices.size());
            std::copy(
                &pendingBatch.boundsMin.x,
                &pendingBatch.boundsMin.x + 3,
                batch.boundsMin);
            std::copy(
                &pendingBatch.boundsMax.x,
                &pendingBatch.boundsMax.x + 3,
                batch.boundsMax);
            output.indices.insert(
                output.indices.end(),
                pendingBatch.indices.begin(),
                pendingBatch.indices.end());
        }

        if (output.vertices.empty() || output.indices.empty() ||
            output.materialBatches.empty()) {
            throw std::runtime_error(
                "Sponza import produced no material draw batches");
        }

        gfxDestroyScene(scene);
        return output;
    } catch (...) {
        gfxDestroyScene(scene);
        throw;
    }
}

} // namespace ddgi::dx12
