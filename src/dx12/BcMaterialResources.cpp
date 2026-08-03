#include "dx12/BcMaterialResources.h"

#include "gfx_scene.h"

#include <algorithm>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <vector>

namespace ddgi::dx12 {
namespace {

GfxTexture loadBcTexture(
    GfxContext context,
    const std::filesystem::path& path,
    DXGI_FORMAT expectedFormat) {
    GfxScene imageScene = gfxCreateScene();
    if (!imageScene) {
        throw std::runtime_error("gfxCreateScene failed for BC texture");
    }
    GfxTexture texture{};
    try {
        if (gfxSceneImport(imageScene, path.string().c_str()) !=
            kGfxResult_NoError) {
            throw std::runtime_error(
                "Unable to load BC texture: " + path.string());
        }
        if (gfxSceneGetImageCount(imageScene) != 1) {
            throw std::runtime_error(
                "BC import did not produce exactly one image: " +
                path.string());
        }
        const GfxRef<GfxImage> image =
            gfxSceneGetImageHandle(imageScene, 0);
        if (!image || image->width != 348 || image->height != 348 ||
            image->format != expectedFormat ||
            (image->flags & kGfxImageFlag_HasMipLevels) != 0) {
            throw std::runtime_error(std::format(
                "Unexpected BC texture layout for {} "
                "(expected 348x348, format {}, one mip)",
                path.string(),
                static_cast<int>(expectedFormat)));
        }
        texture = gfxCreateTexture2D(
            context,
            image->width,
            image->height,
            image->format,
            1);
        if (!texture) {
            throw std::runtime_error(
                "Creating BC texture failed: " + path.string());
        }
        GfxBuffer upload = gfxCreateBuffer(
            context,
            static_cast<std::uint64_t>(image->data.size()),
            image->data.data(),
            kGfxCpuAccess_Write);
        if (!upload) {
            throw std::runtime_error(
                "Creating BC upload buffer failed: " + path.string());
        }
        const GfxResult copyResult =
            gfxCommandCopyBufferToTexture(context, texture, upload);
        gfxDestroyBuffer(context, upload);
        if (copyResult != kGfxResult_NoError) {
            throw std::runtime_error(
                "Uploading BC blocks failed: " + path.string());
        }
        gfxDestroyScene(imageScene);
        return texture;
    } catch (...) {
        if (texture) {
            gfxDestroyTexture(context, texture);
        }
        gfxDestroyScene(imageScene);
        throw;
    }
}

} // namespace

std::vector<BcMaterialGpuData> loadBcMaterials(
    GfxContext context,
    const std::vector<MaterialDrawBatch>& batches,
    const std::filesystem::path& bcRoot) {
    std::uint32_t materialCount = 0;
    for (const MaterialDrawBatch& batch : batches) {
        if (batch.ntcSlot != UINT32_MAX) {
            materialCount = std::max(materialCount, batch.ntcSlot + 1);
        }
    }
    if (materialCount == 0) {
        throw std::runtime_error("No BC comparison material slots were found");
    }

    std::vector<BcMaterialGpuData> materials(materialCount);
    try {
        for (const MaterialDrawBatch& batch : batches) {
            if (batch.ntcSlot == UINT32_MAX) {
                continue;
            }
            BcMaterialGpuData& material = materials[batch.ntcSlot];
            material.materialIndex = batch.materialIndex;
            material.ntcSlot = batch.ntcSlot;
            const std::filesystem::path materialRoot =
                bcRoot / std::format(
                    "material_{:02}",
                    batch.materialIndex);
            material.diffuse = loadBcTexture(
                context,
                materialRoot / "diffuse.dds",
                DXGI_FORMAT_BC7_UNORM_SRGB);
            material.normal = loadBcTexture(
                context,
                materialRoot / "normal.dds",
                DXGI_FORMAT_BC5_UNORM);
            material.metallicRoughness = loadBcTexture(
                context,
                materialRoot / "metallic_roughness.dds",
                DXGI_FORMAT_BC5_UNORM);
        }
        return materials;
    } catch (...) {
        destroyBcMaterials(context, materials);
        throw;
    }
}

void destroyBcMaterials(
    GfxContext context,
    std::vector<BcMaterialGpuData>& materials) {
    for (BcMaterialGpuData& material : materials) {
        if (material.diffuse) {
            gfxDestroyTexture(context, material.diffuse);
        }
        if (material.normal) {
            gfxDestroyTexture(context, material.normal);
        }
        if (material.metallicRoughness) {
            gfxDestroyTexture(context, material.metallicRoughness);
        }
    }
    materials.clear();
}

} // namespace ddgi::dx12
