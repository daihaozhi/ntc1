#include "dx12/NativeMaterialResources.h"

#include "gfx_scene.h"

#include <algorithm>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <vector>

namespace ddgi::dx12 {
namespace {

GfxTexture loadNativeTexture(
    GfxContext context,
    const std::filesystem::path& path,
    DXGI_FORMAT sourceFormat,
    DXGI_FORMAT gpuFormat) {
    GfxScene imageScene = gfxCreateScene();
    if (!imageScene) {
        throw std::runtime_error(
            "gfxCreateScene failed for native texture");
    }
    GfxTexture texture{};
    try {
        if (gfxSceneImport(imageScene, path.string().c_str()) !=
            kGfxResult_NoError) {
            throw std::runtime_error(
                "Unable to load native texture: " + path.string());
        }
        if (gfxSceneGetImageCount(imageScene) != 1) {
            throw std::runtime_error(
                "Native texture import did not produce one image: " +
                path.string());
        }
        const GfxRef<GfxImage> image =
            gfxSceneGetImageHandle(imageScene, 0);
        if (!image || image->width != 1024 || image->height != 1024 ||
            image->bytes_per_channel != 1 ||
            image->format != sourceFormat) {
            throw std::runtime_error(std::format(
                "Unexpected native texture layout for {} "
                "(expected 1024x1024, source format {})",
                path.string(),
                static_cast<int>(sourceFormat)));
        }
        texture = gfxCreateTexture2D(
            context,
            image->width,
            image->height,
            gpuFormat,
            1);
        if (!texture) {
            throw std::runtime_error(
                "Creating native texture failed: " + path.string());
        }
        GfxBuffer upload = gfxCreateBuffer(
            context,
            static_cast<std::uint64_t>(image->data.size()),
            image->data.data(),
            kGfxCpuAccess_Write);
        if (!upload) {
            throw std::runtime_error(
                "Creating native texture upload failed: " +
                path.string());
        }
        const GfxResult copyResult =
            gfxCommandCopyBufferToTexture(context, texture, upload);
        gfxDestroyBuffer(context, upload);
        if (copyResult != kGfxResult_NoError) {
            throw std::runtime_error(
                "Uploading native texture failed: " + path.string());
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

std::vector<NativeMaterialGpuData> loadNativeMaterials(
    GfxContext context,
    const std::vector<MaterialDrawBatch>& batches,
    const std::filesystem::path& materialRoot) {
    std::uint32_t materialCount = 0;
    for (const MaterialDrawBatch& batch : batches) {
        if (batch.ntcSlot != UINT32_MAX) {
            materialCount = std::max(materialCount, batch.ntcSlot + 1);
        }
    }
    if (materialCount == 0) {
        throw std::runtime_error(
            "No native comparison material slots were found");
    }

    std::vector<NativeMaterialGpuData> materials(materialCount);
    try {
        for (const MaterialDrawBatch& batch : batches) {
            if (batch.ntcSlot == UINT32_MAX) {
                continue;
            }
            NativeMaterialGpuData& material =
                materials[batch.ntcSlot];
            material.materialIndex = batch.materialIndex;
            material.ntcSlot = batch.ntcSlot;
            const std::filesystem::path root =
                materialRoot / std::format(
                    "material_{:02}",
                    batch.materialIndex);
            material.diffuse = loadNativeTexture(
                context,
                root / "diffuse.png",
                DXGI_FORMAT_R8G8B8A8_UNORM,
                DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);
            material.normal = loadNativeTexture(
                context,
                root / "normal.png",
                DXGI_FORMAT_R8G8B8A8_UNORM,
                DXGI_FORMAT_R8G8B8A8_UNORM);
            material.roughness = loadNativeTexture(
                context,
                root / "roughness.png",
                DXGI_FORMAT_R8_UNORM,
                DXGI_FORMAT_R8_UNORM);
            material.metallic = loadNativeTexture(
                context,
                root / "metallic.png",
                DXGI_FORMAT_R8_UNORM,
                DXGI_FORMAT_R8_UNORM);
        }
        return materials;
    } catch (...) {
        destroyNativeMaterials(context, materials);
        throw;
    }
}

void destroyNativeMaterials(
    GfxContext context,
    std::vector<NativeMaterialGpuData>& materials) {
    for (NativeMaterialGpuData& material : materials) {
        if (material.diffuse) {
            gfxDestroyTexture(context, material.diffuse);
        }
        if (material.normal) {
            gfxDestroyTexture(context, material.normal);
        }
        if (material.roughness) {
            gfxDestroyTexture(context, material.roughness);
        }
        if (material.metallic) {
            gfxDestroyTexture(context, material.metallic);
        }
    }
    materials.clear();
}

} // namespace ddgi::dx12
