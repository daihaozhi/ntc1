#include "dx12/NtcMaterialResources.h"

#include "d3d12_format.hpp"
#include "gfx_scene.h"
#include "gfx_utility.hpp"

#include <half.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <span>
#include <stdexcept>
#include <vector>

namespace ddgi::dx12 {
namespace {

using Half = half_float::half;

std::filesystem::path materialPath(
    const std::filesystem::path& root,
    std::uint32_t materialIndex,
    const std::string& prefix) {
    return root / std::format("{}{:02}", prefix, materialIndex);
}

std::vector<Half> loadWeights(
    const std::filesystem::path& path,
    const NtcNetworkDesc& network) {
    if (network.layerCount < 2 || network.inputDimension == 0 ||
        network.hiddenDimension == 0 || network.outputDimension == 0 ||
        network.outputDimension > network.sourceOutputDimension) {
        throw std::runtime_error("Invalid NTC network description");
    }
    const std::size_t firstLayerWeightCount =
        network.inputDimension * network.hiddenDimension;
    const std::size_t hiddenLayerWeightCount =
        network.hiddenDimension * network.hiddenDimension;
    const std::size_t hiddenLayerCount = network.layerCount - 2;
    const std::size_t sourceOutputLayerWeightCount =
        network.sourceOutputDimension * network.hiddenDimension;
    const std::size_t outputLayerWeightCount =
        network.outputDimension * network.hiddenDimension;
    const std::size_t sourceWeightCount =
        firstLayerWeightCount +
        hiddenLayerCount * hiddenLayerWeightCount +
        sourceOutputLayerWeightCount;
    const std::size_t effectiveWeightCount =
        firstLayerWeightCount +
        hiddenLayerCount * hiddenLayerWeightCount +
        outputLayerWeightCount;

    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error(
            "Unable to open NTC weights: " + path.string());
    }
    const std::streamsize byteCount = stream.tellg();
    if (byteCount !=
        static_cast<std::streamsize>(sourceWeightCount * sizeof(float))) {
        throw std::runtime_error(std::format(
            "Unexpected NTC weight size for {}: {} bytes",
            path.string(),
            byteCount));
    }
    stream.seekg(0, std::ios::beg);
    std::vector<float> source(sourceWeightCount);
    if (!stream.read(
            reinterpret_cast<char*>(source.data()),
            byteCount)) {
        throw std::runtime_error(
            "Unable to read NTC weights: " + path.string());
    }

    std::vector<Half> converted(effectiveWeightCount);
    for (std::size_t i = 0; i < converted.size(); ++i) {
        converted[i] = Half(source[i]);
    }
    return converted;
}

GfxTexture loadGridTexture(
    GfxContext context,
    const std::filesystem::path& path) {
    GfxScene imageScene = gfxCreateScene();
    if (!imageScene) {
        throw std::runtime_error("gfxCreateScene failed for NTC grid");
    }
    try {
        if (gfxSceneImport(imageScene, path.string().c_str()) !=
            kGfxResult_NoError) {
            throw std::runtime_error(
                "Unable to load NTC grid: " + path.string());
        }
        if (gfxSceneGetImageCount(imageScene) != 1) {
            throw std::runtime_error(
                "NTC grid import did not produce exactly one image");
        }
        const GfxRef<GfxImage> image =
            gfxSceneGetImageHandle(imageScene, 0);
        if (!image || image->bytes_per_channel != 1 ||
            image->channel_count != 4) {
            throw std::runtime_error(
                "NTC grid must be an 8-bit RGBA image: " + path.string());
        }

        GfxTexture texture = gfxCreateTexture2D(
            context,
            image->width,
            image->height,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            1);
        if (!texture) {
            throw std::runtime_error(
                "gfxCreateTexture2D failed for " + path.string());
        }
        const std::uint64_t texelCount =
            static_cast<std::uint64_t>(image->width) *
            image->height;
        const std::uint64_t byteCount =
            texelCount * image->channel_count;
        std::vector<std::uint8_t> pixelMajor(byteCount, 0);
        for (std::uint64_t texel = 0;
             texel < texelCount;
             ++texel) {
            for (std::uint32_t feature = 0;
                 feature < 8;
                 ++feature) {
                const std::uint64_t sourceFlat =
                    texel * 8 + feature;
                const std::uint32_t sourcePackedChannel =
                    static_cast<std::uint32_t>(
                        sourceFlat / texelCount);
                const std::uint64_t sourceTexel =
                    sourceFlat -
                    static_cast<std::uint64_t>(
                        sourcePackedChannel) * texelCount;
                const std::uint8_t sourceByte =
                    image->data[
                        sourceTexel * 4 +
                        sourcePackedChannel / 2];
                const std::uint8_t nibble =
                    (sourcePackedChannel & 1) != 0
                        ? sourceByte >> 4
                        : sourceByte & 0x0f;
                std::uint8_t& destination =
                    pixelMajor[texel * 4 + feature / 2];
                if ((feature & 1) != 0) {
                    destination |=
                        static_cast<std::uint8_t>(nibble << 4);
                } else {
                    destination |= nibble;
                }
            }
        }
        GfxBuffer upload = gfxCreateBuffer(
            context,
            byteCount,
            pixelMajor.data(),
            kGfxCpuAccess_Write);
        if (!upload) {
            gfxDestroyTexture(context, texture);
            throw std::runtime_error(
                "gfxCreateBuffer failed for " + path.string());
        }
        const GfxResult copyResult =
            gfxCommandCopyBufferToTexture(context, texture, upload);
        gfxDestroyBuffer(context, upload);
        if (copyResult != kGfxResult_NoError) {
            gfxDestroyTexture(context, texture);
            throw std::runtime_error(
                "Uploading NTC grid failed for " + path.string());
        }
        gfxDestroyScene(imageScene);
        return texture;
    } catch (...) {
        gfxDestroyScene(imageScene);
        throw;
    }
}

} // namespace

std::vector<NtcMaterialGpuData> loadNtcMaterials(
    GfxContext context,
    const std::vector<MaterialDrawBatch>& batches,
    const std::filesystem::path& ntcRoot,
    const NtcNetworkDesc& network) {
    std::uint32_t materialCount = 0;
    for (const MaterialDrawBatch& batch : batches) {
        if (batch.ntcSlot != UINT32_MAX) {
            materialCount = std::max(materialCount, batch.ntcSlot + 1);
        }
    }
    if (materialCount == 0) {
        throw std::runtime_error("No NTC material batches were found");
    }

    std::vector<NtcMaterialGpuData> materials(materialCount);
    try {
        for (const MaterialDrawBatch& batch : batches) {
            if (batch.ntcSlot == UINT32_MAX) {
                continue;
            }
            const std::filesystem::path path =
                materialPath(
                    ntcRoot,
                    batch.materialIndex,
                    network.materialDirectoryPrefix);
            std::vector<Half> packedSource =
                loadWeights(path / "mlp_weights.bin", network);

            const std::size_t firstLayerWeightCount =
                network.inputDimension * network.hiddenDimension;
            const std::size_t hiddenLayerWeightCount =
                network.hiddenDimension * network.hiddenDimension;
            const std::size_t outputLayerWeightCount =
                network.outputDimension * network.hiddenDimension;
            std::vector<ex::D3D12MatrixInfo<Half>> matrixInfo(
                network.layerCount);
            matrixInfo[0].m_srcData = std::span<const Half>(
                packedSource.data(),
                firstLayerWeightCount);
            matrixInfo[0].m_rowSize = network.hiddenDimension;
            matrixInfo[0].m_columnSize = network.inputDimension;
            std::size_t weightOffset = firstLayerWeightCount;
            for (std::uint32_t layer = 1;
                 layer + 1 < network.layerCount;
                 ++layer) {
                matrixInfo[layer].m_srcData = std::span<const Half>(
                    packedSource.data() + weightOffset,
                    hiddenLayerWeightCount);
                matrixInfo[layer].m_rowSize = network.hiddenDimension;
                matrixInfo[layer].m_columnSize = network.hiddenDimension;
                weightOffset += hiddenLayerWeightCount;
            }
            matrixInfo.back().m_srcData = std::span<const Half>(
                packedSource.data() + weightOffset,
                outputLayerWeightCount);
            matrixInfo.back().m_rowSize = network.outputDimension;
            matrixInfo.back().m_columnSize = network.hiddenDimension;
            std::vector<ex::D3D12MatrixInfo<Half>> scalarMatrixInfo =
                matrixInfo;
            for (auto& info : matrixInfo) {
                info.m_layout = ex::MatrixLayout::MUL_OPTIMAL;
            }

            std::shared_ptr<GfxBuffer> weights =
                ex::packAsD3D12MatrixBuffer<Half>(
                    context,
                    matrixInfo,
                    false);
            if (!weights || !*weights ||
                matrixInfo[0].m_layout != ex::MatrixLayout::MUL_OPTIMAL) {
                throw std::runtime_error(std::format(
                    "MUL_OPTIMAL conversion failed for material {}",
                    batch.materialIndex));
            }

            std::shared_ptr<GfxBuffer> scalarWeights;
            if (network.createScalarWeights) {
                for (auto& info : scalarMatrixInfo) {
                    info.m_layout = ex::MatrixLayout::ROW_MAJOR;
                }
                scalarWeights = ex::packAsD3D12MatrixBuffer<Half>(
                    context,
                    scalarMatrixInfo,
                    false);
                if (!scalarWeights || !*scalarWeights ||
                    scalarMatrixInfo[0].m_layout !=
                        ex::MatrixLayout::ROW_MAJOR) {
                    throw std::runtime_error(std::format(
                        "ROW_MAJOR packing failed for material {}",
                        batch.materialIndex));
                }
            }

            NtcMaterialGpuData& material = materials[batch.ntcSlot];
            material.materialIndex = batch.materialIndex;
            material.ntcSlot = batch.ntcSlot;
            material.highGrid =
                loadGridTexture(context, path / "grid_0.png");
            material.lowGrid =
                loadGridTexture(context, path / "grid_1.png");
            material.weights = std::move(weights);
            material.scalarWeights = std::move(scalarWeights);
            material.firstMatrixSize =
                static_cast<std::uint32_t>(matrixInfo[0].m_dataSize);
            material.hiddenMatrixSize =
                static_cast<std::uint32_t>(matrixInfo[1].m_dataSize);
            material.matrixLayout =
                static_cast<std::uint32_t>(
                    ex::toHlslMatrixLayout(matrixInfo[0].m_layout));
            material.matrixAlignment =
                static_cast<std::uint32_t>(matrixInfo[0].m_alignment);
            material.matrixVectorStrideAlignment =
                static_cast<std::uint32_t>(
                    matrixInfo[0].m_vectorStrideAlignment);
            if (network.createScalarWeights) {
                material.scalarFirstMatrixSize =
                    static_cast<std::uint32_t>(
                        scalarMatrixInfo[0].m_dataSize);
                material.scalarHiddenMatrixSize =
                    static_cast<std::uint32_t>(
                        scalarMatrixInfo[1].m_dataSize);
                material.scalarMatrixLayout =
                    static_cast<std::uint32_t>(
                        ex::toHlslMatrixLayout(
                            scalarMatrixInfo[0].m_layout));
                material.scalarMatrixAlignment =
                    static_cast<std::uint32_t>(
                        scalarMatrixInfo[0].m_alignment);
                material.scalarMatrixVectorStrideAlignment =
                    static_cast<std::uint32_t>(
                        scalarMatrixInfo[0]
                            .m_vectorStrideAlignment);
            }
        }
        if (gfxFinish(context) != kGfxResult_NoError) {
            throw std::runtime_error(
                "Waiting for NTC resource uploads failed");
        }
        return materials;
    } catch (...) {
        destroyNtcMaterials(context, materials);
        throw;
    }
}

void destroyNtcMaterials(
    GfxContext context,
    std::vector<NtcMaterialGpuData>& materials) {
    for (NtcMaterialGpuData& material : materials) {
        material.weights.reset();
        material.scalarWeights.reset();
        if (material.highGrid) {
            gfxDestroyTexture(context, material.highGrid);
        }
        if (material.lowGrid) {
            gfxDestroyTexture(context, material.lowGrid);
        }
        material = {};
    }
    materials.clear();
}

} // namespace ddgi::dx12
