#pragma once

#include "dx12/SponzaScene.h"
#include "gfx.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace ddgi::dx12 {

struct NtcMaterialGpuData {
    std::uint32_t materialIndex{};
    std::uint32_t ntcSlot{};
    GfxTexture highGrid{};
    GfxTexture lowGrid{};
    std::shared_ptr<GfxBuffer> weights;
    std::shared_ptr<GfxBuffer> scalarWeights;
    std::uint32_t firstMatrixSize{};
    std::uint32_t hiddenMatrixSize{};
    std::uint32_t matrixLayout{};
    std::uint32_t matrixAlignment{};
    std::uint32_t matrixVectorStrideAlignment{};
    std::uint32_t scalarFirstMatrixSize{};
    std::uint32_t scalarHiddenMatrixSize{};
    std::uint32_t scalarMatrixLayout{};
    std::uint32_t scalarMatrixAlignment{};
    std::uint32_t scalarMatrixVectorStrideAlignment{};
};

struct NtcNetworkDesc {
    std::uint32_t layerCount{};
    std::uint32_t inputDimension{};
    std::uint32_t hiddenDimension{};
    std::uint32_t sourceOutputDimension{};
    std::uint32_t outputDimension{};
    std::string materialDirectoryPrefix;
    bool createScalarWeights{};
};

std::vector<NtcMaterialGpuData> loadNtcMaterials(
    GfxContext context,
    const std::vector<MaterialDrawBatch>& batches,
    const std::filesystem::path& ntcRoot,
    const NtcNetworkDesc& network);

void destroyNtcMaterials(
    GfxContext context,
    std::vector<NtcMaterialGpuData>& materials);

} // namespace ddgi::dx12
