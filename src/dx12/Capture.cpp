#include "dx12/Capture.h"

#include <stb_image_write.h>

#include <cstdint>
#include <filesystem>
#include <stdexcept>

namespace ddgi::dx12 {

void captureRgba8Texture(
    GfxContext context,
    GfxTexture texture,
    const std::filesystem::path& outputPath) {
    if (texture.getFormat() != DXGI_FORMAT_R8G8B8A8_UNORM &&
        texture.getFormat() != DXGI_FORMAT_R8G8B8A8_UNORM_SRGB) {
        throw std::runtime_error(
            "captureRgba8Texture requires an R8G8B8A8 UNORM format");
    }
    const std::uint32_t width = gfxGetBackBufferWidth(context);
    const std::uint32_t height = gfxGetBackBufferHeight(context);
    const std::uint64_t byteCount =
        static_cast<std::uint64_t>(width) * height * 4;
    GfxBuffer readback = gfxCreateBuffer(
        context,
        byteCount,
        nullptr,
        kGfxCpuAccess_Read);
    if (!readback) {
        throw std::runtime_error("Creating capture readback buffer failed");
    }
    if (gfxCommandCopyTextureToBuffer(context, readback, texture) !=
            kGfxResult_NoError ||
        gfxFinish(context) != kGfxResult_NoError) {
        gfxDestroyBuffer(context, readback);
        throw std::runtime_error("Reading back capture texture failed");
    }
    const void* pixels = gfxBufferGetData(context, readback);
    if (!pixels) {
        gfxDestroyBuffer(context, readback);
        throw std::runtime_error("Mapping capture readback buffer failed");
    }
    std::filesystem::create_directories(outputPath.parent_path());
    const int result = stbi_write_png(
        outputPath.string().c_str(),
        static_cast<int>(width),
        static_cast<int>(height),
        4,
        pixels,
        static_cast<int>(width * 4));
    gfxDestroyBuffer(context, readback);
    if (result == 0) {
        throw std::runtime_error(
            "Writing capture PNG failed: " + outputPath.string());
    }
}

} // namespace ddgi::dx12
