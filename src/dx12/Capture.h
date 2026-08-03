#pragma once

#include "gfx.h"

#include <filesystem>

namespace ddgi::dx12 {

void captureRgba8Texture(
    GfxContext context,
    GfxTexture texture,
    const std::filesystem::path& outputPath);

} // namespace ddgi::dx12
