cmake_minimum_required(VERSION 3.24)

project(sponza1k_ntc_dx12 LANGUAGES CXX)

if(NOT WIN32)
    message(FATAL_ERROR "This renderer requires Windows and DirectX 12.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(MINIDXNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MINIDXNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MINIDXNN_BUILD_CPP_FALLBACK_ONLY OFF CACHE BOOL "" FORCE)
set(GFX_ENABLE_SCENE ON CACHE BOOL "" FORCE)
set(GFX_ENABLE_GUI ON CACHE BOOL "" FORCE)
set(GFX_ENABLE_SCENE_KTX OFF CACHE BOOL "" FORCE)
add_subdirectory(external/MiniDXNN EXCLUDE_FROM_ALL)

add_executable(sponza1k_ntc_dx12
    src/dx12/main.cpp
    src/dx12/BcMaterialResources.cpp
    src/dx12/BcMaterialResources.h
    src/dx12/Capture.cpp
    src/dx12/Capture.h
    src/dx12/NativeMaterialResources.cpp
    src/dx12/NativeMaterialResources.h
    src/dx12/NtcMaterialResources.cpp
    src/dx12/NtcMaterialResources.h
    src/dx12/SponzaScene.cpp
    src/dx12/SponzaScene.h
    shaders/dx12/gbuffer_composite.frag
    shaders/dx12/gbuffer_composite.vert
    shaders/dx12/ntc_probe.frag
    shaders/dx12/ntc_probe.vert
    shaders/dx12/sponza_depth.frag
    shaders/dx12/sponza_depth.vert
    shaders/dx12/sponza_ntc.frag
    shaders/dx12/sponza_ntc.vert
    external/MiniDXNN/example/common/gfx_utility.cpp
)

target_include_directories(sponza1k_ntc_dx12 PRIVATE
    src
    external/MiniDXNN/example/common
    external/MiniDXNN/third_party/gfx_dep/gfx
    external/MiniDXNN/third_party/stb_image_dep
)

target_link_libraries(sponza1k_ntc_dx12 PRIVATE minidxnn-core)

target_compile_definitions(sponza1k_ntc_dx12 PRIVATE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    DDGI_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    DDGI_DX12_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders/dx12"
    DDGI_MINIDXNN_HLSL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/external/MiniDXNN/include/minidxnn/hlsl"
    DDGI_DXC_HLSL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/external/MiniDXNN/third_party/gfx_dep/gfx/third_party/directx-dxc/inc/hlsl"
)

if(MSVC)
    target_compile_options(sponza1k_ntc_dx12 PRIVATE /W4 /permissive- /bigobj)
endif()

add_custom_command(TARGET sponza1k_ntc_dx12 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_newer
            $<TARGET_RUNTIME_DLLS:sponza1k_ntc_dx12>
            $<TARGET_FILE_DIR:sponza1k_ntc_dx12>
    COMMAND_EXPAND_LISTS
    COMMENT "Copying DirectX 12 Agility, DXC, and PIX runtime DLLs"
)
