#include "d3d12_format.hpp"
#include "dx12/BcMaterialResources.h"
#include "dx12/Capture.h"
#include "dx12/NativeMaterialResources.h"
#include "dx12/NtcMaterialResources.h"
#include "dx12/SponzaScene.h"
#include "gfx.h"
#include "gfx_imgui.h"
#include "gfx_window.h"

#include <wrl/client.h>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <array>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(GfxResult result, const char* operation) {
    if (result != kGfxResult_NoError) {
        fail(std::format(
            "{} failed (GfxResult={})",
            operation,
            static_cast<int>(result)));
    }
}

std::uint32_t parseFrameCount(int argc, char** argv) {
    std::uint32_t frames = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--frames" && i + 1 < argc) {
            frames = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        }
    }
    return frames;
}

bool hasArgument(int argc, char** argv, std::string_view argument) {
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == argument) {
            return true;
        }
    }
    return false;
}

std::filesystem::path parseCapturePath(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--capture" && i + 1 < argc) {
            return std::filesystem::absolute(argv[++i]);
        }
    }
    return {};
}

struct Camera {
    glm::vec3 position{0.0f, 3.0f, 0.0f};
    float yaw{1.57079632679f};
    float pitch{-0.04f};
};

enum class SceneMode {
    Ntc,
    NtcH32,
    NtcH32L2,
    NtcH32L2Scalar,
    Native,
    NativeVsNtcH32L2,
    NtcH64VsH32,
    NativeVsNtc,
    BcVsNtc,
};

enum class MaterialPath {
    NtcH64,
    NtcH32,
    NtcH32L2,
    NtcH32L2Scalar,
    Native,
    Bc,
};

bool isComparisonScene(SceneMode mode) {
    return mode == SceneMode::NativeVsNtcH32L2 ||
        mode == SceneMode::NtcH64VsH32 ||
        mode == SceneMode::NativeVsNtc ||
        mode == SceneMode::BcVsNtc;
}

const char* sceneLabel(SceneMode mode) {
    switch (mode) {
    case SceneMode::Ntc:
        return "NTC H64/L3";
    case SceneMode::NtcH32:
        return "NTC H32/L4";
    case SceneMode::NtcH32L2:
        return "NTC H32/2 hidden Matrix Core";
    case SceneMode::NtcH32L2Scalar:
        return "NTC H32/2 hidden Scalar ALU";
    case SceneMode::Native:
        return "Native Sponza 1K";
    case SceneMode::NativeVsNtcH32L2:
        return "Native vs NTC H32/2 hidden";
    case SceneMode::NtcH64VsH32:
        return "NTC H64/L3 vs H32/L4";
    case SceneMode::NativeVsNtc:
        return "Native vs NTC H64/L3";
    case SceneMode::BcVsNtc:
        return "BC vs NTC H64/L3";
    }
    return "Unknown";
}

SceneMode parseSceneMode(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--scene" && i + 1 < argc) {
            const std::string_view scene = argv[++i];
            if (scene == "ntc" || scene == "sponza1kntc") {
                return SceneMode::Ntc;
            }
            if (scene == "ntc-h32" || scene == "sponza1kntch32") {
                return SceneMode::NtcH32;
            }
            if (scene == "ntc-h32-l2" ||
                scene == "sponza1kntch32l2") {
                return SceneMode::NtcH32L2;
            }
            if (scene == "ntc-h32-l2-scalar" ||
                scene == "h32-l2-scalar" ||
                scene == "sponza1kntch32l2scalar") {
                return SceneMode::NtcH32L2Scalar;
            }
            if (scene == "native" || scene == "native-1k" ||
                scene == "sponza1knative") {
                return SceneMode::Native;
            }
            if (scene == "native-h32-l2" ||
                scene == "native-vs-h32-l2" ||
                scene == "sponza1knativeh32l2compare") {
                return SceneMode::NativeVsNtcH32L2;
            }
            if (scene == "ntc-h64-h32" || scene == "h64-h32" ||
                scene == "sponza1kntch64h32compare") {
                return SceneMode::NtcH64VsH32;
            }
            if (scene == "bc-ntc" ||
                scene == "sponza1kntcbccompare") {
                return SceneMode::BcVsNtc;
            }
            if (scene == "native-ntc" ||
                scene == "native-vs-ntc" ||
                scene == "sponza1kntccompare") {
                return SceneMode::NativeVsNtc;
            }
            fail(std::format("Unknown DX12 scene '{}'", scene));
        }
    }
    return SceneMode::Ntc;
}

struct MouseLookState {
    bool dragging{};
    POINT lastPosition{};
};

glm::mat4 makeSponzaViewProjection(
    const Camera& camera,
    std::uint32_t width,
    std::uint32_t height) {
    const glm::vec3 forward(
        std::sin(camera.yaw) * std::cos(camera.pitch),
        std::sin(camera.pitch),
        std::cos(camera.yaw) * std::cos(camera.pitch));
    const glm::mat4 view = glm::lookAtRH(
        camera.position,
        camera.position + forward,
        glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::mat4 projection =
        glm::perspectiveRH_ZO(
            glm::radians(60.0f),
            static_cast<float>(width) /
                static_cast<float>(std::max(height, 1u)),
            0.1f,
            80.0f);
    return projection * view;
}

void updateCamera(
    GfxWindow window,
    Camera& camera,
    MouseLookState& mouseLook,
    float deltaSeconds,
    bool captureMouse,
    bool captureKeyboard) {
    const bool rightMouseDown =
        GetForegroundWindow() == window.getHWND() &&
        (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0;
    POINT cursorPosition{};
    if (rightMouseDown && !captureMouse &&
        GetCursorPos(&cursorPosition)) {
        if (mouseLook.dragging) {
            camera.yaw +=
                static_cast<float>(
                    cursorPosition.x - mouseLook.lastPosition.x) *
                0.0032f;
            camera.pitch -=
                static_cast<float>(
                    cursorPosition.y - mouseLook.lastPosition.y) *
                0.0032f;
            camera.pitch =
                std::clamp(camera.pitch, -1.35f, 1.35f);
        }
        mouseLook.dragging = true;
        mouseLook.lastPosition = cursorPosition;
    } else {
        mouseLook.dragging = false;
    }

    const glm::vec3 forward = glm::normalize(glm::vec3(
        std::sin(camera.yaw) * std::cos(camera.pitch),
        std::sin(camera.pitch),
        std::cos(camera.yaw) * std::cos(camera.pitch)));
    const glm::vec3 right = glm::normalize(
        glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
    const float speed =
        gfxWindowIsKeyDown(window, VK_SHIFT) ? 10.0f : 4.0f;
    const float distance = speed * deltaSeconds;
    if (!captureKeyboard && gfxWindowIsKeyDown(window, 'W')) {
        camera.position += forward * distance;
    }
    if (!captureKeyboard && gfxWindowIsKeyDown(window, 'S')) {
        camera.position -= forward * distance;
    }
    if (!captureKeyboard && gfxWindowIsKeyDown(window, 'D')) {
        camera.position += right * distance;
    }
    if (!captureKeyboard && gfxWindowIsKeyDown(window, 'A')) {
        camera.position -= right * distance;
    }
}

enum TimestampStage : std::size_t {
    kTimestampClear,
    kTimestampDepthPrepass,
    kTimestampGBuffer,
    kTimestampComposite,
    kTimestampUi,
    kTimestampFrame,
    kTimestampCount,
};

constexpr std::array<const char*, kTimestampCount> kTimestampStageNames{
    "Target clears",
    "Depth prepass",
    "Material G-buffer",
    "G-buffer composite",
    "UI render/composite",
    "Total GPU render",
};

struct RenderResources {
    GfxBuffer vertexBuffer{};
    GfxBuffer indexBuffer{};
    std::array<GfxTexture, 5> gbuffer{};
    GfxTexture depth{};
    GfxTexture finalColor{};
    GfxTexture uiOverlay{};
    GfxSamplerState bcSampler{};
    GfxProgram depthProgram{};
    GfxKernel depthKernel{};
    GfxProgram gbufferProgram{};
    GfxKernel gbufferKernel{};
    GfxKernel gbufferH32Kernel{};
    GfxKernel gbufferH32L2Kernel{};
    GfxKernel gbufferH32L2ScalarKernel{};
    GfxKernel gbufferNativeKernel{};
    GfxProgram compositeProgram{};
    GfxKernel compositeKernel{};
    std::array<GfxTimestampQuery, kTimestampCount> timestamps{};
};

void destroyRenderResources(
    GfxContext context,
    RenderResources& resources) {
    for (GfxTimestampQuery timestamp : resources.timestamps) {
        if (timestamp) {
            gfxDestroyTimestampQuery(context, timestamp);
        }
    }
    if (resources.depthKernel) {
        gfxDestroyKernel(context, resources.depthKernel);
    }
    if (resources.bcSampler) {
        gfxDestroySamplerState(context, resources.bcSampler);
    }
    if (resources.depthProgram) {
        gfxDestroyProgram(context, resources.depthProgram);
    }
    if (resources.compositeKernel) {
        gfxDestroyKernel(context, resources.compositeKernel);
    }
    if (resources.compositeProgram) {
        gfxDestroyProgram(context, resources.compositeProgram);
    }
    if (resources.gbufferKernel) {
        gfxDestroyKernel(context, resources.gbufferKernel);
    }
    if (resources.gbufferH32Kernel) {
        gfxDestroyKernel(context, resources.gbufferH32Kernel);
    }
    if (resources.gbufferH32L2Kernel) {
        gfxDestroyKernel(context, resources.gbufferH32L2Kernel);
    }
    if (resources.gbufferH32L2ScalarKernel) {
        gfxDestroyKernel(
            context, resources.gbufferH32L2ScalarKernel);
    }
    if (resources.gbufferNativeKernel) {
        gfxDestroyKernel(context, resources.gbufferNativeKernel);
    }
    if (resources.gbufferProgram) {
        gfxDestroyProgram(context, resources.gbufferProgram);
    }
    if (resources.depth) {
        gfxDestroyTexture(context, resources.depth);
    }
    if (resources.finalColor) {
        gfxDestroyTexture(context, resources.finalColor);
    }
    if (resources.uiOverlay) {
        gfxDestroyTexture(context, resources.uiOverlay);
    }
    for (GfxTexture texture : resources.gbuffer) {
        if (texture) {
            gfxDestroyTexture(context, texture);
        }
    }
    if (resources.indexBuffer) {
        gfxDestroyBuffer(context, resources.indexBuffer);
    }
    if (resources.vertexBuffer) {
        gfxDestroyBuffer(context, resources.vertexBuffer);
    }
    resources = {};
}

GfxProgram createProgram(
    GfxContext context,
    const char* name,
    std::array<const char*, 2>& includePaths) {
    const std::filesystem::path shaderDir = DDGI_DX12_SHADER_DIR;
    GfxProgram program = gfxCreateProgram(
        context,
        name,
        shaderDir.string().c_str(),
        "6_10",
        includePaths.data(),
        static_cast<std::uint32_t>(includePaths.size()));
    if (!program) {
        fail(std::format("gfxCreateProgram({}) failed", name));
    }
    return program;
}

} // namespace

int main(int argc, char** argv) {
    GfxWindow window{};
    GfxContext context{};
    RenderResources render{};
    std::vector<ddgi::dx12::NtcMaterialGpuData> ntcMaterials;
    std::vector<ddgi::dx12::NtcMaterialGpuData> ntcH32Materials;
    std::vector<ddgi::dx12::NtcMaterialGpuData> ntcH32L2Materials;
    std::vector<ddgi::dx12::NativeMaterialGpuData> nativeMaterials;
    std::vector<ddgi::dx12::BcMaterialGpuData> bcMaterials;
    bool imguiInitialized = false;

    try {
        const std::uint32_t frameLimit = parseFrameCount(argc, argv);
        SceneMode sceneMode = parseSceneMode(argc, argv);
        const bool maximizeWindow =
            hasArgument(argc, argv, "--maximized");
        const bool cacheShaders =
            hasArgument(argc, argv, "--cache-shaders");
        const bool vsync =
            !hasArgument(argc, argv, "--no-vsync");
        const bool ntcEnabled =
            !hasArgument(argc, argv, "--disable-ntc");
        const bool backfaceCulling =
            !hasArgument(argc, argv, "--no-culling");
        const bool useVrs2x2 =
            hasArgument(argc, argv, "--vrs-2x2");
        const bool featuresOnly =
            hasArgument(argc, argv, "--features-only");
        const bool mlpOnly =
            hasArgument(argc, argv, "--mlp-only");
        if (featuresOnly && mlpOnly) {
            fail("--features-only and --mlp-only are mutually exclusive");
        }
        const std::uint32_t decodeMode =
            featuresOnly ? 1u : (mlpOnly ? 2u : 0u);
        const std::filesystem::path capturePath =
            parseCapturePath(argc, argv);
        window = gfxCreateWindow(
            1024,
            1024,
            "Sponza1K NTC - DirectX 12 LinAlg",
            maximizeWindow ? kGfxCreateWindowFlag_MaximizeWindow : 0);
        if (!window) {
            fail("gfxCreateWindow failed");
        }

        std::uint32_t contextFlags =
            kGfxCreateContextFlag_EnableExperimentalShaders;
        if (cacheShaders) {
            contextFlags |= kGfxCreateContextFlag_EnableShaderCache;
        }
        context = gfxCreateContext(window, contextFlags);
        if (!context) {
            fail(
                "gfxCreateContext failed. Enable Windows Developer Mode and "
                "verify the Agility SDK preview runtime and developer GPU driver.");
        }
        require(gfxImGuiInitialize(context), "Initialize ImGui");
        imguiInitialized = true;

        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList5>
            vrsCommandList;
        if (useVrs2x2) {
            D3D12_FEATURE_DATA_D3D12_OPTIONS6 options6{};
            if (FAILED(gfxGetDevice(context)->CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS6,
                    &options6,
                    sizeof(options6))) ||
                options6.VariableShadingRateTier ==
                    D3D12_VARIABLE_SHADING_RATE_TIER_NOT_SUPPORTED) {
                fail("2x2 variable-rate shading is not supported");
            }
            if (FAILED(gfxGetCommandList(context)->QueryInterface(
                    IID_PPV_ARGS(&vrsCommandList)))) {
                fail("ID3D12GraphicsCommandList5 is unavailable");
            }
            std::cout << "Variable-rate shading: tier "
                      << static_cast<int>(
                             options6.VariableShadingRateTier)
                      << ", using 2x2 for the NTC G-buffer\n";
        }

        const std::uint32_t tier = gfxGetLinearAlgebraTier(context);
        const GfxMatrixMultiplySupportResult fp16Support =
            gfxCheckMatrixMultiplySupport(context, 7u, 7u, 7u);
        std::cout << "DirectX 12 LinAlg tier: "
                  << gfxGetLinearAlgebraTierName(context) << '\n';
        std::cout << "FP16 matrix-vector multiply: supported="
                  << fp16Support.supported
                  << ", hardwareAccelerated="
                  << fp16Support.hardwareAccelerated << '\n';
        if (tier == 0 || !fp16Support.supported ||
            !fp16Support.hardwareAccelerated) {
            fail(
                "Hardware-accelerated FP16 Linear Algebra Matrix support is required");
        }

        const std::filesystem::path sourceRoot = DDGI_SOURCE_DIR;
        const ddgi::dx12::SponzaSceneData sponza =
            ddgi::dx12::loadSponzaSceneByMaterial(
                sourceRoot / "external/sponza1k/glTF/Sponza.gltf",
                sourceRoot / "external/sponza_20260725");
        std::cout << "Sponza1K material batching: "
                  << sponza.vertices.size() << " vertices, "
                  << sponza.indices.size() / 3 << " triangles, "
                  << sponza.materialBatches.size()
                  << " material draw calls\n";
        glm::vec3 sceneMin(FLT_MAX);
        glm::vec3 sceneMax(-FLT_MAX);
        for (const auto& batch : sponza.materialBatches) {
            sceneMin = glm::min(sceneMin, glm::make_vec3(batch.boundsMin));
            sceneMax = glm::max(sceneMax, glm::make_vec3(batch.boundsMax));
        }
        std::cout << std::format(
            "Scene bounds: [{:.2f}, {:.2f}, {:.2f}] - "
            "[{:.2f}, {:.2f}, {:.2f}]\n",
            sceneMin.x, sceneMin.y, sceneMin.z,
            sceneMax.x, sceneMax.y, sceneMax.z);

        ntcMaterials = ddgi::dx12::loadNtcMaterials(
            context,
            sponza.materialBatches,
            sourceRoot / "external/sponza_20260725",
            {3, 64, 64, 16, 8, "material_"});
        std::cout << "Loaded and converted " << ntcMaterials.size()
                  << " NTC H64/L3 material networks to FP16 MUL_OPTIMAL\n";
        ntcH32Materials = ddgi::dx12::loadNtcMaterials(
            context,
            sponza.materialBatches,
            sourceRoot / "external/sponza_freq5_h32_l3",
            {4, 64, 32, 16, 8, "sponza_f5h32l3_material_"});
        std::cout << "Loaded and converted " << ntcH32Materials.size()
                  << " NTC H32/L4 material networks to FP16 MUL_OPTIMAL\n";
        ntcH32L2Materials = ddgi::dx12::loadNtcMaterials(
            context,
            sponza.materialBatches,
            sourceRoot / "external/sponza_freq5_h32_l2",
            {3, 64, 32, 16, 8,
             "sponza_f5h32l2_material_", true});
        std::cout << "Loaded and converted " << ntcH32L2Materials.size()
                  << " NTC H32/2-hidden material networks to FP16 "
                     "MUL_OPTIMAL and ROW_MAJOR\n";
        nativeMaterials = ddgi::dx12::loadNativeMaterials(
            context,
            sponza.materialBatches,
            sourceRoot / "external/sponza_20260725");
        std::cout << "Loaded " << nativeMaterials.size()
                  << " native Sponza1K material sets (1024x1024)\n";
        bcMaterials = ddgi::dx12::loadBcMaterials(
            context,
            sponza.materialBatches,
            sourceRoot / "external/sponza1k_bc_equal");
        std::cout << "Loaded " << bcMaterials.size()
                  << " BC7/BC5 equal-storage material sets (348x348)\n";

        render.vertexBuffer = gfxCreateBuffer(
            context,
            static_cast<std::uint64_t>(
                sponza.vertices.size() *
                sizeof(ddgi::dx12::SceneVertex)),
            sponza.vertices.data());
        render.indexBuffer = gfxCreateBuffer<std::uint32_t>(
            context,
            static_cast<std::uint32_t>(sponza.indices.size()),
            sponza.indices.data());
        if (!render.vertexBuffer || !render.indexBuffer) {
            fail("Uploading Sponza geometry failed");
        }

        render.gbuffer[0] = gfxCreateTexture2D(
            context, DXGI_FORMAT_R16G16B16A16_FLOAT);
        render.gbuffer[1] = gfxCreateTexture2D(
            context, DXGI_FORMAT_R16G16B16A16_FLOAT);
        render.gbuffer[2] = gfxCreateTexture2D(
            context, DXGI_FORMAT_R16G16B16A16_FLOAT);
        render.gbuffer[3] = gfxCreateTexture2D(
            context, DXGI_FORMAT_R16G16B16A16_FLOAT);
        render.gbuffer[4] = gfxCreateTexture2D(
            context, DXGI_FORMAT_R16G16_FLOAT);
        render.depth = gfxCreateTexture2D(
            context, DXGI_FORMAT_D32_FLOAT);
        render.finalColor = gfxCreateTexture2D(
            context, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);
        render.uiOverlay = gfxCreateTexture2D(
            context, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);
        render.bcSampler = gfxCreateSamplerState(
            context,
            D3D12_FILTER_MIN_MAG_MIP_LINEAR,
            D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            D3D12_TEXTURE_ADDRESS_MODE_WRAP);
        for (GfxTexture texture : render.gbuffer) {
            if (!texture) {
                fail("Creating a G-buffer attachment failed");
            }
        }
        if (!render.depth || !render.finalColor || !render.uiOverlay ||
            !render.bcSampler) {
            fail("Creating a depth/final/UI attachment failed");
        }

        const std::array<std::string, 2> includeStorage{
            DDGI_MINIDXNN_HLSL_DIR,
            DDGI_DXC_HLSL_DIR,
        };
        std::array<const char*, 2> includePaths{
            includeStorage[0].c_str(),
            includeStorage[1].c_str(),
        };
        render.depthProgram =
            createProgram(context, "sponza_depth", includePaths);
        render.gbufferProgram =
            createProgram(context, "sponza_ntc", includePaths);
        render.compositeProgram =
            createProgram(context, "gbuffer_composite", includePaths);

        const ddgi::dx12::NtcMaterialGpuData& layout =
            ntcMaterials.front();
        const std::array<std::string, 5> definitionStorage{
            std::format(
                "MINIDXNN_WEIGHT_MATRIX_LAYOUT={}",
                layout.matrixLayout),
            std::format(
                "MINIDXNN_WEIGHT_MATRIX_ALIGNMENT={}",
                layout.matrixAlignment),
            std::format(
                "MINIDXNN_WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT={}",
                layout.matrixVectorStrideAlignment),
            "NTC_LAYER_COUNT=3",
            "NTC_HIDDEN_DIM=64",
        };
        std::array<const char*, 5> definitions{
            definitionStorage[0].c_str(),
            definitionStorage[1].c_str(),
            definitionStorage[2].c_str(),
            definitionStorage[3].c_str(),
            definitionStorage[4].c_str(),
        };
        const std::array<std::string, 5> h32DefinitionStorage{
            definitionStorage[0],
            definitionStorage[1],
            definitionStorage[2],
            "NTC_LAYER_COUNT=4",
            "NTC_HIDDEN_DIM=32",
        };
        std::array<const char*, 5> h32Definitions{
            h32DefinitionStorage[0].c_str(),
            h32DefinitionStorage[1].c_str(),
            h32DefinitionStorage[2].c_str(),
            h32DefinitionStorage[3].c_str(),
            h32DefinitionStorage[4].c_str(),
        };
        const std::array<std::string, 5> h32L2DefinitionStorage{
            definitionStorage[0],
            definitionStorage[1],
            definitionStorage[2],
            "NTC_LAYER_COUNT=3",
            "NTC_HIDDEN_DIM=32",
        };
        std::array<const char*, 5> h32L2Definitions{
            h32L2DefinitionStorage[0].c_str(),
            h32L2DefinitionStorage[1].c_str(),
            h32L2DefinitionStorage[2].c_str(),
            h32L2DefinitionStorage[3].c_str(),
            h32L2DefinitionStorage[4].c_str(),
        };
        const ddgi::dx12::NtcMaterialGpuData& h32L2Layout =
            ntcH32L2Materials.front();
        const std::array<std::string, 6>
            h32L2ScalarDefinitionStorage{
                std::format(
                    "MINIDXNN_WEIGHT_MATRIX_LAYOUT={}",
                    h32L2Layout.scalarMatrixLayout),
                std::format(
                    "MINIDXNN_WEIGHT_MATRIX_ALIGNMENT={}",
                    h32L2Layout.scalarMatrixAlignment),
                std::format(
                    "MINIDXNN_WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT={}",
                    h32L2Layout
                        .scalarMatrixVectorStrideAlignment),
                "NTC_LAYER_COUNT=3",
                "NTC_HIDDEN_DIM=32",
                "MINIDXNN_USE_SOFTWARE_LINALG_IMPL=1",
            };
        std::array<const char*, 6> h32L2ScalarDefinitions{
            h32L2ScalarDefinitionStorage[0].c_str(),
            h32L2ScalarDefinitionStorage[1].c_str(),
            h32L2ScalarDefinitionStorage[2].c_str(),
            h32L2ScalarDefinitionStorage[3].c_str(),
            h32L2ScalarDefinitionStorage[4].c_str(),
            h32L2ScalarDefinitionStorage[5].c_str(),
        };
        const std::array<std::string, 6> nativeDefinitionStorage{
            definitionStorage[0],
            definitionStorage[1],
            definitionStorage[2],
            definitionStorage[3],
            definitionStorage[4],
            "NTC_NATIVE_ONLY=1",
        };
        std::array<const char*, 6> nativeDefinitions{
            nativeDefinitionStorage[0].c_str(),
            nativeDefinitionStorage[1].c_str(),
            nativeDefinitionStorage[2].c_str(),
            nativeDefinitionStorage[3].c_str(),
            nativeDefinitionStorage[4].c_str(),
            nativeDefinitionStorage[5].c_str(),
        };

        GfxDrawState gbufferDrawState;
        require(
            gfxDrawStateSetColorTarget(
                gbufferDrawState, 0, render.gbuffer[0].getFormat()),
            "Set albedo target format");
        require(
            gfxDrawStateSetColorTarget(
                gbufferDrawState, 1, render.gbuffer[1].getFormat()),
            "Set normal target format");
        require(
            gfxDrawStateSetColorTarget(
                gbufferDrawState, 2, render.gbuffer[2].getFormat()),
            "Set material target format");
        require(
            gfxDrawStateSetColorTarget(
                gbufferDrawState, 3, render.gbuffer[3].getFormat()),
            "Set emissive target format");
        require(
            gfxDrawStateSetColorTarget(
                gbufferDrawState, 4, render.gbuffer[4].getFormat()),
            "Set motion target format");
        require(
            gfxDrawStateSetDepthStencilTarget(
                gbufferDrawState, render.depth.getFormat()),
            "Set depth target format");
        require(
            gfxDrawStateSetDepthFunction(
                gbufferDrawState, D3D12_COMPARISON_FUNC_EQUAL),
            "Set G-buffer depth-equal function");
        require(
            gfxDrawStateSetDepthWriteMask(
                gbufferDrawState, D3D12_DEPTH_WRITE_MASK_ZERO),
            "Disable G-buffer depth writes");
        require(
            gfxDrawStateSetCullMode(
                gbufferDrawState,
                backfaceCulling
                    ? D3D12_CULL_MODE_BACK
                    : D3D12_CULL_MODE_NONE),
            "Configure Sponza culling");

        GfxDrawState depthDrawState;
        require(
            gfxDrawStateSetDepthStencilTarget(
                depthDrawState, render.depth.getFormat()),
            "Set prepass depth target format");
        require(
            gfxDrawStateSetCullMode(
                depthDrawState,
                backfaceCulling
                    ? D3D12_CULL_MODE_BACK
                    : D3D12_CULL_MODE_NONE),
            "Configure prepass culling");
        render.depthKernel = gfxCreateGraphicsKernel(
            context,
            render.depthProgram,
            depthDrawState);
        render.gbufferKernel = gfxCreateGraphicsKernel(
            context,
            render.gbufferProgram,
            gbufferDrawState,
            nullptr,
            definitions.data(),
            static_cast<std::uint32_t>(definitions.size()));
        render.gbufferH32Kernel = gfxCreateGraphicsKernel(
            context,
            render.gbufferProgram,
            gbufferDrawState,
            nullptr,
            h32Definitions.data(),
            static_cast<std::uint32_t>(h32Definitions.size()));
        render.gbufferH32L2Kernel = gfxCreateGraphicsKernel(
            context,
            render.gbufferProgram,
            gbufferDrawState,
            nullptr,
            h32L2Definitions.data(),
            static_cast<std::uint32_t>(h32L2Definitions.size()));
        render.gbufferH32L2ScalarKernel = gfxCreateGraphicsKernel(
            context,
            render.gbufferProgram,
            gbufferDrawState,
            nullptr,
            h32L2ScalarDefinitions.data(),
            static_cast<std::uint32_t>(
                h32L2ScalarDefinitions.size()));
        render.gbufferNativeKernel = gfxCreateGraphicsKernel(
            context,
            render.gbufferProgram,
            gbufferDrawState,
            nullptr,
            nativeDefinitions.data(),
            static_cast<std::uint32_t>(nativeDefinitions.size()));
        GfxDrawState compositeDrawState;
        require(
            gfxDrawStateSetColorTarget(
                compositeDrawState,
                0,
                render.finalColor.getFormat()),
            "Set composite target format");
        require(
            gfxDrawStateSetCullMode(
                compositeDrawState, D3D12_CULL_MODE_NONE),
            "Disable culling for the fullscreen composite");
        render.compositeKernel = gfxCreateGraphicsKernel(
            context,
            render.compositeProgram,
            compositeDrawState);
        if (!render.depthKernel ||
            !render.gbufferKernel ||
            !render.gbufferH32Kernel ||
            !render.gbufferH32L2Kernel ||
            !render.gbufferH32L2ScalarKernel ||
            !render.gbufferNativeKernel ||
            !render.compositeKernel) {
            fail("Creating Sponza graphics kernels failed");
        }
        for (GfxTimestampQuery& timestamp : render.timestamps) {
            timestamp = gfxCreateTimestampQuery(context);
            if (!timestamp) {
                fail("Creating a stage timestamp query failed");
            }
        }

        Camera camera;
        MouseLookState mouseLook;
        bool cameraControlsLocked = false;
        glm::mat4 viewProjection = makeSponzaViewProjection(
            camera,
            gfxGetBackBufferWidth(context),
            gfxGetBackBufferHeight(context));
        require(
            gfxProgramSetParameter(
                context,
                render.depthProgram,
                "ViewProjection",
                viewProjection),
            "Bind prepass ViewProjection");
        require(
            gfxProgramSetParameter(
                context,
                render.gbufferProgram,
                "ViewProjection",
                viewProjection),
            "Bind ViewProjection");
        require(
            gfxProgramSetParameter(
                context,
                render.gbufferProgram,
                "PreviousViewProjection",
                viewProjection),
            "Bind PreviousViewProjection");

        require(
            gfxProgramSetParameter(
                context,
                render.compositeProgram,
                "AlbedoBuffer",
                render.gbuffer[0]),
            "Bind composite albedo");
        require(
            gfxProgramSetParameter(
                context,
                render.compositeProgram,
                "NormalBuffer",
                render.gbuffer[1]),
            "Bind composite normal");
        require(
            gfxProgramSetParameter(
                context,
                render.compositeProgram,
                "MaterialBuffer",
                render.gbuffer[2]),
            "Bind composite material");
        require(
            gfxProgramSetParameter(
                context,
                render.compositeProgram,
                "EmissiveBuffer",
                render.gbuffer[3]),
            "Bind composite emissive");

        std::array<std::vector<float>, kTimestampCount> stageTimes;
        std::uint32_t frameIndex = 0;
        std::array<float, kTimestampCount> latestStageMs{};
        auto previousFrameTime = std::chrono::steady_clock::now();
        while (!gfxWindowIsCloseRequested(window) &&
               (frameLimit == 0 || frameIndex < frameLimit)) {
            gfxWindowPumpEvents(window);
            const auto currentFrameTime = std::chrono::steady_clock::now();
            const float deltaSeconds = std::clamp(
                std::chrono::duration<float>(
                    currentFrameTime - previousFrameTime).count(),
                0.0001f,
                0.1f);
            previousFrameTime = currentFrameTime;
            if (gfxWindowIsMinimized(window)) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(16));
                continue;
            }

            if (!cameraControlsLocked) {
                updateCamera(
                    window,
                    camera,
                    mouseLook,
                    deltaSeconds,
                    ImGui::GetIO().WantCaptureMouse,
                    ImGui::GetIO().WantCaptureKeyboard);
            } else {
                mouseLook.dragging = false;
            }
            glm::mat4 previousViewProjection = viewProjection;
            viewProjection = makeSponzaViewProjection(
                camera,
                gfxGetBackBufferWidth(context),
                gfxGetBackBufferHeight(context));

            ImGui::SetNextWindowBgAlpha(0.82f);
            bool cameraEdited = false;
            if (ImGui::Begin(
                    "Sponza1K NTC",
                    nullptr,
                    ImGuiWindowFlags_AlwaysAutoResize |
                        ImGuiWindowFlags_NoSavedSettings)) {
                ImGui::Text(
                    "FPS: %.1f  (%.2f ms)",
                    ImGui::GetIO().Framerate,
                    deltaSeconds * 1000.0f);
                ImGui::Text(
                    "%s G-buffer GPU: %.3f ms",
                    sceneLabel(sceneMode),
                    latestStageMs[kTimestampGBuffer]);
                ImGui::Text(
                    "NTC decode: %s",
                    ntcEnabled ? "enabled" : "disabled");
                ImGui::Text(
                    "Backface culling: %s",
                    backfaceCulling ? "enabled" : "disabled");
                ImGui::Text(
                    "NTC shading rate: %s",
                    useVrs2x2 ? "2x2" : "1x1");
                ImGui::Text(
                    "Decode mode: %s",
                    featuresOnly
                        ? "features only"
                        : (mlpOnly ? "MLP only" : "full"));
                ImGui::Text(
                    "Resolution: %u x %u",
                    gfxGetBackBufferWidth(context),
                    gfxGetBackBufferHeight(context));
                ImGui::Separator();
                ImGui::TextUnformatted("Camera transform");
                float cameraPosition[3]{
                    camera.position.x,
                    camera.position.y,
                    camera.position.z,
                };
                if (ImGui::InputFloat3(
                        "Position", cameraPosition, "%.3f")) {
                    camera.position = glm::vec3(
                        cameraPosition[0],
                        cameraPosition[1],
                        cameraPosition[2]);
                    cameraEdited = true;
                }
                float cameraRotationDegrees[2]{
                    glm::degrees(camera.yaw),
                    glm::degrees(camera.pitch),
                };
                if (ImGui::InputFloat2(
                        "Yaw / Pitch (deg)",
                        cameraRotationDegrees,
                        "%.2f")) {
                    camera.yaw = glm::radians(
                        cameraRotationDegrees[0]);
                    camera.pitch = std::clamp(
                        glm::radians(cameraRotationDegrees[1]),
                        -1.35f,
                        1.35f);
                    mouseLook.dragging = false;
                    cameraEdited = true;
                }
                ImGui::Checkbox(
                    "Lock camera controls", &cameraControlsLocked);
                if (ImGui::Button("Reset camera")) {
                    camera = Camera{};
                    mouseLook.dragging = false;
                    cameraEdited = true;
                }
                ImGui::Separator();
                int selectedScene =
                    static_cast<int>(sceneMode);
                const char* sceneNames[] = {
                    "Sponza 1K NTC H64/L3",
                    "Sponza 1K NTC H32/L4",
                    "Sponza 1K NTC H32/2 hidden Matrix Core",
                    "Sponza 1K NTC H32/2 hidden Scalar ALU",
                    "Native Sponza 1K",
                    "Native Sponza 1K vs NTC H32/2 hidden",
                    "NTC H64/L3 vs H32/L4",
                    "Native Sponza 1K vs NTC H64/L3",
                    "BC equal storage vs NTC H64/L3",
                };
                if (ImGui::Combo(
                        "Scene",
                        &selectedScene,
                        sceneNames,
                        static_cast<int>(std::size(sceneNames)))) {
                    sceneMode =
                        static_cast<SceneMode>(selectedScene);
                }
                ImGui::TextUnformatted(
                    "RMB drag look | WASD move | Shift accelerate");
            }
            ImGui::End();
            if (cameraEdited) {
                viewProjection = makeSponzaViewProjection(
                    camera,
                    gfxGetBackBufferWidth(context),
                    gfxGetBackBufferHeight(context));
                previousViewProjection = viewProjection;
            }
            require(
                gfxProgramSetParameter(
                    context,
                    render.depthProgram,
                    "ViewProjection",
                    viewProjection),
                "Update prepass ViewProjection");
            require(
                gfxProgramSetParameter(
                    context,
                    render.gbufferProgram,
                    "ViewProjection",
                    viewProjection),
                "Update ViewProjection");
            require(
                gfxProgramSetParameter(
                    context,
                    render.gbufferProgram,
                    "PreviousViewProjection",
                    previousViewProjection),
                "Update PreviousViewProjection");
            if (isComparisonScene(sceneMode)) {
                ImDrawList* drawList = ImGui::GetForegroundDrawList();
                const ImVec2 displaySize = ImGui::GetIO().DisplaySize;
                const float splitX = displaySize.x * 0.5f;
                drawList->AddLine(
                    ImVec2(splitX, 0.0f),
                    ImVec2(splitX, displaySize.y),
                    IM_COL32(255, 210, 64, 255),
                    2.0f);
                drawList->AddText(
                    ImVec2(18.0f, displaySize.y - 34.0f),
                    IM_COL32_WHITE,
                    (sceneMode == SceneMode::NativeVsNtc ||
                     sceneMode == SceneMode::NativeVsNtcH32L2)
                        ? "Native Sponza 1K"
                        : (sceneMode == SceneMode::BcVsNtc
                            ? "BC7/BC5 348x348"
                            : "NTC H64/L3"));
                drawList->AddText(
                    ImVec2(splitX + 18.0f, displaySize.y - 34.0f),
                    IM_COL32_WHITE,
                    sceneMode == SceneMode::NtcH64VsH32
                        ? "NTC H32/L4"
                        : (sceneMode == SceneMode::NativeVsNtcH32L2
                            ? "NTC H32/2 hidden online decode"
                            : "NTC H64/L3 online decode"));
            }

            require(
                gfxCommandBeginTimestampQuery(
                    context, render.timestamps[kTimestampFrame]),
                "Begin total GPU timestamp");
            require(
                gfxCommandBeginTimestampQuery(
                    context, render.timestamps[kTimestampClear]),
                "Begin target-clear timestamp");
            for (GfxTexture texture : render.gbuffer) {
                require(
                    gfxCommandClearTexture(context, texture),
                    "Clear G-buffer attachment");
            }
            require(
                gfxCommandClearTexture(context, render.depth),
                "Clear G-buffer depth");
            require(
                gfxCommandEndTimestampQuery(
                    context, render.timestamps[kTimestampClear]),
                "End target-clear timestamp");
            require(
                gfxCommandBeginTimestampQuery(
                    context,
                    render.timestamps[kTimestampDepthPrepass]),
                "Begin depth-prepass timestamp");
            require(
                gfxCommandBindDepthStencilTarget(
                    context, render.depth),
                "Bind prepass depth target");
            require(
                gfxCommandBindKernel(
                    context, render.depthKernel),
                "Bind Sponza depth kernel");
            require(
                gfxCommandBindVertexBuffer(
                    context, render.vertexBuffer),
                "Bind prepass vertex buffer");
            require(
                gfxCommandBindIndexBuffer(
                    context, render.indexBuffer),
                "Bind prepass index buffer");
            for (const ddgi::dx12::MaterialDrawBatch& batch :
                 sponza.materialBatches) {
                require(
                    gfxCommandDrawIndexed(
                        context,
                        batch.indexCount,
                        1,
                        batch.firstIndex),
                    "Draw depth material batch");
            }
            require(
                gfxCommandEndTimestampQuery(
                    context,
                    render.timestamps[kTimestampDepthPrepass]),
                "End depth-prepass timestamp");
            if (useVrs2x2) {
                vrsCommandList->RSSetShadingRate(
                    D3D12_SHADING_RATE_2X2,
                    nullptr);
                require(
                    gfxResetCommandListState(context),
                    "Reset gfx state after setting 2x2 VRS");
            }

            for (std::uint32_t target = 0;
                 target < render.gbuffer.size();
                 ++target) {
                require(
                    gfxCommandBindColorTarget(
                        context, target, render.gbuffer[target]),
                    "Bind G-buffer color target");
            }
            require(
                gfxCommandBindDepthStencilTarget(context, render.depth),
                "Bind G-buffer depth target");
            require(
                gfxCommandBindKernel(context, render.gbufferKernel),
                "Bind Sponza NTC kernel");
            require(
                gfxCommandBindVertexBuffer(context, render.vertexBuffer),
                "Bind Sponza vertex buffer");
            require(
                gfxCommandBindIndexBuffer(context, render.indexBuffer),
                "Bind Sponza index buffer");
            require(
                gfxCommandBeginTimestampQuery(
                    context, render.timestamps[kTimestampGBuffer]),
                "Begin G-buffer timestamp");

            const auto drawMaterialBatches = [&](MaterialPath path) {
                const bool useH32Path = path == MaterialPath::NtcH32;
                const bool useH32L2Path =
                    path == MaterialPath::NtcH32L2 ||
                    path == MaterialPath::NtcH32L2Scalar;
                const bool useScalarPath =
                    path == MaterialPath::NtcH32L2Scalar;
                const bool useNativePath =
                    path == MaterialPath::Native;
                require(
                    gfxCommandBindKernel(
                        context,
                        useNativePath
                            ? render.gbufferNativeKernel
                            : (useScalarPath
                                ? render.gbufferH32L2ScalarKernel
                                : (useH32L2Path
                                ? render.gbufferH32L2Kernel
                                : (useH32Path
                                    ? render.gbufferH32Kernel
                                    : render.gbufferKernel)))),
                    useNativePath
                        ? "Bind native Sponza 1K kernel"
                        : (useScalarPath
                        ? "Bind Sponza NTC H32/2-hidden Scalar ALU kernel"
                        : (useH32L2Path
                        ? "Bind Sponza NTC H32/2-hidden kernel"
                        : (useH32Path
                            ? "Bind Sponza NTC H32 kernel"
                            : "Bind Sponza NTC H64 kernel"))));
                // Reverse material order so coplanar depth-equal fragments
                // preserve the original first-writer-wins result.
                for (auto batchIterator =
                         sponza.materialBatches.rbegin();
                     batchIterator != sponza.materialBatches.rend();
                     ++batchIterator) {
                    const ddgi::dx12::MaterialDrawBatch& batch =
                        *batchIterator;
                    const bool hasCompressedMaterial =
                        batch.ntcSlot != UINT32_MAX;
                    const bool useBc =
                        path == MaterialPath::Bc &&
                        hasCompressedMaterial;
                    const bool useNative =
                        path == MaterialPath::Native &&
                        hasCompressedMaterial;
                    const bool useNtc =
                        (path == MaterialPath::NtcH64 ||
                         path == MaterialPath::NtcH32 ||
                         path == MaterialPath::NtcH32L2 ||
                         path == MaterialPath::NtcH32L2Scalar) &&
                        ntcEnabled &&
                        hasCompressedMaterial;
                    const std::uint32_t slot =
                        hasCompressedMaterial ? batch.ntcSlot : 0u;
                    const ddgi::dx12::NtcMaterialGpuData& ntcMaterial =
                        useH32L2Path
                            ? ntcH32L2Materials[slot]
                            : (useH32Path
                                ? ntcH32Materials[slot]
                                : ntcMaterials[slot]);
                    const ddgi::dx12::NativeMaterialGpuData&
                        nativeMaterial = nativeMaterials[slot];
                    const ddgi::dx12::BcMaterialGpuData& bcMaterial =
                        bcMaterials[slot];
                    require(
                        gfxProgramSetParameter(
                            context,
                            render.gbufferProgram,
                            "HighGrid",
                            ntcMaterial.highGrid),
                        "Bind material high grid");
                    require(
                        gfxProgramSetParameter(
                            context,
                            render.gbufferProgram,
                            "LowGrid",
                            ntcMaterial.lowGrid),
                        "Bind material low grid");
                    require(
                        gfxProgramSetParameter(
                            context,
                            render.gbufferProgram,
                            "WeightBuffer",
                            useScalarPath
                                ? *ntcMaterial.scalarWeights
                                : *ntcMaterial.weights),
                        "Bind material MLP weights");
                    require(
                        gfxProgramSetParameter(
                            context,
                            render.gbufferProgram,
                            "WeightMatrixSizeFirst",
                            useScalarPath
                                ? ntcMaterial.scalarFirstMatrixSize
                                : ntcMaterial.firstMatrixSize),
                        "Bind first matrix size");
                    require(
                        gfxProgramSetParameter(
                            context,
                            render.gbufferProgram,
                            "WeightMatrixSizeHidden",
                            useScalarPath
                                ? ntcMaterial.scalarHiddenMatrixSize
                                : ntcMaterial.hiddenMatrixSize),
                        "Bind hidden matrix size");
                    require(
                        gfxProgramSetParameter(
                            context,
                            render.gbufferProgram,
                            "BcDiffuse",
                            bcMaterial.diffuse),
                        "Bind BC diffuse");
                    require(
                        gfxProgramSetParameter(
                            context,
                            render.gbufferProgram,
                            "BcNormal",
                            bcMaterial.normal),
                        "Bind BC normal");
                    require(
                        gfxProgramSetParameter(
                            context,
                            render.gbufferProgram,
                            "BcMetallicRoughness",
                            bcMaterial.metallicRoughness),
                        "Bind BC material texture");
                    require(
                        gfxProgramSetParameter(
                            context,
                            render.gbufferProgram,
                            "BcSampler",
                            render.bcSampler),
                        "Bind BC sampler");
                    require(
                        gfxProgramSetParameter(
                            context,
                            render.gbufferProgram,
                            "NativeDiffuse",
                            nativeMaterial.diffuse),
                        "Bind native diffuse");
                    require(
                        gfxProgramSetParameter(
                            context,
                            render.gbufferProgram,
                            "NativeNormal",
                            nativeMaterial.normal),
                        "Bind native normal");
                    require(
                        gfxProgramSetParameter(
                            context,
                            render.gbufferProgram,
                            "NativeRoughness",
                            nativeMaterial.roughness),
                        "Bind native roughness");
                    require(
                        gfxProgramSetParameter(
                            context,
                            render.gbufferProgram,
                            "NativeMetallic",
                            nativeMaterial.metallic),
                        "Bind native metallic");
                    require(
                        gfxProgramSetParameter(
                            context,
                            render.gbufferProgram,
                            "UseNtc",
                            useNtc ? 1u : 0u),
                        "Bind NTC enable flag");
                    require(
                        gfxProgramSetParameter(
                            context,
                            render.gbufferProgram,
                            "UseBc",
                            useBc ? 1u : 0u),
                        "Bind BC enable flag");
                    require(
                        gfxProgramSetParameter(
                            context,
                            render.gbufferProgram,
                            "UseNative",
                            useNative ? 1u : 0u),
                        "Bind native enable flag");
                    require(
                        gfxProgramSetParameter(
                            context,
                            render.gbufferProgram,
                            "HasNormalMap",
                            batch.hasNormalMap ? 1u : 0u),
                        "Bind normal map flag");
                    require(
                        gfxProgramSetParameter(
                            context,
                            render.gbufferProgram,
                            "DecodeMode",
                            decodeMode),
                        "Bind NTC decode mode");
                    require(
                        gfxProgramSetConstants(
                            context,
                            render.gbufferProgram,
                            "BaseColorFactor",
                            batch.baseColorFactor,
                            sizeof(batch.baseColorFactor)),
                        "Bind base color factor");
                    require(
                        gfxProgramSetParameter(
                            context,
                            render.gbufferProgram,
                            "RoughnessFactor",
                            batch.roughnessFactor),
                        "Bind roughness factor");
                    require(
                        gfxProgramSetParameter(
                            context,
                            render.gbufferProgram,
                            "MetallicFactor",
                            batch.metallicFactor),
                        "Bind metallic factor");
                    require(
                        gfxCommandDrawIndexed(
                            context,
                            batch.indexCount,
                            1,
                            batch.firstIndex),
                        "Draw material batch");
                }
            };

            const std::int32_t renderWidth =
                static_cast<std::int32_t>(
                    gfxGetBackBufferWidth(context));
            const std::int32_t renderHeight =
                static_cast<std::int32_t>(
                    gfxGetBackBufferHeight(context));
            if (isComparisonScene(sceneMode)) {
                const std::int32_t splitX = renderWidth / 2;
                require(
                    gfxCommandSetScissorRect(
                        context, 0, 0, splitX, renderHeight),
                    "Set comparison left-half scissor");
                const MaterialPath leftPath =
                    (sceneMode == SceneMode::NativeVsNtc ||
                     sceneMode == SceneMode::NativeVsNtcH32L2)
                        ? MaterialPath::Native
                        : (sceneMode == SceneMode::BcVsNtc
                            ? MaterialPath::Bc
                            : MaterialPath::NtcH64);
                drawMaterialBatches(leftPath);
                require(
                    gfxCommandSetScissorRect(
                        context,
                        splitX,
                        0,
                        renderWidth - splitX,
                        renderHeight),
                    "Set NTC half scissor");
                drawMaterialBatches(
                    sceneMode == SceneMode::NtcH64VsH32
                        ? MaterialPath::NtcH32
                        : (sceneMode == SceneMode::NativeVsNtcH32L2
                            ? MaterialPath::NtcH32L2
                            : MaterialPath::NtcH64));
            } else {
                require(
                    gfxCommandSetScissorRect(
                        context, 0, 0, renderWidth, renderHeight),
                    "Set full-frame scissor");
                drawMaterialBatches(
                    sceneMode == SceneMode::NtcH32
                        ? MaterialPath::NtcH32
                        : (sceneMode == SceneMode::NtcH32L2Scalar
                            ? MaterialPath::NtcH32L2Scalar
                            : (sceneMode == SceneMode::NtcH32L2
                            ? MaterialPath::NtcH32L2
                            : (sceneMode == SceneMode::Native
                                ? MaterialPath::Native
                                : MaterialPath::NtcH64))));
            }
            require(
                gfxCommandSetScissorRect(
                    context, 0, 0, renderWidth, renderHeight),
                "Restore full-frame scissor");

            require(
                gfxCommandEndTimestampQuery(
                    context, render.timestamps[kTimestampGBuffer]),
                "End G-buffer timestamp");
            if (useVrs2x2) {
                vrsCommandList->RSSetShadingRate(
                    D3D12_SHADING_RATE_1X1,
                    nullptr);
                require(
                    gfxResetCommandListState(context),
                    "Reset gfx state after restoring 1x1 VRS");
            }
            require(
                gfxCommandBeginTimestampQuery(
                    context, render.timestamps[kTimestampComposite]),
                "Begin G-buffer composite timestamp");
            require(
                gfxCommandBindColorTarget(
                    context, 0, render.finalColor),
                "Bind final color target");
            require(
                gfxCommandBindKernel(context, render.compositeKernel),
                "Bind G-buffer composite kernel");
            require(
                gfxCommandDraw(context, 3),
                "Draw G-buffer composite");
            require(
                gfxCommandEndTimestampQuery(
                    context, render.timestamps[kTimestampComposite]),
                "End G-buffer composite timestamp");
            require(
                gfxCommandBeginTimestampQuery(
                    context, render.timestamps[kTimestampUi]),
                "Begin UI timestamp");
            require(
                gfxImGuiRender(render.uiOverlay),
                "Render FPS overlay");
            require(
                gfxImGuiComposite(
                    render.finalColor, render.uiOverlay),
                "Composite FPS overlay");
            require(
                gfxCommandEndTimestampQuery(
                    context, render.timestamps[kTimestampUi]),
                "End UI timestamp");
            require(
                gfxCommandEndTimestampQuery(
                    context, render.timestamps[kTimestampFrame]),
                "End total GPU timestamp");
            require(
                gfxCommandResolveTimestamp(context),
                "Resolve stage timestamps");
            require(gfxFrame(context, vsync), "Present Sponza frame");
            require(
                gfxCommandUpdateTimestamp(context),
                "Update stage timestamps");
            for (std::size_t stage = 0;
                 stage < kTimestampCount;
                 ++stage) {
                latestStageMs[stage] = gfxTimestampQueryGetDuration(
                    context, render.timestamps[stage]);
            }
            if (frameIndex > 1) {
                for (std::size_t stage = 0;
                     stage < kTimestampCount;
                     ++stage) {
                    stageTimes[stage].push_back(latestStageMs[stage]);
                }
            }
            ++frameIndex;
        }
        require(gfxFinish(context), "Finish Sponza rendering");
        if (!capturePath.empty()) {
            ddgi::dx12::captureRgba8Texture(
                context, render.finalColor, capturePath);
            std::cout << "Capture: " << capturePath.string() << '\n';
        }

        if (!stageTimes[kTimestampFrame].empty()) {
            std::cout << std::format(
                "Benchmark resolution: {}x{}, Scene={}, VSync={}, "
                "NTC decode={}, Culling={}, VRS={}\n",
                gfxGetBackBufferWidth(context),
                gfxGetBackBufferHeight(context),
                sceneLabel(sceneMode),
                vsync ? "on" : "off",
                sceneMode == SceneMode::Native
                    ? "not used"
                    : (ntcEnabled ? "on" : "off"),
                backfaceCulling ? "back" : "none",
                useVrs2x2 ? "2x2" : "1x1");
            for (std::size_t stage = 0;
                 stage < kTimestampCount;
                 ++stage) {
                std::vector<float>& samples = stageTimes[stage];
                std::sort(samples.begin(), samples.end());
                const float median = samples[samples.size() / 2];
                const std::size_t p90Index = std::min(
                    samples.size() - 1,
                    static_cast<std::size_t>(
                        samples.size() * 0.9f));
                std::cout << std::format(
                    "  {:<22} {:>7.3f} ms median, "
                    "{:>7.3f} ms P90\n",
                    kTimestampStageNames[stage],
                    median,
                    samples[p90Index]);
            }
            std::cout << std::format(
                "  Samples/materials     {:>7} / {}\n",
                stageTimes[kTimestampFrame].size(),
                sponza.materialBatches.size());
            std::cout << std::format(
                "  Depth/G-buffer draws   {:>7} / {}\n",
                sponza.materialBatches.size(),
                sponza.materialBatches.size() *
                    (isComparisonScene(sceneMode) ? 2 : 1));
        }
        std::cout << "Rendered " << frameIndex
                  << " Sponza1K frame(s) with per-material draw calls "
                  << "(Scene=" << sceneLabel(sceneMode) << ").\n";

        destroyRenderResources(context, render);
        ddgi::dx12::destroyNtcMaterials(context, ntcMaterials);
        ddgi::dx12::destroyNtcMaterials(context, ntcH32Materials);
        ddgi::dx12::destroyNtcMaterials(context, ntcH32L2Materials);
        ddgi::dx12::destroyNativeMaterials(
            context, nativeMaterials);
        ddgi::dx12::destroyBcMaterials(context, bcMaterials);
        require(gfxImGuiTerminate(), "Terminate ImGui");
        imguiInitialized = false;
        require(gfxDestroyContext(context), "Destroy DX12 context");
        context = {};
        require(gfxDestroyWindow(window), "Destroy window");
        window = {};
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "sponza1k_ntc_dx12: " << e.what() << '\n';
        if (context) {
            destroyRenderResources(context, render);
            ddgi::dx12::destroyNtcMaterials(context, ntcMaterials);
            ddgi::dx12::destroyNtcMaterials(context, ntcH32Materials);
            ddgi::dx12::destroyNtcMaterials(context, ntcH32L2Materials);
            ddgi::dx12::destroyNativeMaterials(
                context, nativeMaterials);
            ddgi::dx12::destroyBcMaterials(context, bcMaterials);
            if (imguiInitialized) {
                gfxImGuiTerminate();
                imguiInitialized = false;
            }
            gfxDestroyContext(context);
        }
        if (window) {
            gfxDestroyWindow(window);
        }
        return EXIT_FAILURE;
    }
}
