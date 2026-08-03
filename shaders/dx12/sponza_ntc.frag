#include "mlp.hlsl"

#ifndef NTC_LAYER_COUNT
#define NTC_LAYER_COUNT 3
#endif

#ifndef NTC_HIDDEN_DIM
#define NTC_HIDDEN_DIM 64
#endif

#ifndef NTC_NATIVE_ONLY
#define NTC_NATIVE_ONLY 0
#endif

Texture2D<float4> HighGrid;
Texture2D<float4> LowGrid;
ByteAddressBuffer WeightBuffer;
Texture2D<float4> BcDiffuse;
Texture2D<float4> BcNormal;
Texture2D<float4> BcMetallicRoughness;
Texture2D<float4> NativeDiffuse;
Texture2D<float4> NativeNormal;
Texture2D<float> NativeRoughness;
Texture2D<float> NativeMetallic;
SamplerState BcSampler;

uint WeightMatrixSizeFirst;
uint WeightMatrixSizeHidden;
uint UseNtc;
uint UseBc;
uint UseNative;
uint HasNormalMap;
uint DecodeMode;
float4 BaseColorFactor;
float RoughnessFactor;
float MetallicFactor;

#if !NTC_NATIVE_ONLY
using NtcLayerData = mininn::InferenceLayerDataRefNoBias<
    NTC_LAYER_COUNT,
    NTC_HIDDEN_DIM,
    dx::linalg::ComponentType::F16,
    (dx::linalg::MatrixLayoutEnum)MINIDXNN_WEIGHT_MATRIX_LAYOUT,
    dx::linalg::ComponentType::F16,
    mininn::LeakyReluActivation,
    mininn::IdentityActivation,
    dx::linalg::ComponentType::F16,
    MINIDXNN_WEIGHT_MATRIX_ALIGNMENT,
    MINIDXNN_WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT,
    16>;
#endif

struct PixelInput
{
    float4 position : SV_Position;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
    float3 world : POSITION0;
    float4 currentClip : POSITION1;
    float4 previousClip : POSITION2;
};

struct GBufferOutput
{
    float4 albedo : SV_Target0;
    float4 normal : SV_Target1;
    float4 material : SV_Target2;
    float4 emissive : SV_Target3;
    float2 motion : SV_Target4;
};

float triangleWave(float x)
{
    return 2.0f * abs(x - floor(x + 0.5f));
}

half unpackNibble(uint value, bool high)
{
    const uint nibble = high ? value >> 4 : value & 15;
    return (half)((float)nibble / 16.0f - 15.0f / 32.0f);
}

vector<half, 8> loadGridFeatures(
    Texture2D<float4> grid,
    int2 gridCoordinate)
{
    const float4 packed = grid.Load(int3(gridCoordinate, 0));
    const uint4 quantized =
        (uint4)floor(packed * 255.0f + 0.5f);
    vector<half, 8> features;
    features[0] = unpackNibble(quantized.r, false);
    features[1] = unpackNibble(quantized.r, true);
    features[2] = unpackNibble(quantized.g, false);
    features[3] = unpackNibble(quantized.g, true);
    features[4] = unpackNibble(quantized.b, false);
    features[5] = unpackNibble(quantized.b, true);
    features[6] = unpackNibble(quantized.a, false);
    features[7] = unpackNibble(quantized.a, true);
    return features;
}

vector<half, 8> sampleLowGrid(float2 uv)
{
    uint width;
    uint height;
    LowGrid.GetDimensions(width, height);
    const float2 position = uv * float2(width, height);
    const int2 p00 = min(int2(floor(position)), int2(width, height) - 1);
    const float2 weight = frac(position);
    const int2 p10 = min(p00 + int2(1, 0), int2(width, height) - 1);
    const int2 p01 = min(p00 + int2(0, 1), int2(width, height) - 1);
    const int2 p11 = min(p00 + int2(1, 1), int2(width, height) - 1);

    const vector<half, 8> f00 =
        loadGridFeatures(LowGrid, p00);
    const vector<half, 8> f10 =
        loadGridFeatures(LowGrid, p10);
    const vector<half, 8> f01 =
        loadGridFeatures(LowGrid, p01);
    const vector<half, 8> f11 =
        loadGridFeatures(LowGrid, p11);
    const vector<half, 8> top =
        lerp(f00, f10, (half)weight.x);
    const vector<half, 8> bottom =
        lerp(f01, f11, (half)weight.x);
    return lerp(top, bottom, (half)weight.y);
}

vector<half, 64> decodeFeatures(float2 sourceUv)
{
    const float2 uv = frac(sourceUv);
    const float2 positionalUv = frac(uv * 8.0f);
    vector<half, 64> features = (vector<half, 64>)0;

    [unroll]
    for (uint frequency = 0; frequency < 5; ++frequency)
    {
        const float2 scaled = positionalUv * exp2((float)frequency);
        features[frequency * 2 + 0] = (half)triangleWave(scaled.x);
        features[frequency * 2 + 1] = (half)triangleWave(scaled.y);
    }
    features[10] = (half)1.0f;
    features[11] = (half)1.0f;

    uint width;
    uint height;
    HighGrid.GetDimensions(width, height);
    const int2 base = min(
        int2(floor(uv * float2(width, height))),
        int2(width, height) - 1);
    const int2 offsets[4] = {
        int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1)
    };
    uint outputIndex = 12;
    [unroll]
    for (uint corner = 0; corner < 4; ++corner)
    {
        const int2 coordinate =
            min(base + offsets[corner], int2(width, height) - 1);
        const vector<half, 8> cornerFeatures =
            loadGridFeatures(HighGrid, coordinate);
        [unroll]
        for (uint channel = 0; channel < 8; ++channel)
        {
            features[outputIndex++] = cornerFeatures[channel];
        }
    }

    const vector<half, 8> lowFeatures = sampleLowGrid(uv);
    [unroll]
    for (uint channel = 0; channel < 8; ++channel)
    {
        features[outputIndex++] = lowFeatures[channel];
    }
    features[52] = (half)0.0f;
    return features;
}

float3 srgbToLinear(float3 value)
{
    const float3 low = value / 12.92f;
    const float3 high = pow((value + 0.055f) / 1.055f, 2.4f);
    return select(value <= 0.04045f, low, high);
}

float3 applyNormal(PixelInput input, float3 encoded)
{
    const float3 normal = normalize(input.normal);
    const float3 dp1 = ddx(input.world);
    const float3 dp2 = ddy(input.world);
    const float2 duv1 = ddx(input.uv);
    const float2 duv2 = ddy(input.uv);
    const float3 dp2Perp = cross(dp2, normal);
    const float3 dp1Perp = cross(normal, dp1);
    const float3 tangentUnnormalized =
        dp2Perp * duv1.x + dp1Perp * duv2.x;
    const float3 bitangentUnnormalized =
        dp2Perp * duv1.y + dp1Perp * duv2.y;
    const float inverseLength = rsqrt(max(
        dot(tangentUnnormalized, tangentUnnormalized),
        dot(bitangentUnnormalized, bitangentUnnormalized)));
    const float3 tangent = tangentUnnormalized * inverseLength;
    const float3 bitangent = bitangentUnnormalized * inverseLength;
    const float3 mapped = normalize(encoded * 2.0f - 1.0f);
    return normalize(
        tangent * mapped.x +
        bitangent * mapped.y +
        normal * mapped.z);
}

GBufferOutput main(PixelInput input)
{
    float3 albedo = BaseColorFactor.rgb;
    float3 encodedNormal = float3(0.5f, 0.5f, 1.0f);
    float roughness = RoughnessFactor;
    float metallic = MetallicFactor;

    if (UseNative != 0)
    {
        const float2 uv = frac(input.uv);
        albedo *= NativeDiffuse.Sample(BcSampler, uv).rgb;
        encodedNormal = NativeNormal.Sample(BcSampler, uv).rgb;
        roughness *= NativeRoughness.Sample(BcSampler, uv);
        metallic *= NativeMetallic.Sample(BcSampler, uv);
    }
#if !NTC_NATIVE_ONLY
    else if (UseBc != 0)
    {
        const float2 uv = frac(input.uv);
        albedo *= BcDiffuse.Sample(BcSampler, uv).rgb;
        const float2 encodedXy =
            BcNormal.Sample(BcSampler, uv).rg;
        const float2 normalXy = encodedXy * 2.0f - 1.0f;
        const float normalZ =
            sqrt(max(1.0f - dot(normalXy, normalXy), 0.0f));
        encodedNormal =
            float3(encodedXy, normalZ * 0.5f + 0.5f);
        const float2 materialSample =
            BcMetallicRoughness.Sample(BcSampler, uv).rg;
        roughness *= materialSample.r;
        metallic *= materialSample.g;
    }
    else if (UseNtc != 0)
    {
        vector<half, 64> features = (vector<half, 64>)0;
        if (DecodeMode != 2)
        {
            features = decodeFeatures(input.uv);
        }
        else
        {
            features[10] = (half)1.0f;
            features[11] = (half)1.0f;
        }

        if (DecodeMode == 1)
        {
            half featureSum = (half)0.0f;
            [unroll]
            for (uint featureIndex = 0;
                 featureIndex < 64;
                 ++featureIndex)
            {
                featureSum += features[featureIndex];
            }
            const float diagnostic =
                frac((float)featureSum * 0.03125f);
            albedo = diagnostic.xxx;
            encodedNormal = float3(0.5f, 0.5f, 1.0f);
            roughness = diagnostic;
            metallic = diagnostic;
        }
        else
        {
            NtcLayerData layerData;
            layerData.setWeightData(
                WeightBuffer,
                uint2(
                    WeightMatrixSizeFirst,
                    WeightMatrixSizeHidden));
            vector<half, 8> decoded = (vector<half, 8>)0;
            mininn::forward(decoded, features, layerData);

            albedo *= srgbToLinear(saturate(float3(
                decoded[0], decoded[1], decoded[2])));
            metallic *= saturate((float)decoded[3]);
            encodedNormal = saturate(float3(
                decoded[4], decoded[5], decoded[6]));
            roughness *= saturate((float)decoded[7]);
        }
    }
#endif

    const float3 normal =
        HasNormalMap != 0 &&
            (UseNtc != 0 || UseBc != 0 || UseNative != 0)
        ? applyNormal(input, encodedNormal)
        : normalize(input.normal);
    const float2 currentNdc =
        input.currentClip.xy / max(abs(input.currentClip.w), 1e-5f);
    const float2 previousNdc =
        input.previousClip.xy / max(abs(input.previousClip.w), 1e-5f);

    GBufferOutput output;
    output.albedo = float4(albedo, BaseColorFactor.a);
    output.normal = float4(normal * 0.5f + 0.5f, HasNormalMap);
    output.material = float4(
        clamp(roughness, 0.04f, 1.0f),
        saturate(metallic),
        1.0f,
        0.0f);
    output.emissive = 0.0f;
    output.motion = (currentNdc - previousNdc) * 0.5f;
    return output;
}
