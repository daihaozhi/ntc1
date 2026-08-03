#include "mlp.hlsl"

ByteAddressBuffer WeightBuffer;
uint WeightMatrixSizeFirst;
uint WeightMatrixSizeHidden;

using ProbeLayerData = mininn::InferenceLayerDataRefNoBias<
    3,
    64,
    dx::linalg::ComponentType::F16,
    (dx::linalg::MatrixLayoutEnum)MINIDXNN_WEIGHT_MATRIX_LAYOUT,
    dx::linalg::ComponentType::F16,
    mininn::LeakyReluActivation,
    mininn::IdentityActivation,
    dx::linalg::ComponentType::F16,
    MINIDXNN_WEIGHT_MATRIX_ALIGNMENT,
    MINIDXNN_WEIGHT_MATRIX_VECTOR_STRIDE_ALIGNMENT,
    16>;

float4 main(float4 position : SV_Position) : SV_Target
{
    vector<half, 64> features = (vector<half, 64>)0;
    const float2 uv = position.xy / 1024.0f;

    [unroll]
    for (uint i = 0; i < 32; ++i)
    {
        const float frequency = exp2((float)(i & 7));
        features[i * 2 + 0] = (half)sin(uv.x * frequency);
        features[i * 2 + 1] = (half)cos(uv.y * frequency);
    }

    ProbeLayerData layerData;
    layerData.setWeightData(
        WeightBuffer,
        uint2(WeightMatrixSizeFirst, WeightMatrixSizeHidden));

    vector<half, 8> decoded = (vector<half, 8>)0;
    mininn::forward(decoded, features, layerData);

    return float4(
        saturate((float)decoded[0] * 0.5f + 0.5f),
        saturate((float)decoded[1] * 0.5f + 0.5f),
        saturate((float)decoded[2] * 0.5f + 0.5f),
        1.0f);
}
