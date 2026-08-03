Texture2D<float4> AlbedoBuffer;
Texture2D<float4> NormalBuffer;
Texture2D<float4> MaterialBuffer;
Texture2D<float4> EmissiveBuffer;

float4 main(float4 position : SV_Position) : SV_Target
{
    const int2 pixel = int2(position.xy);
    const float3 albedo = AlbedoBuffer.Load(int3(pixel, 0)).rgb;
    const float3 normal =
        normalize(NormalBuffer.Load(int3(pixel, 0)).rgb * 2.0f - 1.0f);
    const float roughness =
        MaterialBuffer.Load(int3(pixel, 0)).r;
    const float3 emissive =
        EmissiveBuffer.Load(int3(pixel, 0)).rgb;
    const float3 lightDirection =
        normalize(float3(0.35f, 0.85f, 0.4f));
    const float diffuse = saturate(dot(normal, lightDirection));
    float3 color =
        albedo * (0.12f + 1.35f * diffuse) +
        emissive +
        (1.0f - roughness) * pow(diffuse, 16.0f) * 0.12f;
    color = color / (1.0f + color);
    return float4(saturate(color), 1.0f);
}
