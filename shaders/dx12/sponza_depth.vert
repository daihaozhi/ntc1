float4x4 ViewProjection;

struct VertexInput
{
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
};

float4 main(VertexInput input) : SV_Position
{
    return mul(ViewProjection, float4(input.position, 1.0f));
}
