float4x4 ViewProjection;
float4x4 PreviousViewProjection;

struct VertexInput
{
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
    float3 world : POSITION0;
    float4 currentClip : POSITION1;
    float4 previousClip : POSITION2;
};

VertexOutput main(VertexInput input)
{
    VertexOutput output;
    const float4 world = float4(input.position, 1.0f);
    output.position = mul(ViewProjection, world);
    output.normal = input.normal;
    output.uv = input.uv;
    output.world = input.position;
    output.currentClip = output.position;
    output.previousClip = mul(PreviousViewProjection, world);
    return output;
}
