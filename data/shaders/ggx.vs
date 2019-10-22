#version 300 es

layout (location = 0) in vec3 aPosition;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 aAlbedo;
layout (location = 3) in vec4 aGloss;
layout (location = 4) in vec2 aTexCoords;
layout (location = 5) in vec3 aTexIds;
layout (location = 6) in vec4 aDiffuseTexProperty;
layout (location = 7) in vec4 aSpecularTexProperty;
layout (location = 8) in vec4 aNormalTexProperty;

flat out float DiffuseTexId;
flat out float SpecularTexId;
flat out float NormalTexId;
flat out vec2 DiffuseTexOffset;
flat out vec2 DiffuseTexSize;
flat out vec2 SpecularTexOffset;
flat out vec2 SpecularTexSize;
flat out vec2 NormalTexOffset;
flat out vec2 NormalTexSize;
out vec2 TexCoord;
out vec3 Position;
out vec3 Normal;
out vec3 Albedo;
out vec3 Specular;
out float Roughness;

uniform mat4 view;
uniform mat4 projection;

void main() {

    Position = aPosition;
    Normal = aNormal;
    Albedo = aAlbedo;
    Specular = aGloss.rgb;
    Roughness = aGloss.a;

    TexCoord = aTexCoords;
    DiffuseTexId = aTexIds.x;
    SpecularTexId = aTexIds.y;
    NormalTexId = aTexIds.z;

    DiffuseTexOffset = aDiffuseTexProperty.xy;
    DiffuseTexSize = aDiffuseTexProperty.zw;
    SpecularTexOffset = aSpecularTexProperty.xy;
    SpecularTexSize = aSpecularTexProperty.zw;
    NormalTexOffset = aNormalTexProperty.xy;
    NormalTexSize = aNormalTexProperty.zw;

    Albedo = aAlbedo;

    gl_Position = projection * view * vec4(aPosition, 1.0f);
}
