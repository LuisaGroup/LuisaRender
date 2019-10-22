#version 300 es

precision highp float;
precision highp sampler2DArray;
precision highp sampler2D;

layout (location = 0) out vec4 FragColor;

flat in float DiffuseTexId;
flat in float SpecularTexId;
flat in vec2 DiffuseTexOffset;
flat in vec2 DiffuseTexSize;
flat in vec2 SpecularTexOffset;
flat in vec2 SpecularTexSize;
in vec2 TexCoord;
in vec3 Position;
in vec3 Normal;
in vec3 Albedo;
in vec3 Specular;
in float Roughness;

uniform sampler2DArray textures;
uniform vec3 cameraPos;

uniform mat4 light_transform;
uniform sampler2D shadow_map;

#define M_PI        3.14159265358979323846264338327950288   /* pi             */
#define M_PI_2      1.57079632679489661923132169163975144   /* pi/2           */
#define M_PI_4      0.785398163397448309615660845819875721  /* pi/4           */
#define M_1_PI      0.318309886183790671537767526745028724  /* 1/pi           */
#define M_2_PI      0.636619772367581343075535053490057448  /* 2/pi           */

uniform vec3 lightDirection;
uniform vec3 lightEmission;

float DistributionGGX(vec3 m, vec3 n, float alpha)
{
    float cos_theta_m = dot(m, n);

    if (cos_theta_m <= 0.0f) {
        return 0.0f;
    }
    float root = alpha / (cos_theta_m * cos_theta_m * (alpha * alpha - 1.0f) + 1.0f);

    return root * root * M_1_PI;
}

float G1(vec3 v, vec3 m, vec3 n, float alpha) {

    float v_dot_n = dot(v, n);
    float v_dot_m = dot(v, m);

    if (v_dot_n * v_dot_m <= 0.0f) {
        return 0.0f;
    }

    float sqr_cos_theta_v = max(abs(v_dot_n * v_dot_n), 0.0001f);
    float sqr_tan_theta_v = (1.0f - sqr_cos_theta_v) / sqr_cos_theta_v;

    return 2.0f / (1.0f + sqrt(1.0f + alpha * alpha * sqr_tan_theta_v));
}

float Geo(vec3 i, vec3 o, vec3 m, vec3 n, float alpha) {
    return G1(i, m, n, alpha) * G1(o, m, n, alpha);
}

float sqr(float x) {
    return x * x;
}

float Fresnel(float HdotV, float eta) {
    float c = HdotV;
    float g = sqrt(eta * eta - 1.0f + c * c);
    return 0.5f * sqr(g - c) / sqr(g + c) * (1.0f + sqr(c * (g + c) - 1.0f) / sqr(c * (g - c) - 1.0f));
}

vec3 sample_texture(float id, vec2 tex_coord, vec2 offset, vec2 size, vec3 default_color, float gamma) {
    if (id < 0.0f) {
        return default_color;
    }
    if (tex_coord.x < 0.0f || tex_coord.x > 1.0f) { tex_coord.x = fract(fract(tex_coord.x + 1.0f)); }
    if (tex_coord.y < 0.0f || tex_coord.y > 1.0f) { tex_coord.y = fract(fract(tex_coord.y + 1.0f)); }
    vec2 coord = (tex_coord * size + offset) / vec2(textureSize(textures, 0).xy);
    vec3 sampled = texture(textures, vec3(coord, id)).rgb;
    return pow(sampled, vec3(gamma));
}

float ShadowCalculation()
{
    // perform perspective divide
    vec4 fragPosLightSpace = light_transform * vec4(Position, 1.0f);
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closestDepth = texture(shadow_map, projCoords.xy).r;
    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    // calculate bias (based on depth map resolution and slope)
    vec3 normal = normalize(Normal);
    vec3 lightDir = normalize(lightDirection);
    float bias = max(0.005f * (1.0 - dot(normal, lightDir)), 0.00005f);
    // PCF
    float shadow = 0.0;
    vec2 texelSize = 1.0 / vec2(textureSize(shadow_map, 0));
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            float pcfDepth = texture(shadow_map, projCoords.xy + vec2(x, y) * texelSize).r;
            shadow += currentDepth - bias > pcfDepth  ? 1.0 : 0.0;
        }
    }
    return shadow / 25.0;
}

void main()
{
    vec3 V = normalize(cameraPos - Position);
    vec3 N = normalize(dot(Normal, V) >= 0.0f ? Normal : -Normal);

    vec3 Kd = sample_texture(DiffuseTexId, TexCoord, DiffuseTexOffset, DiffuseTexSize, Albedo, 2.2f);
    vec3 Ks = sample_texture(SpecularTexId, TexCoord, SpecularTexOffset, SpecularTexSize, Specular, 2.2f);

    // reflectance equation
    vec3 Lo = vec3(0.0);

    // calculate per-light radiance
    vec3 L = normalize(lightDirection);

    float NdotL = dot(N, L);
    float NdotV = dot(N, V);

    if (NdotL > 0.0f) {

        vec3 radiance = lightEmission;
        vec3 specular = vec3(0.0f);

        float specular_strength = max(max(Ks.r, Ks.g), Ks.b);
        if (specular_strength > 0.0f) {
            vec3 H = normalize(V + L);
            float D = DistributionGGX(H, N, Roughness);
            float G = Geo(V, L, H, N, Roughness);
            float F = Fresnel(dot(H, V), 1.5f);
            float nominator = D * G * F;
            float denominator = 4.0f * abs(NdotV);
            specular = Ks * nominator / max(denominator, 0.001f);
        }
        vec3 diffuse = (1.0f - specular_strength) * Kd * abs(NdotL) * M_1_PI;
        float shadow = ShadowCalculation();
        Lo += (1.0f - shadow) * (diffuse + specular) * radiance;
    }
    Lo += 0.15f * Kd;  // ambient
    FragColor = vec4(pow(Lo, vec3(1.0f / 2.2f)), 1.0f);
}
