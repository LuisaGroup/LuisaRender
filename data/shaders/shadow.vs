#version 300 es

layout (location = 0) in vec3 aPosition;

uniform mat4 light_transform;

void main() {
    gl_Position = light_transform * vec4(aPosition, 1.0f);
}
