#version 430 core
in vec2 TexCoord;

uniform sampler2D u_texture;
uniform float u_opacity;

out vec4 FragColor;

void main() {
    vec4 texel = texture(u_texture, TexCoord);
    texel.a *= u_opacity;
    if (texel.a < 0.01)
        discard;
    FragColor = texel;
}
