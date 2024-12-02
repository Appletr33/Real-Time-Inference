#version 330 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D text;
uniform vec3 textColor;

void main() {
    float alpha = texture(text, TexCoords).r; // Sample alpha from the texture
    FragColor = vec4(textColor, alpha);      // Apply color and alpha
}