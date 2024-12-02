#pragma once

#include <string>
#include <map>
#include <glm/glm.hpp>
#include <GL/glew.h>
#include <ft2build.h>
#include FT_FREETYPE_H

class TextRenderer {
public:
    TextRenderer(const std::string& fontPath, unsigned int screenWidth, unsigned int screenHeight);
    ~TextRenderer();

    void RenderText(const std::string& text, float x, float y, float scale, glm::vec3 color);

private:
    struct Character {
        GLuint TextureID;  // Texture ID
        glm::ivec2 Size;   // Size of glyph
        glm::ivec2 Bearing; // Offset from baseline
        GLuint Advance;    // Offset to next glyph
    };

    std::map<char, Character> Characters;
    GLuint VAO, VBO;
    GLuint shaderProgram;

    void LoadFont(const std::string& fontPath);
    void SetupShader(unsigned int screenWidth, unsigned int screenHeight);
    GLuint CompileShader(const std::string& source, GLenum type);
    std::string ReadFile(const std::string& filePath);
};
