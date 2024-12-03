#pragma once

#include <vector>
#include "TextRenderer.h"

struct GLFWwindow;
struct Detection;
class HUD
{
public:
	HUD();
	~HUD();

	void render(const std::vector<Detection> &detections, unsigned int fps, bool aim_active, float min_conf);
private:
	TextRenderer* text_renderer;

	unsigned int window_width;
	unsigned int window_height;

	GLFWwindow* window;
	HWND native_window_handle;
	void make_window_clickthrough(GLFWwindow* window);
};