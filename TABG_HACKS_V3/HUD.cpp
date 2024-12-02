#define GLFW_EXPOSE_NATIVE_WIN32


#include <Windows.h>
#include <stdexcept>
#include <YOLOv11.h>

#include "HUD.h"

#include <GL/GL.h>
#include <GL/GLU.h>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

HUD::HUD()
{
    // Initialize GLFW
    if (!glfwInit()) 
    {
        throw std::invalid_argument("Could not initialize GLFW");
    }

    // Set GLFW hints for transparency
    glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE);
    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE); // Remove window borders
    glfwWindowHint(GLFW_FOCUSED, GLFW_FALSE); // Prevent window from being focused on creation


    // Get primary monitor for fullscreen
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    // Create fullscreen window
    window = glfwCreateWindow(mode->width, mode->height, "HUD", nullptr, nullptr);
    if (!window) 
    {
        glfwTerminate();
        throw std::invalid_argument("GLFW could not create a window!");
    }

    window_width = mode->width;
    window_height = mode->height;

    // Make the window click-through
    make_window_clickthrough(window);

    // Make the OpenGL context current
    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        throw std::invalid_argument("Could not initialize GLEW!");
    }

    // Enable blending for transparency
    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    text_renderer = new TextRenderer("ProtestRevolution-Regular.ttf", mode->width, mode->height);
}

HUD::~HUD()
{
    delete text_renderer;
    glfwSetWindowShouldClose(window, GLFW_TRUE);
    glfwDestroyWindow(window);
    glfwTerminate();
}

/*
    // Calculate the scaling ratios between input and original image dimensions
    const float ratio_h = input_h / (float)image.rows;
    const float ratio_w = input_w / (float)image.cols;

    // Iterate over each detection
    for (int i = 0; i < output.size(); i++)
    {
        auto detection = output[i];
        auto box = detection.bbox;
        auto class_id = detection.class_id;
        auto conf = detection.conf;
        // Assign a color based on the class ID
        cv::Scalar color = cv::Scalar(COLORS[class_id][0], COLORS[class_id][1], COLORS[class_id][2]);

        // Adjust bounding box coordinates based on aspect ratio
        if (ratio_h > ratio_w)
        {
            box.x = box.x / ratio_w;
            box.y = (box.y - (input_h - ratio_w * image.rows) / 2) / ratio_w;
            box.width = box.width / ratio_w;
            box.height = box.height / ratio_w;
        }
        else
        {
            box.x = (box.x - (input_w - ratio_h * image.cols) / 2) / ratio_h;
            box.y = box.y / ratio_h;
            box.width = box.width / ratio_h;
            box.height = box.height / ratio_h;
        }

        // Draw the bounding box on the image
        rectangle(image, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), color, 3);
*/

void HUD::render(const std::vector<Detection> &detections, unsigned int fps, bool aim_active)
{
    // Clear with transparency (alpha 0)
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);


    // RENDER //
    glPointSize(100.0f); // Adjust size as needed
    glBegin(GL_POINTS);
    glColor4f(0.0f, 1.0f, 0.0f, 0.8f); // Semi-transparent red

    for (const auto &detection : detections)
    {
        std::cout << detection.bbox.x << "," << detection.bbox.y << "   ";
    }

    glVertex2f(-0.5f, -0.5f);
    glVertex2f(0.5f, -0.5f);
    glVertex2f(0.0f, 0.5f);
    glEnd();

    if (aim_active)
        text_renderer->RenderText("Aim ON", window_width - 500, window_height - 200, 1, {0, 255, 0});
    else
        text_renderer->RenderText("Aim OFF", window_width - 500, window_height - 200, 1, { 255, 0, 0 });

    std::string fps_text = "Execution Rate:" + std::to_string(fps) + "/s";
    text_renderer->RenderText(fps_text, window_width - 500, window_height - 300, 1, { 255, 255, 255 });

    //// Swap buffers and poll events
    glfwSwapBuffers(window);
    glfwPollEvents();
}

void HUD::make_window_clickthrough(GLFWwindow* window)
{
    HWND hwnd = glfwGetWin32Window(window);
    if (!hwnd) 
    {
        throw std::runtime_error("Could not get HWND from GLFW window");
    }
    native_window_handle = hwnd;

    // Set the window to be layered, transparent, and always on top
    LONG exStyle = GetWindowLong(hwnd, GWL_EXSTYLE);
    SetWindowLong(hwnd, GWL_EXSTYLE, exStyle | WS_EX_LAYERED | WS_EX_TRANSPARENT);

    SetLayeredWindowAttributes(hwnd, 0, 255, LWA_ALPHA);

    // Make the window always on top without activating (SWP_NOACTIVATE)
    SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
}


