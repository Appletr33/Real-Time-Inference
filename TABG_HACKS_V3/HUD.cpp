#define GLFW_EXPOSE_NATIVE_WIN32


#include <Windows.h>
#include <stdexcept>
#include <YOLOv11.h>

#include "HUD.h"

#include <GL/GL.h>
#include <GL/GLU.h>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <dwmapi.h>

HUD::HUD()
{
    // Initialize GLFW
    if (!glfwInit()) 
    {
        throw std::invalid_argument("Could not initialize GLFW");
    }

    // Set GLFW hints for transparency and window behavior
    glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE);
    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE); // Remove window borders
    glfwWindowHint(GLFW_FOCUSED, GLFW_FALSE); // Prevent window from being focused on creation
    glfwWindowHint(GLFW_FLOATING, GLFW_TRUE); // Ensure window stays above others
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); // Prevent window resizing


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

    text_renderer = new TextRenderer("Iceland-Regular.ttf", mode->width, mode->height);

    // Enable DWM composition for better window layering
    HRESULT dwmResult = DwmEnableComposition(DWM_EC_ENABLECOMPOSITION);
    if (FAILED(dwmResult)) {
        std::cerr << "Failed to enable DWM composition.\n";
    }
}

HUD::~HUD()
{
    delete text_renderer;
    glfwSetWindowShouldClose(window, GLFW_TRUE);
    glfwDestroyWindow(window);
    glfwTerminate();
}

// Function to convert screen coordinates to OpenGL NDC
float ToNDC_X(float screen_x, float screen_width) {
    return (2.0f * screen_x) / screen_width - 1.0f;
}

float ToNDC_Y(float screen_y, float screen_height) {
    return 1.0f - (2.0f * screen_y) / screen_height;
}


void HUD::render(const std::vector<Detection> &detections, unsigned int fps, bool aim_active, float min_conf)
{
    // Clear with transparency (alpha 0)
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Set line width for drawing boxes
    glLineWidth(2.0f); // Adjust as needed for visibility

    // Start rendering detections
    glColor4f(0.0f, 1.0f, 0.0f, 0.8f); // Semi-transparent green

    // Calculate the origin (top-left corner) of the 640x640 image in screen space
    float origin_x = (window_width / 2.0f) - 320.0f;
    float origin_y = (window_height / 2.0f) - 320.0f;

    glUseProgram(0);

    for (const auto& detection : detections) {
        if (detection.conf < min_conf)
            continue;


        //// Print detection info for debugging
        //std::cout << "Detection: x=" << detection.bbox.x
        //    << ", y=" << detection.bbox.y
        //    << ", width=" << detection.bbox.width
        //    << ", height=" << detection.bbox.height
        //    << std::endl;

        // Convert detection coordinates to screen space
        float screen_x_min = origin_x + detection.bbox.x;
        float screen_y_min = origin_y + detection.bbox.y;
        float screen_x_max = screen_x_min + detection.bbox.width;
        float screen_y_max = screen_y_min + detection.bbox.height;

        // Normalize screen coordinates to OpenGL NDC
        float norm_x_min = ToNDC_X(screen_x_min, window_width);
        float norm_y_min = ToNDC_Y(screen_y_min, window_height);
        float norm_x_max = ToNDC_X(screen_x_max, window_width);
        float norm_y_max = ToNDC_Y(screen_y_max, window_height);

        // Draw the bounding box
        //glBegin(GL_LINE_LOOP);
        //glVertex2f(norm_x_min, norm_y_min); // Bottom-left
        //glVertex2f(norm_x_max, norm_y_min); // Bottom-right
        //glVertex2f(norm_x_max, norm_y_max); // Top-right
        //glVertex2f(norm_x_min, norm_y_max); // Top-left
        //glEnd();
    }

    //glLineWidth(2.0f); // Adjust as needed for visibility
    //glColor4f(1.0f, 1.0f, 1.0f, 0.4f); // Semi-transparent green
    //glBegin(GL_LINE_LOOP);
    //for (int i = 0; i < 100; ++i) 
    //{
    //    float angle = 2.0f * 3.14159f * i / 100; // Calculate angle in radians
    //    float screen_x = (window_width / 2.0f) + 320.0f * cos(angle); // Circle center at screen center
    //    float screen_y = (window_height / 2.0f) + 320.0f * sin(angle); // Circle center at screen center
    //    float ndc_x = ToNDC_X(screen_x, window_width); // Convert screen X to NDC
    //    float ndc_y = ToNDC_Y(screen_y, window_height); // Convert screen Y to NDC
    //    glVertex2f(ndc_x, ndc_y); // Specify vertex
    //}
    //glEnd();

    if (!(GetAsyncKeyState(VK_RBUTTON) & 0x8000))
    {
        glColor4f(1.0f, 1.0f, 1.0f, 0.4f); // Semi-transparent green
        glPointSize(6.0f); // Adjust point size for visibility
        glBegin(GL_POINTS);
        glVertex2f(0.0f, 0.0f); // Center of the screen in normalized coordinates
        glEnd();
    }



    if (aim_active)
        text_renderer->RenderText("Aim ON", window_width - 300, window_height - 100, 1, {0, 255, 0});
    else
        text_renderer->RenderText("Aim OFF", window_width - 300, window_height - 100, 1, { 255, 0, 0 });

    std::string fps_text = "IPS:" + std::to_string(fps) + "/s";
    text_renderer->RenderText(fps_text, window_width - 300, window_height - 150, 1, { 255, 255, 255 });

    //// Swap buffers and poll events
    glfwSwapBuffers(window);
    glfwPollEvents();
}

// Function to make the window click-through and topmost
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
    SetWindowLong(hwnd, GWL_EXSTYLE, exStyle | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW);

    // Set layered window attributes for per-pixel alpha
    SetLayeredWindowAttributes(hwnd, 0, 255, LWA_ALPHA);

    // Make the window always on top without activating (SWP_NOACTIVATE)
    SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
}

unsigned int HUD::get_window_width()
{
    return window_width;
}

unsigned int HUD::get_window_height()
{
    return window_height;
}


