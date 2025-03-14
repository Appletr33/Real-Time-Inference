#include <windows.h>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>

#include "yolov11.h"
#include "HUD.h"
#include "Aim.h"
#include "ScreenCapture.h"


HHOOK keycallback_hook = nullptr;
// Atomic flags for thread-safe access
std::atomic<bool> should_terminate(false);
std::atomic<bool> aim_active(false);
// High-resolution timer aliases for convenience
using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

constexpr float MIN_CONF = 0.6f;

/**
 * @brief Setting up Tensorrt logger
*/
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}logger;


static LRESULT CALLBACK KeyboardProc(int nCode, WPARAM wParam, LPARAM lParam)
{
    if (nCode == HC_ACTION && wParam == WM_KEYDOWN)
    {
        KBDLLHOOKSTRUCT* kbStruct = (KBDLLHOOKSTRUCT*)lParam;
        if (kbStruct->vkCode == VK_OEM_3)
        {
            // Tilde key (`~`)
            should_terminate = true; // Signal termination
        }
        if (kbStruct->vkCode == 0x51)
        {
            // Q or q
            aim_active = !aim_active;
        }
    }
    return CallNextHookEx(NULL, nCode, wParam, lParam);
}

void KeyboardInputThread()
{
    HHOOK hook = SetWindowsHookEx(WH_KEYBOARD_LL, KeyboardProc, GetModuleHandle(NULL), 0);
    if (!hook)
    {
        throw std::runtime_error("Failed to set keyboard hook");
    }

    // Process messages in this thread
    MSG msg;
    while (!should_terminate.load())
    {
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Avoid busy-waiting
    }

    UnhookWindowsHookEx(hook);
}

int main(int argc, char* argv[]) 
{
    // Initialize COM
    HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (FAILED(hr)) {
        std::cerr << "Failed to initialize COM library. HRESULT: " << std::hex << hr << std::endl;
        return -1;
    }


    unsigned int fps = 0;  // Variable to store FPS
    unsigned int frameCount = 0;  // Count the number of iterations
    TimePoint startTime = Clock::now();  // Record the start time

    // Initialize the Logger
    Logger logger;

    // Define color codes for terminal output
    const std::string RED_COLOR = "\033[31m";
    const std::string GREEN_COLOR = "\033[32m";
    const std::string YELLOW_COLOR = "\033[33m";
    const std::string RESET_COLOR = "\033[0m";

    // Run hack if no arguments
    std::vector<Detection> detections;
    std::string enginePath = "best_model.engine";
    YOLOv11 yolov11(enginePath, logger);
    auto hud = HUD();
    auto aim = Aim(hud.get_window_width(), hud.get_window_height());
    auto screen_capture = ScreenCapture();
    std::thread inputThread(KeyboardInputThread);
    while (!should_terminate.load())
    {
        HRESULT hr = screen_capture.CaptureScreenRegion();
        if (FAILED(hr)) {
            if (hr == DXGI_ERROR_WAIT_TIMEOUT) 
            {
                // No new frame; continue the loop
                continue;
            }
            else 
            {
                std::cerr << RED_COLOR << "Could Not Capture Screen! HRESULT: " << std::hex << hr << RESET_COLOR << std::endl;
                break;
            }
        }

        // Map the resource and get the device pointer
        size_t size = 0;
        cudaArray* cudaArray = nullptr;
        cudaArray = screen_capture.MapResource(&size, *yolov11.get_stream());
        if (cudaArray != nullptr) {
            yolov11.preprocess(cudaArray);

            // Unmap the resource after preprocessing
            screen_capture.UnmapResource(*yolov11.get_stream());
        }

        yolov11.infer();
        yolov11.postprocess(detections);
        

        if (GetAsyncKeyState(VK_RBUTTON) & 0x8000) 
        {
            hud.render(detections, fps, false, MIN_CONF);
            aim.update_targets(detections, false);

        }
        else
        {
            hud.render(detections, fps, aim_active.load(), MIN_CONF);
            aim.update_targets(detections, aim_active.load());
        }
        
        detections.clear();
        frameCount++;

        TimePoint now = Clock::now();
        std::chrono::duration<float> elapsedTime = now - startTime;
        if (elapsedTime.count() >= 1.0f) 
        {
            fps = frameCount;  // Update FPS
            frameCount = 0;  // Reset frame count
            startTime = now;  // Reset timer
        }
    }
    // Uninitialize COM
    CoUninitialize();

    // Join the input thread before exiting
    if (inputThread.joinable())
    {
        inputThread.join();
    }

    return 0;
}