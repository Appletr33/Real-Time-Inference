// ScreenCapture.cpp

#include "ScreenCapture.h"
#include <iostream>

ScreenCapture::ScreenCapture() : cudaResource(nullptr), isCudaRegistered(false)
{
    // Create the device and context
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0,
        D3D11_SDK_VERSION, &g_device, nullptr, &g_context);
    if (FAILED(hr)) {
        std::cerr << "Failed to create D3D11 device. HRESULT: " << std::hex << hr << std::endl;
        // Handle error appropriately
        return;
    }

    // Continue with duplication initialization
    InitializeDuplication();
}


ScreenCapture::~ScreenCapture()
{
    if (isCudaRegistered && cudaResource) {
        cudaGraphicsUnregisterResource(cudaResource);
    }

    if (g_duplication) {
        g_duplication->Release();
    }

    if (g_context) {
        g_context->ClearState();
    }

    if (g_device) {
        g_device->Release();
    }
}

HRESULT ScreenCapture::CaptureScreenRegion()
{
    // Acquire next frame with a 500 ms timeout
    ComPtr<IDXGIResource> desktopResource;
    DXGI_OUTDUPL_FRAME_INFO frameInfo;
    HRESULT hr = g_duplication->AcquireNextFrame(500, &frameInfo, &desktopResource);

    if (FAILED(hr)) {
        if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
            // No new frame available within the timeout
            return hr;
        }
        else if (hr == DXGI_ERROR_ACCESS_LOST) {
            // Access lost; reinitialize duplication
            std::cerr << "Access Lost. Reinitializing duplication." << std::endl;
            InitializeDuplication();
            return hr;
        }
        else {
            // Other errors
            std::cerr << "Failed to acquire next frame. HRESULT: " << std::hex << hr << std::endl;
            return hr;
        }
    }

    // Get the captured texture
    ComPtr<ID3D11Texture2D> desktopImage;
    hr = desktopResource.As(&desktopImage);
    if (FAILED(hr)) {
        std::cerr << "Failed to convert IDXGIResource to ID3D11Texture2D." << std::endl;
        g_duplication->ReleaseFrame();
        return hr;
    }

    // Get the desktop image description
    D3D11_TEXTURE2D_DESC desc;
    desktopImage->GetDesc(&desc);

    // Define the region to capture (centered 640x640)
    RECT screenRect;
    screenRect.left = (desc.Width - 640) / 2;
    screenRect.top = (desc.Height - 640) / 2;
    screenRect.right = screenRect.left + 640;
    screenRect.bottom = screenRect.top + 640;

    // Ensure coordinates are within bounds
    screenRect.left = max(screenRect.left, 0);
    screenRect.top = max(screenRect.top, 0);
    screenRect.right = min(screenRect.right, (LONG)desc.Width);
    screenRect.bottom = min(screenRect.bottom, (LONG)desc.Height);

    // Copy the specified region to the existing texture
    D3D11_BOX sourceRegion;
    sourceRegion.left = screenRect.left;
    sourceRegion.top = screenRect.top;
    sourceRegion.front = 0;
    sourceRegion.right = screenRect.right;
    sourceRegion.bottom = screenRect.bottom;
    sourceRegion.back = 1;

    g_context->CopySubresourceRegion(capturedTexture.Get(), 0, 0, 0, 0, desktopImage.Get(), 0, &sourceRegion);

    // Insert the event query
    g_context->End(g_query.Get());

    // Release the frame
    g_duplication->ReleaseFrame();

    return S_OK;
}

uint8_t* ScreenCapture::MapResource(size_t* size, cudaStream_t stream) {
    if (!isCudaRegistered || !cudaResource) {
        std::cerr << "CUDA resource is not registered or invalid." << std::endl;
        return nullptr;
    }

    // Wait for D3D11 commands to finish
    BOOL data = FALSE;
    while (S_OK != g_context->GetData(g_query.Get(), &data, sizeof(BOOL), 0)) {
        //Sleep(1); // Sleep to avoid busy-waiting
    }

    // Map the CUDA resource
    cudaError_t cudaStatus = cudaGraphicsMapResources(1, &cudaResource, stream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to map CUDA resource. Error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return nullptr;
    }

    // Get device pointer
    uint8_t* devPtr = nullptr;
    cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&devPtr, size, cudaResource);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to get mapped pointer. Error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Unmap resources before returning
        cudaGraphicsUnmapResources(1, &cudaResource, stream);
        return nullptr;
    }

    return devPtr;
}


void ScreenCapture::UnmapResource(cudaStream_t stream) {
    cudaError_t cudaStatus = cudaGraphicsUnmapResources(1, &cudaResource, stream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to unmap CUDA resource. Error: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
}

void ScreenCapture::InitializeDuplication()
{
    // Release existing duplication
    if (g_duplication) {
        g_duplication->Release();
        g_duplication = nullptr;
    }

    // Reinitialize duplication
    HRESULT hr = S_OK;
    // Get DXGI device
    ComPtr<IDXGIDevice> dxgiDevice;
    hr = g_device.As(&dxgiDevice);
    if (FAILED(hr)) {
        std::cerr << "Failed to get IDXGIDevice during reinitialization." << std::endl;
        return;
    }

    // Get adapter
    ComPtr<IDXGIAdapter> adapter;
    hr = dxgiDevice->GetAdapter(&adapter);
    if (FAILED(hr)) {
        std::cerr << "Failed to get IDXGIAdapter during reinitialization." << std::endl;
        return;
    }

    DXGI_ADAPTER_DESC adapterDesc;
    hr = adapter->GetDesc(&adapterDesc);
    if (FAILED(hr)) {
        std::cerr << "Failed to get adapter description. HRESULT: " << std::hex << hr << std::endl;
        return;
    }

    // Log the GPU description
    std::wcout << L"Selected GPU: " << adapterDesc.Description << std::endl;
    std::wcout << L"Vendor ID: " << adapterDesc.VendorId << L", Device ID: " << adapterDesc.DeviceId << std::endl;

    // Use cudaD3D11GetDevice to get the CUDA device corresponding to the DXGI adapter
    int cudaDevice = -1;
    cudaError_t cudaStatus = cudaD3D11GetDevice(&cudaDevice, adapter.Get());
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to get CUDA device from D3D11 adapter. Error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Handle error appropriately
        return;
    }
    else {
        // Set the CUDA device
        cudaStatus = cudaSetDevice(cudaDevice);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "Failed to set CUDA device. Error: " << cudaGetErrorString(cudaStatus) << std::endl;
            // Handle error appropriately
            return;
        }
    }

    // Initialize the CUDA context
    cudaStatus = cudaFree(0); // This initializes the CUDA context
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to initialize CUDA context. Error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Handle error appropriately
        return;
    }

    // Get output (monitor)
    ComPtr<IDXGIOutput> output;
    hr = adapter->EnumOutputs(0, &output);
    if (FAILED(hr)) {
        std::cerr << "Failed to get IDXGIOutput during reinitialization." << std::endl;
        return;
    }

    // Get output1
    ComPtr<IDXGIOutput1> output1;
    hr = output.As(&output1);
    if (FAILED(hr)) {
        std::cerr << "Failed to get IDXGIOutput1 during reinitialization." << std::endl;
        return;
    }

    // Create duplication
    hr = output1->DuplicateOutput(g_device.Get(), &g_duplication);
    if (FAILED(hr)) {
        std::cerr << "Failed to recreate duplication during reinitialization." << std::endl;
        return;
    }

    // Create the query for synchronization
    D3D11_QUERY_DESC queryDesc = {};
    queryDesc.Query = D3D11_QUERY_EVENT;
    queryDesc.MiscFlags = 0;
    hr = g_device->CreateQuery(&queryDesc, &g_query);
    if (FAILED(hr)) {
        std::cerr << "Failed to create D3D11 query. HRESULT: " << std::hex << hr << std::endl;
        return;
    }

    // Create the texture once
    D3D11_TEXTURE2D_DESC regionDesc = {};
    regionDesc.Width = 640;
    regionDesc.Height = 640;
    regionDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM; // Ensure compatible format
    regionDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    regionDesc.MiscFlags = 0;
    regionDesc.Usage = D3D11_USAGE_DEFAULT;
    regionDesc.CPUAccessFlags = 0;
    regionDesc.ArraySize = 1;
    regionDesc.MipLevels = 1;
    regionDesc.SampleDesc.Count = 1;

    hr = g_device->CreateTexture2D(&regionDesc, nullptr, &capturedTexture);
    if (FAILED(hr)) {
        std::cerr << "Failed to create texture for screen capture. HRESULT: " << std::hex << hr << std::endl;
        // Handle error appropriately
    }

    cudaStatus = cudaGraphicsD3D11RegisterResource(&cudaResource, capturedTexture.Get(), cudaGraphicsRegisterFlagsNone);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to register texture with CUDA. Error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }
    else {
        isCudaRegistered = true;
        std::cout << "Texture successfully registered with CUDA." << std::endl;
    }

    std::cout << "Duplication reinitialized successfully." << std::endl;
}