// ScreenCapture.h
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

using Microsoft::WRL::ComPtr;

class ScreenCapture {
public:
    ScreenCapture();
    ~ScreenCapture();
    HRESULT CaptureScreenRegion();
    void InitializeDuplication(); // To reinitialize duplication if needed
    cudaGraphicsResource* cudaResource;     // CUDA graphics resource
    ID3D11Texture2D* capturedTexture; // Persistent texture
    // New functions to handle resource mapping and unmapping
    cudaArray* MapResource(size_t* size, cudaStream_t stream);
    void UnmapResource(cudaStream_t stream);


private:
    ComPtr<ID3D11Device> g_device;
    ComPtr<ID3D11DeviceContext> g_context;
    ComPtr<IDXGIOutputDuplication> g_duplication;
    bool isCudaRegistered;
    // Query for synchronization
    ComPtr<ID3D11Query> g_query;
};