#ifndef  __GPU_HEADER__
#define  __GPU_HEADER__

#include <stdio.h>

#include "matrix.h"
#include "options.h"

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void HandleError(cudaError_t err, const char *file, int line);

class Gpu {
    cudaEvent_t start, stop;
	int myId;
	void printInfoDevice(int id);
	void run(cudaPitchedPtr A, cudaPitchedPtr B, cudaPitchedPtr C, int blockSizeX, int blockSizeY);
public:
    Gpu();
    Gpu(int major, int minor);
    ~Gpu();
	cudaPitchedPtr matmulEasyGPU(cudaPitchedPtr hostA, cudaPitchedPtr hostB, int blockSizeX, int blockSizeY);
    cudaPitchedPtr matmulPaddingGPU(cudaPitchedPtr hostA, cudaPitchedPtr hostB, int blockSizeX, int blockSizeY);
    cudaPitchedPtr matmulSharedTaskGPU(cudaPitchedPtr hostA, cudaPitchedPtr hostB, int blockSizeX, int blockSizeY);
    cudaPitchedPtr matmulGPU(cudaPitchedPtr hostA, cudaPitchedPtr hostB, int blockSizeX, int blockSizeY, Options opt);
	cudaPitchedPtr matmulFastGPU(cudaPitchedPtr hostA, cudaPitchedPtr hostB, Options opt);
    void startTime();
    float getTime();

	void infoDevices();
	void infoMyDevice();
	void setDevice(int major, int minor);
	int warpSize();
};

#endif /* __GPU_HEADER__ */
