#include "gpu.h"

void HandleError( cudaError_t err, const char *file, int line )
{
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

Gpu::Gpu()
{
	myId = 0;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
}

Gpu::Gpu(int major, int minor)
{
	setDevice(major, minor);
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
}

Gpu::~Gpu()
{
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
}

void Gpu::startTime()
{
    HANDLE_ERROR(cudaEventRecord(start, 0));
}

float Gpu::getTime()
{
    float elapsedTime;
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    return elapsedTime;
}

__global__ void kernelUsually(cudaPitchedPtr A, cudaPitchedPtr B, cudaPitchedPtr C)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < C.ysize && j < C.xsize) {
        Type sum = 0;
        for (int k = 0; k < A.xsize; k++)
            sum += get_elem(A, i, k) * get_elem(B, k, j);
        get_addr(C, i, j) = sum;
    }
}

__global__ void kernelTranspose(cudaPitchedPtr A, cudaPitchedPtr B, cudaPitchedPtr C)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < C.ysize && j < C.xsize) {
        Type sum = 0;
        for (int k = 0; k < A.xsize; k++)
            sum += get_elem(A, k, i) * get_elem(B, k, j);
        get_addr(C, i, j) = sum;
    }
}

__global__ void kernelSharedTask16(cudaPitchedPtr A, cudaPitchedPtr B, cudaPitchedPtr C)
{
	int y = threadIdx.y + blockIdx.y * blockDim.x;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = threadIdx.x + threadIdx.y * blockDim.x;

	extern __shared__ Type sharedMemory[];
	__shared__ Type *a;
	__shared__ Type *b;

	a = sharedMemory;
	b = sharedMemory + blockDim.x * blockDim.x;

	a[offset] = get_elem(A, y, blockDim.x * blockIdx.z + threadIdx.x);
	b[offset] = get_elem(B, blockDim.x * blockIdx.z + threadIdx.y, x);
	offset = threadIdx.x + (threadIdx.y + 16) * blockDim.x;
	a[offset] = get_elem(A, y + 16, blockDim.x * blockIdx.z + threadIdx.x);
	b[offset] = get_elem(B, blockDim.x * blockIdx.z + threadIdx.y + 16, x);

	__syncthreads();
	Type sum = 0;
	for (int k = 0; k < blockDim.x; k++)
		sum += a[blockDim.x * threadIdx.y + k]
			*  b[blockDim.x * k + threadIdx.x];
	atomicAdd(&get_addr(C, y, x), sum);

	sum = 0;
	for (int k = 0; k < blockDim.x; k++)
		sum += a[blockDim.x * (threadIdx.y + 16) + k]
			*  b[blockDim.x * k + threadIdx.x];
	atomicAdd(&get_addr(C, y + 16, x), sum);
}

__global__ void kernelSharedTask32(cudaPitchedPtr A, cudaPitchedPtr B, cudaPitchedPtr C)
{
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = threadIdx.x + threadIdx.y * blockDim.x;

	extern __shared__ Type sharedMemory[];
	__shared__ Type *a;
	__shared__ Type *b;

	a = sharedMemory;
	b = sharedMemory + blockDim.x * blockDim.x;

	Type sum = 0;
	a[offset] = get_elem(A, y, blockDim.x * blockIdx.z + threadIdx.x);
	b[offset] = get_elem(B, blockDim.y * blockIdx.z + threadIdx.y, x);
	__syncthreads();
	for (int k = 0; k < blockDim.x; k++)
		sum += a[blockDim.x * threadIdx.y + k]
			*  b[blockDim.x * k + threadIdx.x];
	atomicAdd(&get_addr(C, y, x), sum);
}

void Gpu::run(cudaPitchedPtr devA, cudaPitchedPtr devB, cudaPitchedPtr devC, int blockSizeX, int blockSizeY)
{
    dim3 block(blockSizeX, blockSizeY);
    dim3 grid((devC.xsize + blockSizeX - 1) / blockSizeX,
				(devC.ysize + blockSizeY - 1) / blockSizeY);

	//if (blockSizeX % warpSize())
        //kernelTranspose<<< grid, block >>>(devA, devB, devC);
	//else
    kernelUsually<<< grid, block >>>(devA, devB, devC);
}


cudaPitchedPtr Gpu::matmulEasyGPU(cudaPitchedPtr hostA, cudaPitchedPtr hostB, int blockSizeX, int blockSizeY)
{
    cudaPitchedPtr hostC = makeMatrix(hostA.ysize, hostB.xsize);

    size_t sizeA, sizeB, sizeC;
    sizeA = sizeof(Type) * hostA.ysize * hostA.xsize;
    sizeB = sizeof(Type) * hostB.ysize * hostB.xsize;
    sizeC = sizeof(Type) * hostC.ysize * hostC.xsize;

    cudaPitchedPtr devA, devB, devC;
    devA = make_cudaPitchedPtr(0, hostA.xsize * sizeof(Type), hostA.xsize, hostA.ysize);
    devB = make_cudaPitchedPtr(0, hostB.xsize * sizeof(Type), hostB.xsize, hostB.ysize);
    devC = make_cudaPitchedPtr(0, hostC.xsize * sizeof(Type), hostC.xsize, hostC.ysize);

    HANDLE_ERROR(cudaMalloc((void **)&devA.ptr, sizeA));
    HANDLE_ERROR(cudaMalloc((void **)&devB.ptr, sizeB));
    HANDLE_ERROR(cudaMalloc((void **)&devC.ptr, sizeC));

    HANDLE_ERROR(cudaMemcpy(devA.ptr, hostA.ptr, sizeA, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(devB.ptr, hostB.ptr, sizeB, cudaMemcpyHostToDevice));

	run(devA, devB, devC, blockSizeX, blockSizeY);

    HANDLE_ERROR(cudaMemcpy(hostC.ptr, devC.ptr, sizeC, cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(devA.ptr));
    HANDLE_ERROR(cudaFree(devB.ptr));
    HANDLE_ERROR(cudaFree(devC.ptr));

    HANDLE_ERROR(cudaEventRecord(stop, 0));

    return hostC;
}

cudaPitchedPtr Gpu::matmulPaddingGPU(cudaPitchedPtr hostA, cudaPitchedPtr hostB, int blockSizeX, int blockSizeY)
{
    cudaPitchedPtr hostC = makeMatrix(hostA.ysize, hostB.xsize);

    cudaPitchedPtr devA, devB, devC;
    devA = make_cudaPitchedPtr(0, 0, hostA.xsize, hostA.ysize);
    devB = make_cudaPitchedPtr(0, 0, hostB.xsize, hostB.ysize);
    devC = make_cudaPitchedPtr(0, 0, hostC.xsize, hostC.ysize);

    HANDLE_ERROR(cudaMallocPitch((void **)&devA.ptr, &devA.pitch, devA.xsize * sizeof(Type), devA.ysize));
    HANDLE_ERROR(cudaMallocPitch((void **)&devB.ptr, &devB.pitch, devB.xsize * sizeof(Type), devB.ysize));
    HANDLE_ERROR(cudaMallocPitch((void **)&devC.ptr, &devC.pitch, devC.xsize * sizeof(Type), devC.ysize));

    HANDLE_ERROR(cudaMemcpy2D(devA.ptr, devA.pitch,
							hostA.ptr, hostA.pitch,
							hostA.pitch, hostA.ysize,
							cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy2D(devB.ptr, devB.pitch,
							hostB.ptr, hostB.pitch,
							hostB.pitch, hostB.ysize,
							cudaMemcpyHostToDevice));

	run(devA, devB, devC, blockSizeX, blockSizeY);

    HANDLE_ERROR(cudaMemcpy2D(hostC.ptr, hostC.pitch,
							devC.ptr, devC.pitch,
							hostC.pitch, devC.ysize,
							cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(devA.ptr));
    HANDLE_ERROR(cudaFree(devB.ptr));
    HANDLE_ERROR(cudaFree(devC.ptr));

    HANDLE_ERROR(cudaEventRecord(stop, 0));

    return hostC;
}

cudaPitchedPtr Gpu::matmulSharedTaskGPU(cudaPitchedPtr hostA, cudaPitchedPtr hostB, int blockSizeX, int blockSizeY)
{
    cudaPitchedPtr hostC = makeMatrix(hostA.ysize, hostB.xsize);

	if (blockSizeY != 16)
		blockSizeY = 32;
	blockSizeX = 32;

	int newWidthA  = blockSizeX * ((hostA.xsize + blockSizeX - 1) / blockSizeX);
	int newHeightA = blockSizeX * ((hostA.ysize + blockSizeX - 1) / blockSizeX);
	int newWidthB  = blockSizeX * ((hostB.xsize + blockSizeX - 1) / blockSizeX);
	int newHeightB = blockSizeX * ((hostB.ysize + blockSizeX - 1) / blockSizeX);
	int newWidthC  = blockSizeX * ((hostC.xsize + blockSizeX - 1) / blockSizeX);
	int newHeightC = blockSizeX * ((hostC.ysize + blockSizeX - 1) / blockSizeX);

    cudaPitchedPtr devA, devB, devC;
    devA = make_cudaPitchedPtr(0, newWidthA * sizeof(Type), newWidthA, newHeightA);
    devB = make_cudaPitchedPtr(0, newWidthB * sizeof(Type), newWidthB, newHeightB);
    devC = make_cudaPitchedPtr(0, newWidthC * sizeof(Type), newWidthC, newHeightC);

    HANDLE_ERROR(cudaMalloc((void **)&devA.ptr, newWidthA * newHeightA * sizeof(Type)));
    HANDLE_ERROR(cudaMalloc((void **)&devB.ptr, newWidthB * newHeightB * sizeof(Type)));
    HANDLE_ERROR(cudaMalloc((void **)&devC.ptr, newWidthC * newHeightC * sizeof(Type)));

	HANDLE_ERROR(cudaMemset(devA.ptr, 0, newWidthA * newHeightA * sizeof(Type)));
	HANDLE_ERROR(cudaMemset(devB.ptr, 0, newWidthB * newHeightB * sizeof(Type)));
	HANDLE_ERROR(cudaMemset(devC.ptr, 0, newWidthC * newHeightC * sizeof(Type)));

    HANDLE_ERROR(cudaMemcpy2D(devA.ptr, devA.pitch,
							hostA.ptr, hostA.pitch,
							hostA.pitch, hostA.ysize,
							cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy2D(devB.ptr, devB.pitch,
							hostB.ptr, hostB.pitch,
							hostB.pitch, hostB.ysize,
							cudaMemcpyHostToDevice));

    dim3 block(blockSizeX, blockSizeY);
    dim3 grid((devC.xsize + blockSizeX - 1) / blockSizeX,
				(devC.ysize + blockSizeX - 1) / blockSizeX,
				(devA.ysize / blockSizeX));
	size_t sizeSharedMemory = 2 * blockSizeX * blockSizeX * sizeof(Type);
	//printf("grid.z = %d\ngrid.y = %d\ngrid.x = %d\n", grid.z, grid.y, grid.x);
	if (blockSizeY == 16)
        kernelSharedTask16<<< grid, block, sizeSharedMemory >>>(devA, devB, devC);
	else
        kernelSharedTask32<<< grid, block, sizeSharedMemory >>>(devA, devB, devC);

    HANDLE_ERROR(cudaMemcpy2D(hostC.ptr, hostC.pitch,
							devC.ptr, devC.pitch,
							hostC.pitch, hostC.ysize,
							cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(devA.ptr));
    HANDLE_ERROR(cudaFree(devB.ptr));
    HANDLE_ERROR(cudaFree(devC.ptr));

    HANDLE_ERROR(cudaEventRecord(stop, 0));

	return hostC;
}
