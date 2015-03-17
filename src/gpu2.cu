#include "gpu.h"

__global__ void kernelUsually(cudaPitchedPtr A, cudaPitchedPtr B, cudaPitchedPtr C);
__global__ void kernelSharedSquare(cudaPitchedPtr A, cudaPitchedPtr B, cudaPitchedPtr C);


cudaPitchedPtr Gpu::matmulGPU(cudaPitchedPtr hostA, cudaPitchedPtr hostB,
								int blockSizeX, int blockSizeY, Options opt)
{
    cudaPitchedPtr hostC = makeMatrix(hostA.ysize, hostB.xsize);

	//if (blockSizeX % warpSize())
		//hostA = transposeMatrix(hostA);

	int newWidthA, newHeightA, newWidthB, newHeightB, newWidthC, newHeightC;
	if (opt.sharedMemory()) {
		newWidthA  = blockSizeX * ((hostA.xsize + blockSizeX - 1) / blockSizeX);
		newHeightA = blockSizeY * ((hostA.ysize + blockSizeY - 1) / blockSizeY);
		newWidthB  = blockSizeX * ((hostB.xsize + blockSizeX - 1) / blockSizeX);
		newHeightB = blockSizeY * ((hostB.ysize + blockSizeY - 1) / blockSizeY);
		newWidthC  = blockSizeX * ((hostC.xsize + blockSizeX - 1) / blockSizeX);
		newHeightC = blockSizeY * ((hostC.ysize + blockSizeY - 1) / blockSizeY);
	} else {
		newWidthA  = hostA.xsize;
		newHeightA = hostA.ysize;
		newWidthB  = hostB.xsize;
		newHeightB = hostB.ysize;
		newWidthC  = hostC.xsize;
		newHeightC = hostC.ysize;
	}

    size_t sizeA, sizeB, sizeC;
    sizeA = sizeof(Type) * newWidthA * newHeightA;
    sizeB = sizeof(Type) * newWidthB * newHeightB;
    sizeC = sizeof(Type) * newWidthC * newHeightC;

    cudaPitchedPtr devA, devB, devC;
    devA = make_cudaPitchedPtr(0, newWidthA * sizeof(Type), newWidthA, newHeightA);
    devB = make_cudaPitchedPtr(0, newWidthB * sizeof(Type), newWidthB, newHeightB);
    devC = make_cudaPitchedPtr(0, newWidthC * sizeof(Type), newWidthC, newHeightC);

	if (opt.usePadding()) {
        HANDLE_ERROR(cudaMallocPitch((void **)&devA.ptr, &devA.pitch, devA.xsize * sizeof(Type), devA.ysize));
        HANDLE_ERROR(cudaMallocPitch((void **)&devB.ptr, &devB.pitch, devB.xsize * sizeof(Type), devB.ysize));
        HANDLE_ERROR(cudaMallocPitch((void **)&devC.ptr, &devC.pitch, devC.xsize * sizeof(Type), devC.ysize));
	} else {
        HANDLE_ERROR(cudaMalloc((void **)&devA.ptr, sizeA));
        HANDLE_ERROR(cudaMalloc((void **)&devB.ptr, sizeB));
        HANDLE_ERROR(cudaMalloc((void **)&devC.ptr, sizeC));
	}

	if (opt.sharedMemory()) {
		HANDLE_ERROR(cudaMemset(devA.ptr, 0, newWidthA * newHeightA * sizeof(Type)));
		HANDLE_ERROR(cudaMemset(devB.ptr, 0, newWidthB * newHeightB * sizeof(Type)));
		HANDLE_ERROR(cudaMemset(devC.ptr, 0, newWidthC * newHeightC * sizeof(Type)));
	}

	if (opt.sharedMemory() || opt.usePadding()) {
        HANDLE_ERROR(cudaMemcpy2D(devA.ptr, devA.pitch,
								hostA.ptr, hostA.pitch,
								hostA.pitch, hostA.ysize,
								cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy2D(devB.ptr, devB.pitch,
								hostB.ptr, hostB.pitch,
								hostB.pitch, hostB.ysize,
								cudaMemcpyHostToDevice));
	} else {
        HANDLE_ERROR(cudaMemcpy(devA.ptr, hostA.ptr, sizeA, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(devB.ptr, hostB.ptr, sizeB, cudaMemcpyHostToDevice));
	}

    dim3 block(blockSizeX, blockSizeY);
    dim3 grid((devC.xsize + blockSizeX - 1) / blockSizeX,
				(devC.ysize + blockSizeY - 1) / blockSizeY);
	printf("grid.Y = %d\ngrid.X = %d\n", grid.y, grid.x);

	if (opt.sharedMemory()) {
		size_t sizeSharedMemory = 2 * blockSizeX * blockSizeY * sizeof(Type);
        //kernelSharedSquare<<< grid, block, sizeSharedMemory >>>(devA, devB, devC);
	} else {
        kernelUsually<<< grid, block >>>(devA, devB, devC);
	}

	if (opt.sharedMemory() || opt.usePadding()) {
        HANDLE_ERROR(cudaMemcpy2D(hostC.ptr, hostC.pitch,
								devC.ptr, devC.pitch,
								hostC.pitch, hostC.ysize,
								cudaMemcpyDeviceToHost));
	} else {
        HANDLE_ERROR(cudaMemcpy(hostC.ptr, devC.ptr, sizeC, cudaMemcpyDeviceToHost));
	}

    HANDLE_ERROR(cudaFree(devA.ptr));
    HANDLE_ERROR(cudaFree(devB.ptr));
    HANDLE_ERROR(cudaFree(devC.ptr));

    HANDLE_ERROR(cudaEventRecord(stop, 0));

    return hostC;
}

__global__ void kernelSharedFast1(cudaPitchedPtr A, cudaPitchedPtr B, cudaPitchedPtr C)
{
	int y = threadIdx.y + blockIdx.y * blockDim.x;
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ Type a[32][32];
	__shared__ Type b[32][32];

	Type sum1, sum2, sum3, sum4;
	sum1 = sum2 = sum3 = sum4 = 0.;
	int num_iter = A.xsize / blockDim.x;
	for (int i = 0; i < num_iter; i++) {
		a[threadIdx.y +  0][threadIdx.x] = get_elem(A, y +  0, blockDim.x * i + threadIdx.x);
		a[threadIdx.y +  8][threadIdx.x] = get_elem(A, y +  8, blockDim.x * i + threadIdx.x);
		a[threadIdx.y + 16][threadIdx.x] = get_elem(A, y + 16, blockDim.x * i + threadIdx.x);
		a[threadIdx.y + 24][threadIdx.x] = get_elem(A, y + 24, blockDim.x * i + threadIdx.x);

		b[threadIdx.y +  0][threadIdx.x] = get_elem(B, blockDim.x * i + threadIdx.y +  0, x);
		b[threadIdx.y +  8][threadIdx.x] = get_elem(B, blockDim.x * i + threadIdx.y +  8, x);
		b[threadIdx.y + 16][threadIdx.x] = get_elem(B, blockDim.x * i + threadIdx.y + 16, x);
		b[threadIdx.y + 24][threadIdx.x] = get_elem(B, blockDim.x * i + threadIdx.y + 24, x);

		__syncthreads();
		Type b_tmp;
		for (int k = 0; k < blockDim.x; k++) {
			b_tmp = b[k][threadIdx.x];
			sum1 += a[threadIdx.y +  0][k] * b_tmp;
			sum2 += a[threadIdx.y +  8][k] * b_tmp;
			sum3 += a[threadIdx.y + 16][k] * b_tmp;
			sum4 += a[threadIdx.y + 24][k] * b_tmp;
		}
		__syncthreads();
	}

	get_addr(C, y +  0, x) = sum1;
	get_addr(C, y +  8, x) = sum2;
	get_addr(C, y + 16, x) = sum3;
	get_addr(C, y + 24, x) = sum4;
}

__global__ void kernelSharedFast2(cudaPitchedPtr A, cudaPitchedPtr B, cudaPitchedPtr C)
{
	int y = threadIdx.y + blockIdx.y * blockDim.x;
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ Type a[24][32 + 1];
	__shared__ Type b[32][32];

	Type sum1, sum2, sum3, sum4, b_tmp;
	sum1 = sum2 = sum3 = sum4 = 0.;
	int num_iter = A.xsize / blockDim.x;
	for (int i = 0; i < num_iter; i++) {
		b[threadIdx.y +  0][threadIdx.x] = get_elem(B, blockDim.x * i + threadIdx.y +  0, x);
		b[threadIdx.y +  8][threadIdx.x] = get_elem(B, blockDim.x * i + threadIdx.y +  8, x);
		b[threadIdx.y + 16][threadIdx.x] = get_elem(B, blockDim.x * i + threadIdx.y + 16, x);
		b[threadIdx.y + 24][threadIdx.x] = get_elem(B, blockDim.x * i + threadIdx.y + 24, x);

		a[threadIdx.y +  0][threadIdx.x] = get_elem(A, y +  0, blockDim.x * i + threadIdx.x);
		a[threadIdx.y +  8][threadIdx.x] = get_elem(A, y +  8, blockDim.x * i + threadIdx.x);
		a[threadIdx.y + 16][threadIdx.x] = get_elem(A, y + 16, blockDim.x * i + threadIdx.x);
		__syncthreads();
		for (int k = 0; k < 32; k++) {
			b_tmp = b[k][threadIdx.x];
			sum1 += a[threadIdx.y +  0][k] * b_tmp;
			sum2 += a[threadIdx.y +  8][k] * b_tmp;
			sum3 += a[threadIdx.y + 16][k] * b_tmp;
		}
		__syncthreads();

		a[threadIdx.y][threadIdx.x] = get_elem(A, y + 24, blockDim.x * i + threadIdx.x);
		__syncthreads();
		for (int k = 0; k < 32; k++)
			sum4 += a[threadIdx.y][k] * b[k][threadIdx.x];
		__syncthreads();
	}

	get_addr(C, y +  0, x) = sum1;
	get_addr(C, y +  8, x) = sum2;
	get_addr(C, y + 16, x) = sum3;
	get_addr(C, y + 24, x) = sum4;
}

__global__ void kernelSharedFast3(cudaPitchedPtr A, cudaPitchedPtr B, cudaPitchedPtr C)
{
	int y = threadIdx.y + blockIdx.y * blockDim.x;
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ Type a[32][33];
	__shared__ Type b[31][32];

	Type sum1, sum2, sum3, sum4, b_tmp, b_last;
	sum1 = sum2 = sum3 = sum4 = 0.;
	int num_iter = A.xsize / blockDim.x;
	for (int i = 0; i < num_iter; i++) {
		b[threadIdx.y +  0][threadIdx.x] = get_elem(B, blockDim.x * i + threadIdx.y + 24, x);
		__syncthreads();
		b_last = b[7][threadIdx.x];
		if (threadIdx.y != 7)
			b[threadIdx.y + 24][threadIdx.x] = b[threadIdx.y][threadIdx.x];
		a[threadIdx.y +  0][threadIdx.x] = get_elem(A, y +  0, blockDim.x * i + threadIdx.x);
		a[threadIdx.y +  8][threadIdx.x] = get_elem(A, y +  8, blockDim.x * i + threadIdx.x);
		a[threadIdx.y + 16][threadIdx.x] = get_elem(A, y + 16, blockDim.x * i + threadIdx.x);
		a[threadIdx.y + 24][threadIdx.x] = get_elem(A, y + 24, blockDim.x * i + threadIdx.x);

		__syncthreads(); // for b_last
		b[threadIdx.y +  0][threadIdx.x] = get_elem(B, blockDim.x * i + threadIdx.y +  0, x);
		b[threadIdx.y +  8][threadIdx.x] = get_elem(B, blockDim.x * i + threadIdx.y +  8, x);
		b[threadIdx.y + 16][threadIdx.x] = get_elem(B, blockDim.x * i + threadIdx.y + 16, x);

		__syncthreads();
		for (int k = 0; k < 31; k++) {
			b_tmp = b[k][threadIdx.x];
			sum1 += a[threadIdx.y +  0][k] * b_tmp;
			sum2 += a[threadIdx.y +  8][k] * b_tmp;
			sum3 += a[threadIdx.y + 16][k] * b_tmp;
			sum4 += a[threadIdx.y + 24][k] * b_tmp;
		}
		sum1 += a[threadIdx.y +  0][31] * b_last;
		sum2 += a[threadIdx.y +  8][31] * b_last;
		sum3 += a[threadIdx.y + 16][31] * b_last;
		sum4 += a[threadIdx.y + 24][31] * b_last;
		__syncthreads();
	}

	get_addr(C, y +  0, x) = sum1;
	get_addr(C, y +  8, x) = sum2;
	get_addr(C, y + 16, x) = sum3;
	get_addr(C, y + 24, x) = sum4;
}

cudaPitchedPtr Gpu::matmulFastGPU(cudaPitchedPtr hostA, cudaPitchedPtr hostB, Options opt)
{
	int blockSizeX = 32, blockSizeY = 8;
    cudaPitchedPtr hostC = makeMatrix(hostA.ysize, hostB.xsize);

	int newWidthA, newHeightA, newWidthB, newHeightB, newWidthC, newHeightC;
	newWidthA  = blockSizeX * ((hostA.xsize + blockSizeX - 1) / blockSizeX);
	newHeightA = blockSizeX * ((hostA.ysize + blockSizeX - 1) / blockSizeX);
	newWidthB  = blockSizeX * ((hostB.xsize + blockSizeX - 1) / blockSizeX);
	newHeightB = blockSizeX * ((hostB.ysize + blockSizeX - 1) / blockSizeX);
	newWidthC  = blockSizeX * ((hostC.xsize + blockSizeX - 1) / blockSizeX);
	newHeightC = blockSizeX * ((hostC.ysize + blockSizeX - 1) / blockSizeX);

    cudaPitchedPtr devA, devB, devC;
    devA = make_cudaPitchedPtr(0, 0, newWidthA, newHeightA);
    devB = make_cudaPitchedPtr(0, 0, newWidthB, newHeightB);
    devC = make_cudaPitchedPtr(0, 0, newWidthC, newHeightC);

    HANDLE_ERROR(cudaMallocPitch((void **)&devA.ptr, &devA.pitch, devA.xsize * sizeof(Type), devA.ysize));
    HANDLE_ERROR(cudaMallocPitch((void **)&devB.ptr, &devB.pitch, devB.xsize * sizeof(Type), devB.ysize));
    HANDLE_ERROR(cudaMallocPitch((void **)&devC.ptr, &devC.pitch, devC.xsize * sizeof(Type), devC.ysize));

	HANDLE_ERROR(cudaMemset(devA.ptr, 0, devA.pitch * devA.ysize));
	HANDLE_ERROR(cudaMemset(devB.ptr, 0, devB.pitch * devB.ysize));
	HANDLE_ERROR(cudaMemset(devC.ptr, 0, devC.pitch * devC.ysize));

    HANDLE_ERROR(cudaMemcpy2D(devA.ptr, devA.pitch,
							hostA.ptr, hostA.pitch,
							hostA.pitch, hostA.ysize,
							cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy2D(devB.ptr, devB.pitch,
							hostB.ptr, hostB.pitch,
							hostB.pitch, hostB.ysize,
							cudaMemcpyHostToDevice));

    dim3 block(blockSizeX, blockSizeY);
    dim3 grid(devC.xsize / blockSizeX, devC.ysize / blockSizeX);

	if (opt.getFastLevel() == 1)
		kernelSharedFast1<<< grid, block >>>(devA, devB, devC);
	else if (opt.getFastLevel() == 2)
		kernelSharedFast2<<< grid, block >>>(devA, devB, devC);
	else if (opt.getFastLevel() == 3)
		kernelSharedFast3<<< grid, block >>>(devA, devB, devC);

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
