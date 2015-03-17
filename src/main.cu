#include <iostream>

#include "options.h"
#include "gpu.h"
#include "matrix.h"

using namespace std;

int main(int argc, char *argv[])
{
    srand(time(NULL));
    Options opt(argc, argv);
    if (opt.errorMode()) {
        cout << opt.errorPrint() << endl;
        return 1;
    }
    if (opt.helpMode()) {
        cout << opt.helpPrint() << endl;
        return 0;
    }
    if (opt.versionMode()) {
        cout << opt.versionPrint() << endl;
        return 0;
    }
	if (opt.debugMode())
		opt.infoPrint();

    unsigned long m = opt.getM();
    unsigned long n = opt.getN();
    unsigned long p = opt.getP();
	int blockSizeX = opt.getSizeX();
	int blockSizeY = opt.getSizeY();

    cudaPitchedPtr A = makeMatrix(m, n);
	cudaPitchedPtr B = makeMatrix(n, p);
	if (opt.debugMode()) {
		printMatrix(A);
		printMatrix(B);
	}
	cudaPitchedPtr resultGpu = makeMatrix(0,0);
	cudaPitchedPtr resultHost = makeMatrix(0,0);

	Gpu gpu(2, 0);
	if (opt.debugMode()) {
		gpu.infoDevices();
		cout << "My Device" << endl;
		gpu.infoMyDevice();
	}
    gpu.startTime();
	if (opt.usePadding())
        resultGpu = gpu.matmulPaddingGPU(A, B, blockSizeX, blockSizeY);
	else if (opt.sharedMemory())
        resultGpu = gpu.matmulSharedTaskGPU(A, B, blockSizeX, blockSizeY);
	else if (opt.fastMode())
		resultGpu = gpu.matmulFastGPU(A, B, opt);
	else
        resultGpu = gpu.matmulEasyGPU(A, B, blockSizeX, blockSizeY);

    cout << m << " " << gpu.getTime() << endl;
	if (opt.debugMode())
		printMatrix(resultGpu);

    if (opt.checkResult()) {
        resultHost = matmulHOST(A, B);
		if (opt.debugMode())
			printMatrix(resultHost);
        if (compareMatrix(resultHost, resultGpu))
            cout << "GPU and HOST result is equal" << endl;
        else
            cout << "Error: GPU not equal HOST" << endl;
    }

    delete [] (Type *)resultHost.ptr;
    delete [] (Type *)resultGpu.ptr;
    delete [] (Type *)A.ptr;
    delete [] (Type *)B.ptr;

    return 0;
}
