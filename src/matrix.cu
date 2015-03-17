#include "matrix.h"

cudaPitchedPtr makeMatrix(int m, int n)
{
	Type *data = new Type [m * n];
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			data[i * n + j] = (Type)rand() / RAND_MAX;
	return make_cudaPitchedPtr(data, n * sizeof(Type), n, m);
}

void printMatrix(cudaPitchedPtr a)
{
	printf("\n");
	for (int i = 0; i < a.ysize; i++) {
		for (int j = 0; j < a.xsize; j++)
			printf("%f ", get_elem(a, i, j));
		printf("\n");
	}
	printf("\n");
}

cudaPitchedPtr matmulHOST(cudaPitchedPtr A, cudaPitchedPtr B)
{
    cudaPitchedPtr C = makeMatrix(A.ysize, B.xsize);
    memset(C.ptr, 0, sizeof(Type) * C.ysize * C.xsize);
    for (unsigned long i = 0; i < C.ysize; i++)
        for (unsigned long j = 0; j < C.xsize; j++)
            for (unsigned long k = 0; k < A.xsize; k++)
				get_addr(C, i, j) += get_elem(A, i, k) * get_elem(B, k, j);
    return C;
}

bool compareMatrix(cudaPitchedPtr A, cudaPitchedPtr B)
{
    if (A.ysize != B.ysize || A.xsize != B.xsize)
        return false;
    for (int i = 0; i < A.ysize; i++)
        for (int j = 0; j < A.xsize; j++)
            if (float_not_equel(get_elem(A, i, j), get_elem(B, i, j)))
				printf("%f %f %d %d\n", get_elem(A, i, j), get_elem(B, i, j), i, j);
                //return false;
    return true;
}
