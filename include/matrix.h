#ifndef __MATRIX_HEADER__
#define __MATRIX_HEADER__

#include <stdio.h>

#define get_elem(array, Row, Column) \
(((Type*)((char*)array.ptr + (Row) * array.pitch))[(Column)])

#define get_addr(array, Row, Column) \
((Type&)(((Type*)((char*)array.ptr + (Row) * array.pitch))[(Column)]))

#define float_not_equel(A, B) \
(fabs((1.0+A) - (1.0+B)) > 0.0001)

typedef float Type;

cudaPitchedPtr matmulHOST(cudaPitchedPtr A, cudaPitchedPtr B);
bool compareMatrix(cudaPitchedPtr A, cudaPitchedPtr B);

cudaPitchedPtr makeMatrix(int m, int n);
cudaPitchedPtr transposeMatrix(cudaPitchedPtr a);
void printMatrix(cudaPitchedPtr a);

#endif /* __MATRIX_HEADER__ */
