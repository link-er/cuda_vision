#include "multiplier.h"
#include <iostream>

__global__ void multiplyby2(int n, float *a)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    a[i*n + j] = 2 * a[i*n + j];
}

MULTIPLIER::MULTIPLIER(int rows, int columns)
{
    m = rows;
    n = columns;
    cudaMalloc((void **) &a, m*n*sizeof(float));
}

void MULTIPLIER::compute(float** a_)
{
    float* temp = new float[m*n];
    for(int i=0; i<m; i++)
        for(int j=0; j<n; j++)
            temp[i*n + j] = a_[i][j];

    cudaMemcpy(a, temp, m*n*sizeof(float), cudaMemcpyHostToDevice);

    multiplyby2<<<m,n>>>(n, a);

    cudaMemcpy(temp, a, m*n*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0; i<m; i++)
        for(int j=0; j<n; j++)
            a_[i][j] = temp[i*n + j];

    cudaFree(a);
    delete [] temp;
}

