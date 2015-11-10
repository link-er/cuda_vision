#include "addgpu.h"

__global__ void add(float *a, float *b, float *c, float *e)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    y[i] = a*x[i] + y[i];
}

ADDGPU::ADDGPU(int n_block_, int n_thread_)
    :n_block(n_block_), n_thread(n_thread_)
{
    n = 200;
    k = 500;
    m = 400;
    cudaMalloc((void **) &a, n*k*sizeof(float));
    cudaMalloc((void **) &b, k*m*sizeof(float));
    cudaMalloc((void **) &c, n*m*sizeof(float));
    cudaMalloc((void **) &e, n*sizeof(float));
}

void ADDGPU::compute(float* a_, float* b_, float* c_, float* e_, float* d_)
{
    cudaMemcpy(x, x_, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y, y_, n*sizeof(float), cudaMemcpyHostToDevice);

    axpy<<<n_block,n_thread>>>(a,x,y);

    cudaMemcpy(z_, y, n*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(x);
    cudaFree(y);
}

