#include "addgpu.h"

__global__ void add(float **a, float **b, float **c, float *e, float *d, int n, int k)
{
    // finding index for the element currently calculated
    int i = threadIdx.x;
    float temp;
    d[i] = 0;
    for(int j=0;j<n;j++){
        temp = 0;
        for(int l=0;l<k;l++){
            temp += a[j][l] * b[l][i];
        }
        d[i] += c[j][i] + e[j];
    }
}

ADDGPU::ADDGPU()
{
    n = 200;
    k = 500;
    m = 400;
    // allocate space for our variables
    cudaMalloc((void **) &a, n*k*sizeof(float));
    cudaMalloc((void **) &b, k*m*sizeof(float));
    cudaMalloc((void **) &c, n*m*sizeof(float));
    cudaMalloc((void **) &e, n*sizeof(float));
    cudaMalloc((void **) &d, m*sizeof(float));
}

void ADDGPU::compute(float** a_, float** b_, float** c_, float* e_, float* d_)
{
    // copy from host to device, to allocated memory
    cudaMemcpy(a, a_, n*k*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b, b_, k*m*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c, c_, n*m*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(e, e_, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d, d_, m*sizeof(float), cudaMemcpyHostToDevice);

    // call with specifing number of blocks and number of threads
    add<<<1,m>>>(a,b,c,e,d, n, k);

    // copy the result back to host
    cudaMemcpy(d_, d, m*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(e);
    cudaFree(d);
}

