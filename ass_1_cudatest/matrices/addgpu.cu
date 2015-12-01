#include "addgpu.h"

__global__ void add(float *a, float *b, float *c, float *e, float *d, int n, int k, int m)
{
    // finding index for the element currently calculated
    int i = threadIdx.x;
    float temp;
    d[i] = 0;
    for(int j=0; j<n; j++){
        temp = 0;
        for(int l=0; l<k; l++){
            temp += a[j*n + l] * b[l*k + i];
        }
        d[i] += c[j*n + i] + e[j];
    }
}

ADDGPU::ADDGPU(int pass_n, int pass_k, int pass_m)
{
    n = pass_n;
    k = pass_k;
    m = pass_m;
    // allocate space for our variables
    cudaMalloc((void **) &a1, n*k*sizeof(float));
    cudaMalloc((void **) &b1, k*m*sizeof(float));
    cudaMalloc((void **) &c1, n*m*sizeof(float));
    cudaMalloc((void **) &e1, n*sizeof(float));
    cudaMalloc((void **) &d1, m*sizeof(float));
}

void ADDGPU::compute(float** a_, float** b_, float** c_, float* e_, float* d_)
{
    // copy from host to device, to allocated memory
    float* temp_a = new float[n*k];
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < k; ++j)
            temp_a[i*n + j] = a_[i][j];
    cudaMemcpy(a1, temp_a, n*k*sizeof(float), cudaMemcpyHostToDevice);

    float* temp_b = new float[k*m];
    for(int i = 0; i < k; ++i)
        for(int j = 0; j < m; ++j)
            temp_b[i*k + j] = b_[i][j];
    cudaMemcpy(b1, temp_b, k*m*sizeof(float), cudaMemcpyHostToDevice);

    float* temp_c = new float[k*m];
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < m; ++j)
            temp_c[i*n + j] = c_[i][j];
    cudaMemcpy(c1, temp_c, n*m*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(e1, e_, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d1, d_, m*sizeof(float), cudaMemcpyHostToDevice);

    // call with specifing number of blocks and number of threads
    add<<<1,m>>>(a1, b1, c1, e1, d1, n, k, m);

    // copy the result back to host
    cudaMemcpy(d_, d1, m*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(a1);
    cudaFree(b1);
    cudaFree(c1);
    cudaFree(e1);
    cudaFree(d1);
    delete [] temp_a;
    delete [] temp_b;
    delete [] temp_c;
}

