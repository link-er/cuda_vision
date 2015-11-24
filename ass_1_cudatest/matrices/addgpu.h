#ifndef ADDGPU_H
#define ADDGPU_H

#include <cuda.h>
#include <cuda_runtime.h>

class ADDGPU
{
public:
    ADDGPU(int pass_n, int pass_k, int pass_m);
    ~ADDGPU() {}

    int n, m, k;
    float *a;
    float *b;
    float *c;
    float *e;
    float *d;

    void compute(float** a_, float** b_, float** c_, float* e_, float* d_);
};

#endif // ADDGPU_H
