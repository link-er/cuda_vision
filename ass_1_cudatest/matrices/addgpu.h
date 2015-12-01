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
    float *a1;
    float *b1;
    float *c1;
    float *e1;
    float *d1;

    void compute(float** a_, float** b_, float** c_, float* e_, float* d_);
};

#endif // ADDGPU_H
