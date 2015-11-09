#ifndef ADDGPU_H
#define ADDGPU_H

#include <cuda.h>
#include <cuda_runtime.h>

class ADDGPU
{
public:
    ADDGPU() {}
    ADDGPU(int n_block_, int n_thread_);
    ~ADDGPU() {}

    int n_block, n_thread, n, m, k;
    float *a;
    float *b;
    float *c;
    float *e;

    void compute(float* a_, float* b_, float* c_, float* e_, float* d_);
};

#endif // ADDGPU_H
