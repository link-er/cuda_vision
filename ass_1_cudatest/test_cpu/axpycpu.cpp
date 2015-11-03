#include "axpycpu.h"

AXPYCPU::AXPYCPU(int n_block_, int n_thread_, float a_)
    :n_block(n_block_), n_thread(n_thread_), a(a_)
{
    n = n_block * n_thread;
}

void AXPYCPU::compute(float* x, float* y, float* z)
{
    for(int i=0;i<n;i++){
        z[i] = x[i]*a + y[i];
    }
}

