#ifndef MULTIPLIER_H
#define MULTIPLIER_H

#include <cuda.h>
#include <cuda_runtime.h>

class MULTIPLIER
{
public:
    MULTIPLIER() {}
    MULTIPLIER(int rows, int columns);
    ~MULTIPLIER() {}

    int m, n;
    float *a;

    void compute(float** a_);
};

#endif // MULTIPLIER_H
