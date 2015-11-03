#ifndef AXPYCPU_H
#define AXPYCPU_H

class AXPYCPU
{
public:
    AXPYCPU() {}
    AXPYCPU(int n_block_, int n_thread_, float a_);
    ~AXPYCPU() {}

    int n_block, n_thread, n;
    float a;
    float *x;
    float *y;

    void compute(float* x_, float* y_, float* z_);
};

#endif // AXPYCPU_H
