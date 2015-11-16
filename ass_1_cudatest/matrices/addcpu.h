#ifndef ADDCPU_H
#define ADDCPU_H

class ADDCPU
{
public:
    ADDCPU();
    ~ADDCPU() {}

    int n, m, k;
    float **a;
    float **b;
    float **c;
    float *e;

    void compute(float** a_, float** b_, float** c_, float* e_, float* d_);
};

#endif // ADDCPU_H
