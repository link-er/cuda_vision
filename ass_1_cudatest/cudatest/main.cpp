#include <iostream>
#include <time.h>
#include "axpygpu.h"
#include "axpycpu.h"

using namespace std;

int main (int argc, char** argv)
{
    int n_block = 4000;
    int n_thread = 500;
    int a = 2.0;
    AXPYGPU axpy_gpu(n_block, n_thread, a);
    AXPYCPU axpy_cpu(n_block, n_thread, a);

    int n = n_block*n_thread;
    float *x, *y, *z_gpu, *z_cpu;
    x = new float[n];
    y = new float[n];
    z_gpu = new float[n];
    z_cpu = new float[n];

    for(int i=0;i<n;i++){
        x[i] = (float) rand() / RAND_MAX;
        y[i] = (float) rand() / RAND_MAX;
    }

    clock_t tStart = clock();
    // z = a*x+y
    axpy_gpu.compute(x, y, z_gpu);
    double time_taken = (double)(clock() - tStart)/CLOCKS_PER_SEC;
    cout<<"Time taken with gpu: "<<time_taken<<endl;

    tStart = clock();
    axpy_cpu.compute(x, y, z_cpu);
    time_taken = (double)(clock() - tStart)/CLOCKS_PER_SEC;
    cout<<"Time taken with cpu: "<<time_taken<<endl;

    bool result_validity = true;
    for(int i=0;i<n;i++)
        result_validity = result_validity && (z_gpu[i] == z_cpu[i]);
    cout<<"Result is valid: "<<result_validity<<endl;

    delete[] x;
    delete[] y;
    delete[] z_gpu;
    delete[] z_cpu;

    return 0;
}

