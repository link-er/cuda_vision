#include <iostream>
#include <time.h>
#include "addgpu.h"
#include "addcpu.h"

using namespace std;

int main (int argc, char** argv)
{
    int n_block = 4000;
    int n_thread = 500;
    ADDGPU add_gpu(n_block, n_thread);
    ADDCPU add_cpu();

    int n = 200;
    int k = 500;
    int m = 400;
    float *a, *b, *c, *e, *d_gpu, *d_cpu;
    a = new float[n][k];
    b = new float[k][m];
    c = new float[n][m];
    e = new float[n];
    d_gpu = new float[m];
    d_cpu = new float[m];

    for(int i=0;i<n;i++){
        for(int j=0;j<k;j++){
            a[i][j] = (float) rand() / RAND_MAX;
        }
    }
    for(int i=0;i<k;i++){
        for(int j=0;j<m;j++){
            b[i][j] = (float) rand() / RAND_MAX;
        }
    }
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            c[i][j] = (float) rand() / RAND_MAX;
        }
    }
    for(int i=0;i<n;i++){
        e[i] = (float) rand() / RAND_MAX;
    }

    clock_t tStart = clock();
    // z = a*x+y
    add_gpu.compute(a, b, c, e, d_gpu);
    double time_taken = (double)(clock() - tStart)/CLOCKS_PER_SEC;
    cout<<"Time taken with gpu: "<<time_taken<<endl;

    tStart = clock();
    add_cpu.compute(a, b, c, e, d_cpu);
    time_taken = (double)(clock() - tStart)/CLOCKS_PER_SEC;
    cout<<"Time taken with cpu: "<<time_taken<<endl;

    bool result_validity = true;
    for(int i=0;i<m;i++)
        result_validity = result_validity && (d_gpu[i] == d_cpu[i]);
    cout<<"Result is valid: "<<result_validity<<endl;

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] e;
    delete[] d_gpu;
    delete[] d_cpu;

    return 0;
}

