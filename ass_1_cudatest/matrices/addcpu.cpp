#include "addcpu.h"

ADDCPU::ADDCPU(int pass_n, int pass_k, int pass_m)
{
    n = pass_n;
    k = pass_k;
    m = pass_m;
}

void ADDCPU::compute(float** a, float** b, float** c, float* e, float* d)
{
    float temp;
    for(int i=0;i<m;i++){
      d[i] = 0;
      for(int j=0;j<n;j++){
        temp = 0;
        for(int l=0;l<k;l++){
          temp += a[j][l]*b[l][i];
        }
        d[i] += c[j][i] + e[j];
      }
    }
}

