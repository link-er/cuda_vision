#include "addcpu.h"

ADDCPU::ADDCPU()
{
    n = 200;
    k = 500;
    m = 400;
}

void ADDCPU::compute(float* a, float* b, float* c, float* e, float* d)
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

