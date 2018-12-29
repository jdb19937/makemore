#include "cudamem.hh"
#include <stdio.h>
#include "random.hh"

int main() {
  unsigned int n = 1000000;
  double *yy = new double[n];
  for (unsigned int i = 0; i < n; ++i)
    yy[i] = 2.0;

  double *x;
  cumake(&x, n);

  encude(yy, n, x);
  printf("%lf\n", cusumsq(x, n));
}
  
