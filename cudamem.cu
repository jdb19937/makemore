#define __MAKEMORE_CUDAMEM_CU__ 1
#include "cudamem.hh"

#include <assert.h>
#include <stdio.h>
#include <stdint.h>

namespace makemore {

void encudev(const void *a, unsigned int n, void *da) {
  cudaMemcpy(da, a, n, cudaMemcpyHostToDevice);
}

void decudev(const void *da, unsigned int n, void *a) {
  cudaMemcpy(a, da, n, cudaMemcpyDeviceToHost);
}

void cumakev(void **dp, unsigned int n) {
  void *d = NULL;
//fprintf(stderr, "makev %u -> ", n);
  assert(0 == cudaMalloc((void **)&d, n));
//fprintf(stderr, "%llu\n", d);
  assert(d != NULL);
  assert(dp != NULL);
  *dp = d;
}

void cucarvev(void **dp, unsigned int n, void **base, void *top) {
  assert(*(uint8_t **)base + n <= (uint8_t *)top);
  *(uint8_t **)dp = *(uint8_t **)base;
  *(uint8_t **)base += n;
}

void cufreev(void *x) {
//fprintf(stderr, "cufree %llu\n", x);
  cudaFree(x);
}

void cuzerov(void *x, unsigned int n) {
  cudaMemset((void *)x, 0, n);
}

void cucopyv(const void *x, unsigned int n, void *y) {
  cudaMemcpy(y, x, n, cudaMemcpyDeviceToDevice);
}

__global__ void gpu_muld(const double *a, const double b, unsigned int n, double *c) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = (a[i] * b);
}

void cumuld(const double *a, const double b, unsigned int n, double *c) {
  int bs = 128;
  int gs = ((n + bs - 1) / bs);
  gpu_muld<<<gs, bs>>>(a, b, n, c);
}

__global__ void gpu_mulvec(const double *a, const double *b, unsigned int n, double *c) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = (a[i] * b[i]);
}

void cumulvec(const double *a, const double *b, unsigned int n, double *c) {
  int bs = 128;
  int gs = ((n + bs - 1) / bs);
  gpu_mulvec<<<gs, bs>>>(a, b, n, c);
}

__global__ void gpu_bcevec(const double *a, const double *b, unsigned int n, double *c) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = (a[i]/(b[i] + 1e-9)) - (1-a[i])/(1 - b[i] + 1e-9);
}

void cubcevec(const double *a, const double *b, unsigned int n, double *c) {
  int bs = 128;
  int gs = ((n + bs - 1) / bs);
  gpu_bcevec<<<gs, bs>>>(a, b, n, c);
}


__global__ void gpu_divsqrtvec(const double *a, const double *b, unsigned int n, double *c) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = (a[i] / sqrt(b[i]));
}

void cudivsqrtvec(const double *a, const double *b, unsigned int n, double *c) {
  int bs = 128;
  int gs = ((n + bs - 1) / bs);
  gpu_divsqrtvec<<<gs, bs>>>(a, b, n, c);
}
__global__ void gpu_mulsqrtvec(const double *a, const double *b, unsigned int n, double *c) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = (a[i] * sqrt(b[i]));
}

void cumulsqrtvec(const double *a, const double *b, unsigned int n, double *c) {
  int bs = 128;
  int gs = ((n + bs - 1) / bs);
  gpu_mulsqrtvec<<<gs, bs>>>(a, b, n, c);
}

__global__ void gpu_cuaddvec(const double *a, const double *b, unsigned int n, double *c) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = a[i] + b[i];
}

void cuaddvec(const double *a, const double *b, unsigned int n, double *c) {
  int bs = 128;
  int gs = ((n + bs - 1) / bs);
  gpu_cuaddvec<<<gs, bs>>>(a, b, n, c);
}

__global__ void gpu_cusubvec(const double *a, const double *b, unsigned int n, double *c) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = a[i] - b[i];
}

void cusubvec(const double *a, const double *b, unsigned int n, double *c) {
  int bs = 128;
  int gs = ((n + bs - 1) / bs);
  gpu_cusubvec<<<gs, bs>>>(a, b, n, c);
}

__global__ void gpu_cumulvec(const double *a, double m, int n, double *b) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    b[i] = a[i] * m;
}

void cumulvec(const double *a, double m, unsigned int n, double *b) {
  int bs = 128;
  int gs = ((n + bs - 1) / bs);
  gpu_cumulvec<<<gs, bs>>>(a, m, n, b);
}

__global__ void gpu_cuexpand(double *a, int n, double m) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    a[i] = (a[i] - 0.5) * m + 0.5;
}

void cuexpand(double *a, unsigned int n, double m) {
  int bs = 128;
  int gs = ((n + bs - 1) / bs);
  gpu_cuexpand<<<gs, bs>>>(a, n, m);
}

__global__ void gpu_cufocus(double *a, const double *xp, const double *yp, unsigned int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n)
    return;

  double x = xp[i];
  double y = yp[i];

  double d2 = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5);
  double f = 1.0 - 2.0 * d2;
  if (f < 0.0)
    f = 0.0;
  if (f > 1.0)
    f = 1.0;


// In[2]:= f[x_,y_] := 1 - 2 * ((x-1/2)^2 + (y-1/2)^2) 
// In[6]:= Integrate[f[x,y], {x,0,1},{y,0,1}]
//
//         2
// Out[6]= -
//         3
//  a[i] *= f * 1.5;

// double g = 1.5;


// In[26]:= Integrate[f[x,y]^2, {x,0,1},{y,0,1}]
// 
//          22
// Out[26]= --
//          45

// double g = 45.0 / 22.0;

double g = 2.0;
if (f < 0.1)
f = 0.1;

  a[i] *= f * f * g;


}

void cufocus(double *a, const double *x, const double *y, unsigned int n) {
  int bs = 128;
  int gs = ((n + bs - 1) / bs);
  gpu_cufocus<<<gs, bs>>>(a, x, y, n);
}


__global__ void gpu_cutwiddle3(const double *z, unsigned int w, unsigned int h, double *lo, double *hi) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  // assert(w % 2 == 0);
  // assert(h % 2 == 0);

  unsigned int nw = w / 2;
  // unsigned int nh = h / 2;

  unsigned int x = i % nw;
  unsigned int y = i / nw;
  y *= 2;
  x *= 2;

  unsigned int ilo = i * 3, ihi = i * 9;
  unsigned int w3 = w * 3;

  if (hi) {

//  for (unsigned int y = 0; y < h; y += 2) {
//    for (unsigned int x = 0; x < w; x += 2) {
      for (unsigned int c = 0; c < 3; ++c) {
        unsigned int p = y * w3 + x * 3 + c;

        double m = (z[p] + z[p + 3] + z[p + w3] + z[p + w3 + 3]) / 4.0;
        double l = (z[p] + z[p + w3]) / 2.0 - m;
        double t = (z[p] + z[p + 3]) / 2.0 - m;
        double s = (z[p] + z[p + w3 + 3]) / 2.0 - m;

        lo[ilo++] = m;
        hi[ihi++] = 0.5 + l / 2.0;
        hi[ihi++] = 0.5 + t / 2.0;
        hi[ihi++] = 0.5 + s / 2.0;
      }
//    }
//  }

  } else {

//  for (unsigned int y = 0; y < h; y += 2) {
//    for (unsigned int x = 0; x < w; x += 2) {
      for (unsigned int c = 0; c < 3; ++c) {
        unsigned int p = y * w3 + x * 3 + c;

        double m = (z[p] + z[p + 3] + z[p + w3] + z[p + w3 + 3]) / 4.0;

        lo[ilo++] = m;
      }
//    }
//  }

  }
}

void cutwiddle3(const double *z, unsigned int w, unsigned int h, double *lo, double *hi) {
  assert(w % 2 == 0);
  assert(h % 2 == 0);

  unsigned int n = (w * h / 4);

  int bs = 128;
  int gs = ((n + bs - 1) / bs);

  gpu_cutwiddle3<<<gs, bs>>>(z, w, h, lo, hi);
}





__global__ void gpu_cucutpaste(
  const double *a, const double *b,
  unsigned int rows, unsigned int acols, unsigned int bcols, unsigned int ccols,
  double *c
) {
  unsigned int ci = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int cn = ccols * rows;
  if (ci >= cn)
    return;
  unsigned int ccol = ci % ccols;
  unsigned int row = ci / ccols;

  unsigned int k = ccols - bcols;
  if (ccol < k) {
    unsigned int acol = ccol;
    unsigned int ai = acols * row + acol;
    c[ci] = a[ai];
  } else {
    unsigned int bcol = ccol - k;
    unsigned int bi = bcols * row + bcol;
    c[ci] = b[bi];
  }
}

void cucutpaste(
  const double *a, const double *b,
  unsigned int rows, unsigned int acols, unsigned int bcols, unsigned int ccols,
  double *c
) {
  assert(ccols >= bcols);
  unsigned int cn = ccols * rows;

  int bs = 128;
  int gs = ((cn + bs - 1) / bs);
  gpu_cucutpaste<<<gs, bs>>>(a, b, rows, acols, bcols, ccols, c);
}


__global__ void gpu_cucutadd(
  const double *a,
  unsigned int rows, unsigned int acols, unsigned int bcols,
  double *b
) {
  unsigned int bi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int bn = bcols * rows;
  if (bi >= bn)
    return;
  unsigned int bcol = bi % bcols;
  unsigned int row = bi / bcols;

  unsigned int acol = bcol + (acols - bcols);
  unsigned int ai = acols * row + acol;

  b[bi] += a[ai];
}

void cucutadd(
  const double *a,
  unsigned int rows, unsigned int acols, unsigned int bcols,
  double *b
) {
  assert(acols >= bcols);

  unsigned int bn = bcols * rows;
  int bs = 128;
  int gs = ((bn + bs - 1) / bs);
  gpu_cucutadd<<<gs, bs>>>(a, rows, acols, bcols, b);
}


__global__ void gpu_sumsq(
  const double *a, unsigned int n, double *sumsqp
) {
  unsigned int si = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int i0 = si * 128;
  if (i0 >= n)
    return;
  unsigned int i1 = (i0 + 128 >= n) ? n : (i0 + 128);
  
  double s = 0;
  for (unsigned int i = i0; i < i1; ++i)
    s += a[i] * a[i];
  sumsqp[si] = s;
}

double cusumsq(
  const double *a, unsigned int n
) {
  if (n == 0)
    return 0;

  double *sumsqp = NULL;
  unsigned int sumsqn = ((n + 127) / 128);
  cumake(&sumsqp, sumsqn);

  int bs = 128;
  int gs = (sumsqn + bs - 1) / bs;
  gpu_sumsq<<<gs, bs>>>(a, n, sumsqp);

  double *sumsqv = new double[sumsqn];
  decude(sumsqp, sumsqn, sumsqv);

  double s = 0;
  for (int i = 0; i < sumsqn; ++i)
    s += sumsqv[i];

  cufree(sumsqp);
  delete[] sumsqv;

  return s;
}


__global__ void gpu_maxabs(
  const double *a, unsigned int n, double *maxp
) {
  unsigned int si = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int i0 = si * 128;
  if (i0 >= n)
    return;
  unsigned int i1 = (i0 + 128 >= n) ? n : (i0 + 128);

  unsigned int i = i0;
  double s = fabs(a[i]);
  ++i;
  
  for (; i < i1; ++i) {
    double aa = fabs(a[i]);
    if (aa > s)
      s = aa;
  }

  maxp[si] = s;
}

double cumaxabs(
  const double *a, unsigned int n
) {
  if (n == 0)
    return 0;

  double *maxp = NULL;
  unsigned int maxn = ((n + 127) / 128);
  cumake(&maxp, maxn);

  int bs = 128;
  int gs = (maxn + bs - 1) / bs;
  gpu_maxabs<<<gs, bs>>>(a, n, maxp);

  double *maxv = new double[maxn];
  decude(maxp, maxn, maxv);

  double s = maxv[0];
  for (int i = 1; i < maxn; ++i)
    if (maxv[i] > s)
      s = maxv[i];

  cufree(maxp);
  delete[] maxv;

  return s;
}

__global__ void gpu_meandevcols(
  const double *a, unsigned int rows, unsigned int cols, double *m, double *d
) {
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols)
    return;

  double mean = 0;
  for (unsigned int row = 0; row < rows; ++rows)
    mean += a[row * cols + col]; 
  mean /= (double)rows;
  m[col] = mean;

  double dev = 0;
  for (unsigned int row = 0; row < rows; ++rows) {
    double z = a[row * cols + col] - mean;
    dev += z * z;
  }
  dev /= (double)rows;
  dev = sqrt(dev);
  d[col] = dev;
}

void cumeandevcols(
  const double *a, unsigned int rows, unsigned int cols, double *m, double *d
) {
  assert(rows > 0);
  assert(cols > 0);

  int bs = 128;
  int gs = ((cols + bs - 1) / bs);
  gpu_meandevcols<<<gs, bs>>>(a, rows, cols, m, d);
}


__global__ void gpu_normalize(
  const double *a, unsigned int rows, unsigned int cols,
  const double *m, const double *d, double *b
) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * cols)
    return;
  unsigned int col = i % cols;

  double x = a[i];
  x -= m[col];
  x /= d[col];
  b[i] = x;
}

void cunormalize(
  const double *a, unsigned int rows, unsigned int cols,
  const double *m, const double *d, double *b
) {
  assert(rows > 0);
  assert(cols > 0);
  unsigned int n = rows * cols;

  int bs = 128;
  int gs = ((n + bs - 1) / bs);
  gpu_normalize<<<gs, bs>>>(a, rows, cols, m, d, b);
}

__global__ void gpu_upmean(
  const double *m,
  unsigned int rows, unsigned int cols,
  unsigned int n,
  double *c
) {
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols)
    return;

  double w = c[col] * n;
  for (unsigned int i = 0; i < rows; ++i) {
    double z = m[i * cols + col];
    w += z;
  }

  w /= ((double)n + (double)rows);
  c[col] = w ;
}

void cuupmean(
  const double *m,
  unsigned int rows, unsigned int cols,
  unsigned int n,
  double *c
) {
  unsigned int cn = cols;
  int bs = 256;
  int gs = ((cn + bs - 1) / bs);
  gpu_upmean<<<gs, bs>>>(m, rows, cols, n, c);
}

__global__ void gpu_upmeanexp(
  const double *m,
  unsigned int cols,
  double decay,
  double *c
) {
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols)
    return;
  c[col] = c[col] * (1.0 - decay) + m[col] * decay;
}

void cuupmeanexp(
  const double *m,
  unsigned int cols,
  double decay,
  double *c
) {
  unsigned int cn = cols;
  int bs = 256;
  int gs = ((cn + bs - 1) / bs);
  gpu_upmeanexp<<<gs, bs>>>(m, cols, decay, c);
}

__global__ void gpu_upvarexp(
  const double *m,
  const double *mean,
  unsigned int cols,
  double decay,
  double *c
) {
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols)
    return;

  double q = m[col] - mean[col];
  double z = q * q;

  c[col] = c[col] * (1.0 - decay) + z * decay;
}

void cuupvarexp(
  const double *m,
  const double *mean,
  unsigned int cols,
  double decay,
  double *c
) {
  unsigned int cn = cols;
  int bs = 256;
  int gs = ((cn + bs - 1) / bs);
  gpu_upvarexp<<<gs, bs>>>(m, mean, cols, decay, c);
}


__global__ void gpu_upvariance(
  const double *m,
  unsigned int rows, unsigned int cols,
  unsigned int n,
  double *c
) {
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols)
    return;

  double w = c[col] * (double)n;
  for (unsigned int i = 0; i < rows; ++i) {
    double z = m[i * cols + col] * m[i * cols + col];
    w += z;
  }
  w /= (double)(n + rows);

  c[col] = w;
}

void cuupvariance(
  const double *m,
  unsigned int rows, unsigned int cols,
  unsigned int n,
  double *c
) {
  unsigned int cn = cols;
  int bs = 256;
  int gs = ((cn + bs - 1) / bs);
  gpu_upvariance<<<gs, bs>>>(m, rows, cols, n, c);
}


__global__ void gpu_upcovariance(
  const double *m,
  unsigned int rows, unsigned int cols,
  unsigned int n,
  unsigned int steps,
  double *c
) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  i *= steps;

  unsigned int i1 = i + steps;
  if (i1 > cols * cols)
    i1 = cols * cols;

  while (i < i1) {
    unsigned int acol = i / cols;
    unsigned int bcol = i % cols;

    double w = c[i] * (double)n;
    for (unsigned int row = 0; row < rows; ++row) {
      double z = m[row * cols + acol] * m[row * cols + bcol];
      w += z;
    }

    w /= ((double)(n + rows));
    c[i] = w;
    ++i;
  }
}

void cuupcovariance(
  const double *m,
  unsigned int rows, unsigned int cols,
  unsigned int n, unsigned int steps,
  double *c
) {
  unsigned int cn = (cols * cols + steps - 1) / steps;
  int bs = 32;
  int gs = ((cn + bs - 1) / bs);
  gpu_upcovariance<<<gs, bs>>>(m, rows, cols, n, steps, c);
}


__global__ void gpu_covariance(
  const double *m,
  unsigned int rows, unsigned int cols,
  double *c
) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cols * cols)
    return;
  unsigned int acol = i / cols;
  unsigned int bcol = i % cols;

  double z = 0;
  for (unsigned int row = 0; row < rows; ++row)
    z += m[row * cols + acol] * m[row * cols + bcol];
  z /= (double)rows;

  c[i] = z;
}

void cucovariance(
  const double *m,
  unsigned int rows, unsigned int cols,
  double *c
) {
  unsigned int cn = cols * cols;
  int bs = 256;
  int gs = ((cn + bs - 1) / bs);
  gpu_covariance<<<gs, bs>>>(m, rows, cols, c);
}

__global__ void gpu_chol(double *U, unsigned int num_rows, int ops_per_thread)
{
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  
  unsigned int i, j, k;
  
  for (k = 0; k < num_rows; k++) {
    if (tx == 0) {
      U[k * num_rows + k] = sqrt(U[k * num_rows + k]);
    
      for (j = (k + 1); j < num_rows; j++) {
        U[k * num_rows + j] /= U[k * num_rows + k];
//        if (isnan(U[k * num_rows + j]))
//          U[k * num_rows + j] = 0;
      }
    }
    
    __syncthreads();

    int istart = tx * ops_per_thread + k + 1;
    int iend = (istart + ops_per_thread);
    if (iend > num_rows)
      iend = num_rows;
    
    for (i = istart; i < iend; i++) {
      for (j = i; j < num_rows; j++) {
        U[i * num_rows + j] -= U[k * num_rows + i] * U[k * num_rows + j];
      }
    }
  
    __syncthreads();
  }

  __syncthreads();
  
  
  unsigned int istart = tx * ops_per_thread;
  unsigned int iend = istart + ops_per_thread;
  if (iend > num_rows)
    iend = num_rows;
  
  for (i = istart; i < iend; i++) {
    for (j = 0; j < i; j++) {
      U[i * num_rows + j] = 0.0;
    }
//    for (j = i; j < num_rows; j++) {
//      if (isnan(U[i * num_rows + j]))
//        U[i * num_rows + j] = 0.0;
//    }
  }
}

void cuchol(
  double *m, unsigned int rows
) {
  unsigned int ops = 1;
  unsigned int rn = (rows + ops - 1) / ops;
  int bs = 128;
  int gs = ((rn + bs - 1) / bs);
  gpu_chol<<<gs, bs>>>(m, rows, ops);
}

__global__ void gpu_matxvec(const double *m, const double *x, unsigned int w, unsigned int h, double *y) {
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= w)
    return;

  double z = 0;
  for (unsigned int row = 0; row < h; ++row)
    z += m[row * w + col] * x[row];
  y[col] = z;
}

void cumatxvec(
  const double *m, const double *x, unsigned int w, unsigned int h, double *y
) {
  int bs = 256;
  int gs = ((w + bs - 1) / bs);
  gpu_matxvec<<<gs, bs>>>(m, x, w, h, y);
}

__global__ void gpu_tmatxvec(const double *m, const double *x, unsigned int w, unsigned int h, double *y) {
  unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= h)
    return;

  double z = 0;
  for (unsigned int col = 0; col < w; ++col)
    z += m[row * w + col] * x[col];

  y[row] = z;
}

void cutmatxvec(
  const double *m, const double *x, unsigned int w, unsigned int h, double *y
) {
  int bs = 256;
  int gs = ((h + bs - 1) / bs);
  gpu_tmatxvec<<<gs, bs>>>(m, x, w, h, y);
}

__global__ void gpu_matxpose(const double *m, unsigned int w, unsigned int h, double *u) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= w * h)
    return;
  unsigned int row = i / w;
  unsigned int col = i % w;

  u[col * h + row] = m[i];
}
  
void cumatxpose(
  const double *m, unsigned int w, unsigned int h, double *y
) {
  int bs = 256;
  int gs = ((w * h + bs - 1) / bs);
  gpu_matxpose<<<gs, bs>>>(m, w, h, y);
}


  

};
