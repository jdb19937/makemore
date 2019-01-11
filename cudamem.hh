#ifndef __MAKEMORE_CUDAMEM_HH__
#define __MAKEMORE_CUDAMEM_HH__ 1

extern void decudev(const void *da, unsigned int n, void *a);
template <class T> inline void decude(const T *da, unsigned int n, T *a)
  { decudev(da, n * sizeof(T), a); }

extern void encudev(const void *a, unsigned int n, void *da);
template <class T> inline void encude(const T *a, unsigned int n, T *da)
  { encudev(a, n * sizeof(T), da); }

extern void cumakev(void **, unsigned int n = 1);
template <class T> inline void cumake(T **x, unsigned int n = 1)
  { cumakev((void **)x, n * sizeof(T)); }
template <class T> T *cunew(unsigned int n = 1)
  { T *x; cumake(&x, n); return x; }

extern void cufreev(void *);
template <class T> void cufree(T *x)
  { cufreev(x); }

extern void cuzerov(void *, unsigned int);
template <class T> inline void cuzero(T *x, unsigned int n)
  { cuzerov(x, n * sizeof(T)); }
  
extern void cucopyv(const void *, unsigned int, void *);
template <class T> inline void cucopy(const T *da, unsigned int n, T *db)
  { cucopy(da, n * sizeof(T), db); }

extern void cuaddvec(const double *a, const double *b, unsigned int n, double *c);
extern void cusubvec(const double *a, const double *b, unsigned int n, double *c);
extern void cucutpaste(
  const double *a, const double *b,
  unsigned int rows, unsigned int acols, unsigned int bcols, unsigned int ccols,
  double *c
);
extern void cucutadd(
  const double *a, unsigned int rows, unsigned int acols,
  unsigned int bcols, double *b
);

extern double cusumsq(
  const double *a, unsigned int n
);

extern double cumaxabs(
  const double *a, unsigned int n
);

extern void cuexpand(double *a, unsigned int n, double m);

#endif
