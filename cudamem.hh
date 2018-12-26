#ifndef __CUDAMEM_HH__
#define __CUDAMEM_HH__ 1

extern void decude(const double *da, unsigned int n, double *a);
extern void encude(const double *a, unsigned int n, double *da);
extern double *cumake(unsigned int n);
extern void cufree(double *);
extern void cuzero(double *, unsigned int);
extern void cucopy(const double *, unsigned int, double *);
extern void cuaddvec(const double *a, const double *b, unsigned int n, double *c);

#endif
