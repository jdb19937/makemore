#ifndef __MAKEMORE_TWIDDLE_HH__
#define __MAKEMORE_TWIDDLE_HH__ 1

extern void untwiddle1(const double *lo, const double *hi, unsigned int w, unsigned int h, double *z);
extern void untwiddle3(const double *lo, const double *hi, unsigned int w, unsigned int h, double *z);
extern void twiddle1(const double *z, unsigned int w, unsigned int h, double *lo, double *hi);
extern void twiddle3(const double *z, unsigned int w, unsigned int h, double *lo, double *hi);

#endif
