#ifndef __MAKEMORE_CLOSEST_HH__
#define __MAKEMORE_CLOSEST_HH__ 1

namespace makemore {

unsigned int closest(const double *x, const double *m, unsigned int k, unsigned int n, unsigned int *used = NULL, unsigned int max_used = 1);
unsigned int maxdot(const double *x, const double *m, unsigned int k, unsigned int n);

};

#endif
