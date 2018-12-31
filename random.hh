#ifndef __MAKEMORE_RANDOM_HH__
#define __MAKEMORE_RANDOM_HH__

extern double randgauss();
extern double randrange(double a, double b);
inline double randunit() {
  return randrange(0, 1);
}

#endif
