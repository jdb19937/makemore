#ifndef __MAKEMORE_RANDOM_HH__
#define __MAKEMORE_RANDOM_HH__

extern void seedrand();
extern void seedrand(unsigned int n);

extern double randgauss();
extern double randrange(double a, double b);
extern int randrange(int a, int b);
inline double randunit() {
  return randrange(0.0, 1.0);
}
inline unsigned int randint() {
  return randrange(0, 1 << 31);
}

#endif
