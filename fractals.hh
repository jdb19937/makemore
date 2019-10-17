#ifndef __MAKEMORE_FRACTALS_HH__
#define __MAKEMORE_FRACTALS_HH__ 1

#include <stdint.h>

namespace makemore {

#ifndef __MAKEMORE_FRACTALS_CC__
extern void julia(uint8_t *rgb, double x, double y);
extern void burnship(uint8_t *rgb, double ra, double rb);
extern void mandelbrot(uint8_t *rgb);
#endif

}

#endif

