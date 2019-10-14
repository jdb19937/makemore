#ifndef __MAKEMORE_JULIA_HH__
#define __MAKEMORE_JULIA_HH__ 1

#include <stdint.h>

namespace makemore {

#ifndef __MAKEMORE_JULIA_CC__
extern void julia(uint8_t *rgb, double x, double y);
#endif

}

#endif

