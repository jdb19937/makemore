#ifndef __MAKEMORE_MOB_HH__
#define __MAKEMORE_MOB_HH__ 1

#include "supergen.hh"
#include "styler.hh"
#include "partrait.hh"
#include "automasker.hh"

namespace makemore {

#ifndef __MAKEMORE_MOB_CC__
extern void make_mob(Supergen *gen, Styler *sty, Automasker *am, Partrait *prt);
#endif

}

#endif
