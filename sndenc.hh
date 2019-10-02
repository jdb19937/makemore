#ifndef __MAKEMORE_SNDENC_HH__
#define __MAKEMORE_SNDENC_HH__ 1

#include "layout.hh"
#include "multitron.hh"
#include "project.hh"
#include "parson.hh"
#include "mapfile.hh"
#include "partrait.hh"
#include "cholo.hh"
#include "styler.hh"
#include "supertron.hh"
#include "soundpic.hh"

#include <vector>
#include <string>
#include <map>

namespace makemore {

struct Sndenc : Project {
  bool ctract;

  Layout *inplay, *ctrlay;
  Layout *encinlay;

  Mapfile *encmap;

  Supertron *enc;

  double *cuencin, *cuencinp;
  double *ctrbuf, *inpbuf;

  unsigned int rounds;

  Sndenc(const std::string &_dir, unsigned int _mbn);
  ~Sndenc();

  void report(const char *prog);
  void load();
  void save();

  void encode(const Soundpic &sndpic, double *ctr);
  void burn(const class Sndgen &gen, double nu);
};

}

#endif
