#ifndef __MAKEMORE_ENCGENDIS_HH__
#define __MAKEMORE_ENCGENDIS_HH__ 1

#include "layout.hh"
#include "topology.hh"
#include "multitron.hh"
#include "vocab.hh"
#include "project.hh"
#include "script.hh"
#include "convo.hh"
#include "parson.hh"
#include "mapfile.hh"
#include "cholo.hh"

#include <vector>
#include <string>
#include <map>

namespace makemore {

struct Encgendis : Project {
  double decay;

  Layout *tgtlay, *ctrlay, *ctxlay, *knblay;
  Layout *encinlay, *geninlay, *disinlay;
  Layout *seginlay, *segoutlay;

  Topology *enctop, *gentop, *distop, *inenctop, *ingentop;
  Topology *segtop;
  Mapfile *encmap, *genmap, *dismap, *inencmap, *ingenmap;
  Mapfile *segmap;

  Tron *enc, *gen, *dis;
  Tron *encpass, *genpass;
  Tron *encgen, *genenc;

  Tron *seg;

  Tron *inenc, *ingen;

  double *cuinencin, *cuingenin;
  double *cuencin, *cuenctgt, *cudistgt;
  double *cugenin, *cugentgt, *cudisin, *cudisfin, *cusegtgt, *cusegin;
  double *realctr, *fakectr, *morectr, *distgt, *fakectx;

  uint8_t *bctxbuf, *btgtbuf, *boutbuf, *bsepbuf, *bctrbuf;
  uint16_t *sadjbuf;
  double *ctxbuf, *ctrbuf, *knbbuf;
  double *tgtbuf, *sepbuf, *segbuf;
  double *outbuf, *adjbuf;

  double *cutgtlayx, *cutgtlayy;

  unsigned int rounds;

  Encgendis(const std::string &_dir, unsigned int _mbn);
  ~Encgendis();

  void report(const char *prog);
  void load();
  void save();

  void observe(double mu, double yo, double wu, const double *realness);
  void observepre(double mu, double yo, double wu);
  void scramble(double dev = 1.0);
  void generate();
  void segment();
  void inscramble(double *knobs, unsigned int n, Cholo *cholo);
  void ingenerate();
  void encode();
  void inencode();
  void inburn(const double *cusamp, unsigned int n, double nu, double pi);
  void burnseg(double nu, double pi);
  void inconc(const double *samp, unsigned int n, double nu, double pi);
  void burn(double nu, double pi);
  void burngen(double pi);
  void burnenc(double nu);

  void bootpre(class Cholo *, double nu);
};

}

#endif
