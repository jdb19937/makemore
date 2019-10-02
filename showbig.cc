#include <stdio.h>
#include <dirent.h>

#include "encgen.hh"
#include "strutils.hh"
#include "ppm.hh"
#include "urb.hh"
#include "zone.hh"
#include "cudamem.hh"
#include "numutils.hh"
#include "imgutils.hh"
#include "cholo.hh"
#include "closest.hh"
#include "partrait.hh"
#include "catalog.hh"
#include "impdis.hh"
#include "superenc.hh"
#include "supergen.hh"

using namespace makemore;
using namespace std;

int main(int argc, char **argv) {
  unsigned long seed = 0;
  if (argc > 1)
    seed = strtoul(argv[1], NULL, 0);

  double dev = 1.0;
  if (argc > 2)
    dev = strtod(argv[2], NULL);

  double pi = 0.001;
  double nu = 0.001;

  unsigned int mbn = 1;
  Superenc enc("nenc.proj", 1);
  Supergen gen("ngen.proj", 1);
  Styler sty("nsty.proj");

#if 0
Impdis impdis("id.proj", mbn);
#endif

  unsigned int w = 256;
  unsigned int h = 256;

  unsigned int dim = 2;
  PPM out(w * dim, h * dim);
  seedrand(seed);

  uint8_t *rgb = new uint8_t[w * h * 3];

  double *tmpd = new double[10 << 20];

  // assert(egd.ctrlay->n == 512);
double *ctr = new double[enc.enc->outn];

  for (unsigned int y = 0; y < dim; ++y) {
  for (unsigned int x = 0; x < dim; ++x) {
    Partrait *npar = new Partrait(256, 256);
    npar->fill_black();
    npar->set_pose(Pose::STANDARD);
    Partrait showpar(256, 256);
    Parson prs;

    Pose pose = Pose::STANDARD;
    pose.angle += randrange(-0.05, 0.05);
    pose.stretch += randrange(-0.05, 0.05);
    pose.skew += randrange(-0.05, 0.05);

memset(prs.tags, 0, sizeof(prs.tags));
//prs.add_tag("male");

    for (unsigned int j = 0; j < Parson::ncontrols; ++j)
      prs.controls[j] = randgauss() * dev;
    prs.angle = randrange(-0.05, 0.05);
    prs.stretch = 1.0 + randrange(-0.05, 0.05);
    prs.skew = randrange(-0.05, 0.05);

    sty.generate(prs, ctr);
    gen.generate(ctr, npar, NULL);

//for (unsigned int i = 0; i < 256*256; ++i)
//npar->alpha[i] = 255;

Partrait spar(256, 256);
spar.fill_gray();
spar.set_pose(npar->get_pose());
npar->warpover(&spar);

    out.paste(spar.rgb, w, h, x * w, y * h);


  }
  }

  out.write(stdout);
}
