#include "project.hh"
#include "topology.hh"
#include "random.hh"
#include "cudamem.hh"
#include "ppm.hh"

#include <math.h>

typedef std::vector<double> Vec;
using namespace makemore;

void untwiddle3(const double *lo, const double *hi, unsigned int w, unsigned int h, double *z) {
  assert(w % 2 == 0 && h % 2 == 0);
  unsigned int nw = w / 2;
  unsigned int nh = h / 2;

  unsigned int ilo = 0, ihi = 0;

  unsigned int w3 = w * 3;
  for (unsigned int y = 0; y < h; y += 2) {
    for (unsigned int x = 0; x < w; x += 2) {
      for (unsigned int c = 0; c < 3; ++c) {
        unsigned int p = y * w3 + x * 3 + c;

        double m = lo[ilo++];
        double l = (hi[ihi++] - 0.5) * 2.0;
        double t = (hi[ihi++] - 0.5) * 2.0;
        double s = (hi[ihi++] - 0.5) * 2.0;

        z[p] = m + l + t + s;
        z[p+3] = m - l + t - s;
        z[p+w3] = m + l - t - s;
        z[p+w3+3] = m - l - t + s;
      }
    }
  }
}

int main(int argc, char **argv) {
  assert(argc > 1);
  seedrand();

  unsigned int mbn = 1;
  ZoomProject *p = new ZoomProject(argv[1], mbn);
  unsigned int *mb = new unsigned int[mbn];

  unsigned int nc = p->controlslay->n;
  unsigned int lfn = p->lofreqlay->n;
  unsigned int an = p->attrslay->n;
  unsigned int sn = p->sampleslay->n;
  unsigned int hfn = sn;
  unsigned int dim = round(sqrt(hfn*4/9));
  assert(dim * dim * 9 == hfn * 4);
  unsigned int labn = dim * dim * 3;
  unsigned int cn = p->contextlay->n;
  assert(cn == lfn + an);
  unsigned int csn = sn + cn;

  Dataset *attrs = p->attrs;
  Dataset *hifreq = p->hifreq;
  Dataset *lofreq = p->lofreq;

  assert(hifreq->n == lofreq->n);
  assert(hifreq->k == hfn);
  assert(lofreq->k == lfn);

  Tron *enctron = p->enctron;
  Tron *gentron = p->gentron;
  Tron *encpasstron = p->encpasstron;
  Tron *encgentron = p->encgentron;

  assert(gentron->inn == mbn * (nc + cn));
  double *genin = NULL;
  cumake(&genin, mbn * (nc + cn));
  const double *genout;

  double *gentgt = NULL;
  cumake(&gentgt, mbn * sn);

  unsigned int i = 0;

  double *clo = new double[lfn];
  double *chi = new double[hfn];
  double *lab = new double[labn * 2];

  while (1) {
    hifreq->pick_minibatch(mbn, mb);
    attrs->encude_minibatch(mb, mbn, genin, 0, nc + cn);
    lofreq->encude_minibatch(mb, mbn, genin, an, nc + cn);
    assert(an + lfn == cn);

    double r[nc];
    for (unsigned int j = 0; j < nc; ++j) r[j] = rnd();
    encude(r, nc, genin + cn);

    genout = gentron->feed(genin, NULL);

    lofreq->copy_minibatch(mb, mbn, clo);
    hifreq->copy_minibatch(mb, mbn, chi);
    untwiddle3(clo, chi, dim, dim, lab);

    decude(genout, hfn, chi);
    untwiddle3(clo, chi, dim, dim, lab + labn);

    {
      PPM p;
      p.unvectorize(lab, dim, dim * 2);
      p.write(stdout);
      fprintf(stderr, "frame %u\n", i);
    }

    enctron->sync(0);
    gentron->sync(0);
    ++i;
  }

  return 0;
}







