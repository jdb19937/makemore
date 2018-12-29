#include "project.hh"
#include "topology.hh"
#include "random.hh"
#include "cudamem.hh"
#include "ppm.hh"

#include <math.h>

int main() {
  seedrand();

  unsigned int mbn = 1;
  Project *p = new Project("gen8.proj", mbn);
  unsigned int *mb = new unsigned int[mbn];

  assert(p->contextlay->n == 40);
  assert(p->sampleslay->n == 192);

  unsigned int encsz = mbn * (p->contextlay->n + p->sampleslay->n);
  double *encin = NULL;
  cumake(&encin, encsz);

  assert(p->context->n == p->samples->n);
  assert(p->context->k == p->contextlay->n);
  assert(p->samples->k == p->sampleslay->n);

  Passthrutron *encpasstron = passthrutron(p->contextlay->n, mbn, p->enctron);
  Passthrutron *genpasstron = passthrutron(p->contextlay->n, mbn, p->gentron);
  Compositron *encgentron = compositron(encpasstron, p->gentron);

  unsigned int i = 0;

  double *genin;
  cumake(&genin, mbn * (p->contextlay->n + p->controlslay->n));

  double *cencin = new double[encsz];
  double *cgenout = new double[encsz];
  const double *genout;

//fprintf(stderr, "encgentron in=%u outn=%u\n", encgentron->inn, encgentron->outn);

  while (1) {
    p->context->pick_minibatch(mbn, mb);

    p->context->encude_minibatch(mb, mbn, genin,
      0, p->context->k + p->controlslay->n);

    double r[p->controlslay->n];
    for (unsigned int j = 0; j < p->controlslay->n; ++j) r[j] = 0.5;
    encude(r, p->controlslay->n, genin + p->context->k);

    genout = p->gentron->feed(genin, NULL);

     {
       double lab[192 * 2];
       p->samples->copy(mb[0], lab);
       decude(genout, 192, lab + 192);
       PPM p;
       p.unvectorize(lab, 8, 16);
       p.write(stdout);
     }

     p->enctron->sync(0);
     p->gentron->sync(0);
    ++i;
  }
  return 0;
}
