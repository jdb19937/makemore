#include "project.hh"
#include "topology.hh"
#include "random.hh"
#include "cudamem.hh"
#include "ppm.hh"

#include <math.h>

using namespace makemore;

int main() {
  seedrand();

  unsigned int mbn = 1;
  SimpleProject *p = new SimpleProject("gen8.proj", mbn);
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

  double *cencin = new double[encsz];
  double *cgenout = new double[encsz];
  const double *genout;

//fprintf(stderr, "encgentron in=%u outn=%u\n", encgentron->inn, encgentron->outn);

  while (1) {
    p->context->pick_minibatch(mbn, mb);

    p->context->encude_minibatch(mb, mbn, encin,
      0, p->context->k + p->samples->k);
    p->samples->encude_minibatch(mb, mbn, encin,
      p->context->k, p->context->k + p->samples->k);

    genout = encgentron->feed(encin, NULL);

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
