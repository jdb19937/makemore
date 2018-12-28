#include "project.hh"
#include "topology.hh"
#include "random.hh"
#include "cudamem.hh"

#include <math.h>

int main() {
#if 0
  Topology *enctop = new Topology;
  enctop->load_file("gen8.proj/enc.top");

  Network *encnet = new Network(enctop, 0, 1, "gen8.proj/enc.net");
  encnet->tron->randomize();
#endif

  Project *p = new Project("gen8.proj");
  unsigned int mbn = p->mbn;
  unsigned int *mb = new unsigned int[mbn];

  unsigned int encsz = mbn * (p->contextlay->n + p->sampleslay->n);
  double *encin = NULL;
  cumake(&encin, encsz);

  assert(p->context->n == p->samples->n);
  assert(p->context->k == p->contextlay->n);
  assert(p->samples->k == p->sampleslay->n);

  Passthrutron *encpasstron = passthrutron(p->contextlay->n, mbn, p->enctron);
  Passthrutron *genpasstron = passthrutron(p->contextlay->n, mbn, p->gentron);
  Compositron *encgentron = compositron(encpasstron, genpasstron);

  unsigned int i = 0;
  double cerr2 = 0.5;

  double *cencin = new double[encsz];
  double *cgenout = new double[encsz];
  const double *genout;

//fprintf(stderr, "encgentron in=%u outn=%u\n", encgentron->inn, encgentron->outn);

  while (1) {
    p->samples->pick_minibatch(mbn, mb);
    p->samples->encude_minibatch(mb, mbn, encin, p->context->k, p->context->k + p->samples->k);
    p->context->encude_minibatch(mb, mbn, encin, 0, p->context->k + p->samples->k);

    genout = encgentron->feed(encin, NULL);

//    encgentron->target(encin);
//    encgentron->train(0.01);

//fprintf(stderr, "hi %u %u (%u %u) encin=%lu genout=%lu\n", i, encsz, p->context->k, p->samples->k, encin, genout);

    decude(encin, encsz, cencin);
    decude(genout, encsz, cgenout);

    double err2 = 0;
    for (unsigned int j = 0; j < encsz; ++j)
      err2 += (cencin[j] - cgenout[j]) * (cencin[j] - cgenout[j]);
    err2 /= encsz;
    err2 = sqrt(err2);

    cerr2 *= 0.999;
    cerr2 += 0.001 * err2;

    if (i % 1000 == 0) {
       fprintf(stderr, "i=%u encsz=%u (%lf, %lf) cerr2=%lf\n", i, encsz, cencin[100], cgenout[100], cerr2);
    }
    ++i;
  }
  return 0;
}
