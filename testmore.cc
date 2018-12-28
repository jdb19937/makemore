#include "project.hh"
#include "topology.hh"
#include "random.hh"

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
  double *encin = new double[mbn * (p->contextlay->n + p->sampleslay->n)];

  unsigned int i = 0;
  while (1) {
    p->samples->pick_minibatch(mbn, mb);
    p->samples->encude_minibatch(mb, mbn, encin, p->contextlay->n, p->contextlay->n + p->sampleslay->n);
    assert(p->context->n == p->samples->n);
    p->context->encude_minibatch(mb, mbn, encin, 0, p->contextlay->n + p->sampleslay->n);

    const double *encout = p->enctron->feed(encin, NULL);
    const double *genout = p->gentron->feed(encout, p->enctron->foutput());

    if (i % 1000 == 0) {
       p->gentron->report();
    }
    ++i;
  }
  return 0;
}
