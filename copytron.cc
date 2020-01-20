#include <string>

#include "supertron.hh"
#include "random.hh"
#include "cudamem.hh"

using namespace makemore;

int usage() {
  fprintf(stderr, "Usage: copytron from.map to.map\n");
  return 1;
}

int main(int argc, char **argv) {
  seedrand();

  --argc;
  ++argv;
  if (argc == 0)
    return usage();
  const char *fromfn = argv[0];

  --argc;
  ++argv;
  if (argc == 0)
    return usage();
  const char *tofn = argv[0];

  FILE *fromfp;
  assert(fromfp = fopen(fromfn, "r"));
  Mapfile *mapfrom = new Mapfile(fromfn);
  Supertron *stfrom = new Supertron(mapfrom);

  FILE *tofp;
  assert(tofp = fopen(tofn, "r+"));
  Mapfile *mapto = new Mapfile(tofn);
  Supertron *stto = new Supertron(mapto);

  assert(stfrom->layers.size() == stto->layers.size());
  for (unsigned int i = 0; i < stfrom->layers.size(); ++i) {
    Supertron::Layer *fromlay = stfrom->layers[i];
    Supertron::Layer *tolay = stto->layers[i];
    assert(fromlay->wn == tolay->wn);
    cucopy(fromlay->weight, fromlay->wn, tolay->weight);
  }

  mapto->save();
  return 0;
}
