#include "project.hh"
#include "topology.hh"
#include "multitron.hh"
#include "random.hh"
#include "mapfile.hh"

using namespace makemore;

int usage() {
  fprintf(stderr, "Usage: makemap [--activated] [--normalized] [--dispersion d] file.top file.map\n");
  return 1;
}

int main(int argc, char **argv) {
  bool activated = false;
  bool normalized = false;
  double dispersion = 1.0;

  ++argv;
  --argc;
  if (argc < 1)
    return usage();

  while (*argv[0] == '-') {
    const char *arg = argv[0];

    if (!strcmp(arg, "--activated")) {
      activated = true;

    } else if (!strcmp(arg, "--normalized")) {

      normalized = true;

    } else if (!strcmp(arg, "--dispersion")) {
      ++argv;
      --argc;
      if (argc < 1)
        return usage();

      dispersion = strtod(argv[0], NULL);

    } else {
      return usage();
    }

    ++argv;
    --argc;
    if (argc < 1)
      return usage();
  }

  if (argc < 2)
    return usage();

  seedrand();

  Topology *top = new Topology;
  const char *topfn = argv[0];
  top->load_file(topfn);

  const char *mapfn = argv[1];
  Mapfile *mapfile = new Mapfile(mapfn);
  Multitron *tron = new Multitron(*top, mapfile, 1, activated, normalized);

  tron->randomize(dispersion);
  mapfile->save();

  delete tron;
  delete top;
  delete mapfile;
  return 0;
}
