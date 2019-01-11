#include "project.hh"
#include "topology.hh"
#include "random.hh"
#include "cudamem.hh"
#include "scrambler.hh"

#include <math.h>

int usage() {
  fprintf(stderr,
    "Usage: learnmore\n"
    "  [--nu nu]\n"
    "  [--mbn mbn]\n"
    "  dir.proj\n"
  );
  return 1;
}

int main(int argc, char **argv) {
  seedrand();

  unsigned int mbn = 8;
  double nu = 0.005, mu = 0.005, xi = 0.0;

  ++argv;
  --argc;
  if (argc < 1)
    return usage();
  while (*argv[0] == '-') {
    const char *arg = argv[0];

    if (!strcmp(arg, "--mbn")) {
      ++argv;
      --argc;
      if (argc < 1)
        return usage();
      mbn = (unsigned int)atoi(argv[0]);

    } else if (!strcmp(arg, "--nu")) {
      ++argv;
      --argc;
      if (argc < 1)
        return usage();
      nu = strtod(argv[0], NULL);
    } else if (!strcmp(arg, "--mu")) {
      ++argv;
      --argc;
      if (argc < 1)
        return usage();
      mu = strtod(argv[0], NULL);

    } else if (!strcmp(arg, "--xi")) {
      ++argv;
      --argc;
      if (argc < 1)
        return usage();
      xi = strtod(argv[0], NULL);

    } else {
      return usage();
    }

    ++argv;
    --argc;
    if (argc < 1)
      return usage();
  }

  if (argc < 1)
    return usage();

  const char *project_dir = argv[0];
  Scrambler *proj = new Scrambler(project_dir, mbn);

  fprintf(stderr, "learnmore project=%s mu=%lf nu=%lf xi=%lf\n", project_dir, mu, nu, xi);

  unsigned int i = 0;
  while (1) {
    proj->load_ctxtgt(stdin);
    proj->present(nu, mu, xi);

    if (i % 1000 == 0) {
      proj->report("learnmore");
      proj->save();
    }
    ++i;
  }

  return 0;
}
