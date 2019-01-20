#include "project.hh"
#include "topology.hh"
#include "random.hh"
#include "cudamem.hh"
#include "project.hh"

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
  double nu = 0.001, mu = 0, xi = 0, pi = 0, dcut = 0.3, yo = 0, wu = 0;

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

    } else if (!strcmp(arg, "--pi")) {
      ++argv;
      --argc;
      if (argc < 1)
        return usage();
      pi = strtod(argv[0], NULL);

    } else if (!strcmp(arg, "--yo")) {
      ++argv;
      --argc;
      if (argc < 1)
        return usage();
      yo = strtod(argv[0], NULL);

    } else if (!strcmp(arg, "--wu")) {
      ++argv;
      --argc;
      if (argc < 1)
        return usage();
      wu = strtod(argv[0], NULL);

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
  Project *proj = new Project(project_dir, mbn);

  fprintf(stderr, "learnmore project=%s mu=%lf nu=%lf pi=%lf xi=%lf yo=%lf wu=%lf dcut=%lf\n", project_dir, mu, nu, pi, xi, yo, wu, dcut);

  unsigned int i = 0;
  while (1) {
    if (nu > 0 || pi > 0) {
      proj->load_ctxtgt(stdin);
      proj->train_fidelity(nu, pi, dcut);
    }

    if (mu > 0) {
      proj->load_ctxtgt(stdin);
      proj->train_judgement(mu, dcut);
    }

    if (xi > 0) {
      proj->load_ctxtgt(stdin);
      proj->train_creativity(xi, dcut);
    }

    if (yo > 0) {
      proj->load_ctxtgt(stdin);
      proj->train_recombine(yo, wu, 8);
    }
 
    if (i % 1000 == 0) {
      proj->report("learnmore");
      proj->save();
    }
    ++i;
  }

  return 0;
}
