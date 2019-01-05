#include "project.hh"
#include "topology.hh"
#include "random.hh"
#include "cudamem.hh"

#include <math.h>

int usage() {
  fprintf(stderr,
    "Usage: learnmore\n"
    "  [--rate nu]\n"
    "  [--dpres dpres]\n"
    "  [--fpres fpres]\n"
    "  [--cpres cpres]\n"
    "  [--zpres zpres]\n"
    "  [--fcut fcut]\n"
    "  [--dcut dcut]\n"
    "  [--batch mbn]\n"
    "  dir.proj\n"
  );
  return 1;
}

int main(int argc, char **argv) {
  seedrand();

  unsigned int mbn = 8;
  double nu = 0.005;
  double dpres = 0.0;
  double fpres = 1.0;
  double cpres = 0;
  double zpres = 0;
  double fcut = 0;
  double dcut = 0;

  ++argv;
  --argc;
  if (argc < 1)
    return usage();
  while (*argv[0] == '-') {
    const char *arg = argv[0];

    if (!strcmp(arg, "--batch")) {
      ++argv;
      --argc;
      if (argc < 1)
        return usage();
      mbn = (unsigned int)atoi(argv[0]);

    } else if (!strcmp(arg, "--rate")) {
      ++argv;
      --argc;
      if (argc < 1)
        return usage();
      nu = strtod(argv[0], NULL);

    } else if (!strcmp(arg, "--dpres")) {
      ++argv;
      --argc;
      if (argc < 1)
        return usage();
      dpres = strtod(argv[0], NULL);

    } else if (!strcmp(arg, "--fpres")) {
      ++argv;
      --argc;
      if (argc < 1)
        return usage();
      fpres = strtod(argv[0], NULL);

    } else if (!strcmp(arg, "--cpres")) {
      ++argv;
      --argc;
      if (argc < 1)
        return usage();
      cpres = strtod(argv[0], NULL);

    } else if (!strcmp(arg, "--zpres")) {
      ++argv;
      --argc;
      if (argc < 1)
        return usage();
      zpres = strtod(argv[0], NULL);

    } else if (!strcmp(arg, "--fcut")) {
      ++argv;
      --argc;
      if (argc < 1)
        return usage();
      fcut = strtod(argv[0], NULL);

    } else if (!strcmp(arg, "--dcut")) {
      ++argv;
      --argc;
      if (argc < 1)
        return usage();
      dcut = strtod(argv[0], NULL);

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

  fprintf(stderr, "project_dir=%s dpres=%lf fpres=%lf cpres=%lf zpres=%lf fcut=%lf dcut=%lf mbn=%u\n",
    project_dir, dpres, fpres, cpres, zpres, fcut, dcut, mbn
  );

  Project *p = open_project(project_dir, 8);

  unsigned int i = 0;
  while (1) {
    p->learn(stdin, nu, dpres, fpres, cpres, zpres, fcut, dcut, i);

    if (i % 1000 == 0) {
      p->report("learnmore", i);
      p->save();
    }
    ++i;
  }

  return 0;
}
