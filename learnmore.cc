#include "project.hh"
#include "topology.hh"
#include "random.hh"
#include "cudamem.hh"

#include <math.h>

int usage() {
  fprintf(stderr, "Usage: learnmore <--center|--fidelity> [--sequential] [--rate nu] dir.proj\n");
  return 1;
}

int main(int argc, char **argv) {
  seedrand();

  unsigned int mbn = 8;
  double nu = 0.005;
  bool sequential = false;
  Project::ControlSource control_source = Project::CONTROL_SOURCE_UNKNOWN;

  ++argv;
  --argc;
  if (argc < 1)
    return usage();
  while (*argv[0] == '-') {
    const char *arg = argv[0];

    if (!strcmp(arg, "--center")) {
      assert(control_source == Project::CONTROL_SOURCE_UNKNOWN);
      control_source = Project::CONTROL_SOURCE_CENTER;
    } else if (!strcmp(arg, "--fidelity")) {
      assert(control_source == Project::CONTROL_SOURCE_UNKNOWN);
      control_source = Project::CONTROL_SOURCE_TRAINING;

    } else if (!strcmp(arg, "--sequential")) {
      sequential = true;

    } else if (!strcmp(arg, "--rate")) {
      ++argv;
      --argc;
      if (argc < 1)
        return usage();
      nu = strtod(arg, NULL);

    } else {
      return usage();
    }

    ++argv;
    --argc;
    if (argc < 1)
      return usage();
  }

  if (control_source == Project::CONTROL_SOURCE_UNKNOWN)
    return usage();
  if (argc < 1)
    return usage();

  const char *project_dir = argv[0];
  Project *p = open_project(project_dir, 8);

  unsigned int i = 0;
  while (1) {
    p->learn(control_source, nu, i, sequential);
    if (i % 100 == 0) {
      p->report("learnmore", i);
      p->save();
    }
    ++i;
  }

  return 0;
}
