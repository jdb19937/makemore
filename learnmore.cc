#include "project.hh"
#include "topology.hh"
#include "random.hh"
#include "cudamem.hh"

#include <math.h>

int usage() {
  fprintf(stderr, "Usage: learnmore <--center|--fidelity> dir.proj\n");
  return 1;
}

int main(int argc, char **argv) {
  seedrand();

  unsigned int mbn = 8;
  double nu = 0.001;

  if (argc < 2)
    return usage();
  Project::ControlSource control_source;
  if (!strcmp(argv[1], "--center")) {
    control_source = Project::CONTROL_SOURCE_CENTER;
    ++argv;
    --argc;
  } else if (!strcmp(argv[1], "--fidelity")) {
    control_source = Project::CONTROL_SOURCE_TRAINING;
    ++argv;
    --argc;
  } else {
    return usage();
  }

  if (argc < 2)
    return usage();
  const char *project_dir = argv[1];
  Project *p = open_project(project_dir, 8);

  unsigned int i = 0;
  while (1) {
    p->learn(control_source, nu, i);
    if (i % 100 == 0) {
      p->report("learnmore", i);
      p->save();
    }
    ++i;
  }

  return 0;
}
