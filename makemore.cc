#include "project.hh"
#include "topology.hh"
#include "random.hh"
#include "cudamem.hh"

#include <math.h>

int main(int argc, char **argv) {
  assert(argc > 1);
  const char *project_dir = argv[1];
  Project *p = open_project(project_dir, 1);

  p->generate(
    Project::CONTEXT_SOURCE_TRAINING,
    Project::CONTROL_SOURCE_CENTER
  );

  fprintf(stderr, "generated %u doubles\n", p->outputlay->n);

  p->write_ppm(stdout);

  return 0;
}
