#include "project.hh"
#include "topology.hh"
#include "random.hh"
#include "cudamem.hh"

#include <math.h>

int usage() {
  fprintf(stderr, "Usage: makemore <--center|--random|--fidelity> dir.proj\n");
  return 1;
}

int main(int argc, char **argv) {
  seedrand();

  unsigned int mbn = 1;
  Project::ControlSource control_source = Project::CONTROL_SOURCE_UNKNOWN;
  Project::ContextSource context_source = Project::CONTEXT_SOURCE_UNKNOWN;
  Project::OutputFormat output_format = Project::OUTPUT_FORMAT_UNKNOWN;

  ++argv;
  --argc;
  if (argc < 1)
    return usage();
  while (*argv[0] == '-') {
    const char *arg = argv[0];

    if (!strcmp(arg, "--center")) {
      assert(control_source == Project::CONTROL_SOURCE_UNKNOWN);
      control_source = Project::CONTROL_SOURCE_CENTER;
    } else if (!strcmp(arg, "--random")) {
      assert(control_source == Project::CONTROL_SOURCE_UNKNOWN);
      control_source = Project::CONTROL_SOURCE_RANDOM;
    } else if (!strcmp(arg, "--fidelity")) {
      assert(control_source == Project::CONTROL_SOURCE_UNKNOWN);
      control_source = Project::CONTROL_SOURCE_TRAINING;

    } else if (!strcmp(arg, "--stdin")) {
      assert(context_source == Project::CONTEXT_SOURCE_UNKNOWN);
      context_source = Project::CONTEXT_SOURCE_STDIN;

    } else if (!strcmp(arg, "--train")) {
      assert(context_source == Project::CONTEXT_SOURCE_UNKNOWN);
      context_source = Project::CONTEXT_SOURCE_TRAINING;

    } else if (!strcmp(arg, "--ppm")) {
      assert(output_format == Project::OUTPUT_FORMAT_UNKNOWN);
      output_format = Project::OUTPUT_FORMAT_PPM;

    } else if (!strcmp(arg, "--raw")) {
      assert(output_format == Project::OUTPUT_FORMAT_UNKNOWN);
      output_format = Project::OUTPUT_FORMAT_RAW;

    } else {
      return usage();
    }

    ++argv;
    --argc;
    if (argc < 1)
      return usage();
  }

  if (context_source == Project::CONTEXT_SOURCE_UNKNOWN)
    context_source = Project::CONTEXT_SOURCE_TRAINING;
  if (control_source == Project::CONTROL_SOURCE_UNKNOWN)
    control_source = Project::CONTROL_SOURCE_RANDOM;
  if (output_format == Project::OUTPUT_FORMAT_UNKNOWN)
    output_format = Project::OUTPUT_FORMAT_RAW;

  const char *project_dir = argv[0];
  Project *p = open_project(project_dir, 1);

  unsigned int i = 0;
  while (1) {
    p->generate(
      context_source,
      control_source
    );

    if (output_format == Project::OUTPUT_FORMAT_PPM) {
      p->write_ppm(stdout);
    } else if (output_format == Project::OUTPUT_FORMAT_RAW) {
      size_t ret = fwrite(p->output(), sizeof(double), mbn * p->outputlay->n, stdout);
      assert(ret == mbn * p->outputlay->n);
    }

    if (i % 100 == 0) {
      fprintf(stderr, "makemore i=%d\n", i);
      p->load();
    }

    ++i;
  }

  return 0;
}
