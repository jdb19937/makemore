#include "project.hh"
#include "topology.hh"
#include "random.hh"
#include "cudamem.hh"

#include <math.h>

int usage() {
  fprintf(stderr, "Usage: makemore [--raw|--ppm] [--dev deviation] [--fidelity] [--batch mbn] dir.proj\n");
  return 1;
}

int main(int argc, char **argv) {
  seedrand();

  unsigned int mbn = 1;
  unsigned int loadint = 100;
  double dev = 0.0;
  int fidelity = 0;
  Project::OutputFormat output_format = Project::OUTPUT_FORMAT_UNKNOWN;

  ++argv;
  --argc;
  if (argc < 1)
    return usage();
  while (*argv[0] == '-') {
    const char *arg = argv[0];

    if (!strcmp(arg, "--fidelity")) {
      fidelity = 1;
    } else if (!strcmp(arg, "--target")) {
      fidelity = 2;

    } else if (!strcmp(arg, "--ppm")) {
      assert(output_format == Project::OUTPUT_FORMAT_UNKNOWN);
      output_format = Project::OUTPUT_FORMAT_PPM;

    } else if (!strcmp(arg, "--raw")) {
      assert(output_format == Project::OUTPUT_FORMAT_UNKNOWN);
      output_format = Project::OUTPUT_FORMAT_RAW;

    } else if (!strcmp(arg, "--batch")) {
      ++argv;
     --argc;
     if (argc < 1)
       return usage();
      mbn = (unsigned)atoi(argv[0]);
      assert(mbn > 0);
    } else if (!strcmp(arg, "--dev")) {
      ++argv;
     --argc;
     if (argc < 1)
       return usage();

      dev = strtod(argv[0], NULL);
      assert(dev >= 0);
    } else if (!strcmp(arg, "--loadint")) {
      ++argv;
     --argc;
     if (argc < 1)
       return usage();

      loadint = (unsigned)atoi(argv[0]);
      assert(loadint > 0);
    } else {
      return usage();
    }

    ++argv;
    --argc;
    if (argc < 1)
      return usage();
  }

  if (output_format == Project::OUTPUT_FORMAT_UNKNOWN)
    output_format = Project::OUTPUT_FORMAT_RAW;

  const char *project_dir = argv[0];
  Project *p = open_project(project_dir, mbn);

  unsigned int i = 0;
  while (1) {
    if (fidelity == 0) {
      p->loadcontext(stdin);
      p->randcontrols(dev);
      p->nulladjust();
      p->generate();
    } else if (fidelity == 1) {
      p->loadbatch(stdin);
      p->nulladjust();
      p->regenerate();
    } else {
      assert(0);
    }
      

    if (output_format == Project::OUTPUT_FORMAT_PPM) {
      p->write_ppm(stdout);
    } else if (output_format == Project::OUTPUT_FORMAT_RAW) {
      size_t ret = fwrite(p->output(), 1, mbn * p->outputlay->n, stdout);
      assert(ret == mbn * p->outputlay->n);
    }

    if (i % 100 == 0) {
      fprintf(stderr, "makemore i=%d\n", i);
    }
    if (i % loadint == 0) {
      p->load();
    }

    ++i;
  }

  return 0;
}
