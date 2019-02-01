#include "project.hh"
#include "topology.hh"
#include "random.hh"
#include "cudamem.hh"
#include "project.hh"
#include "pipeline.hh"

#include "parson.hh"

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

Pipeline *open_pipeline(unsigned int mbn) {
  Pipeline *pipe = new Pipeline(mbn);
  pipe->add_stage(new Project("test8.proj", mbn));
  pipe->add_stage(new Project("test16.proj", mbn));
  pipe->add_stage(new Project("test32.proj", mbn));
  pipe->add_stage(new Project("test64.proj", mbn));
  return pipe;
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

  ParsonDB *parsons = new ParsonDB(argv[0]);
  Pipeline *pipe = open_pipeline(mbn);

  fprintf(stderr, "learnmore mu=%lf nu=%lf pi=%lf xi=%lf yo=%lf wu=%lf\n", mu, nu, pi, xi, yo, wu);

  unsigned int i = 0;
  while (1) {
    pipe->load();


    double *ctx = pipe->ctxbuf, *out = pipe->outbuf;
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      Parson *p = parsons->pick();

      for (unsigned int j = 0; j < Parson::nattrs; ++j)
        *ctx++ = (double)(p->attrs[j] + 0.5) / 256.0;
      for (unsigned int j = 0; j < Parson::dim * Parson::dim * 3; ++j)
        *out++ = p->target[j];
    }

    pipe->ctrlock = 0;
    pipe->tgtlock = -1;

    pipe->reencode();
    pipe->burn(-1, nu, pi);

    if (i % 100 == 0) {
      pipe->report("learnmore");
      pipe->save();
    }
    ++i;
  }

  return 0;
}
