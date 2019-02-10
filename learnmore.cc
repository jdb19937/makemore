#include <string.h>

#include <math.h>

#include "project.hh"
#include "topology.hh"
#include "random.hh"
#include "cudamem.hh"
#include "project.hh"
#include "pipeline.hh"

#include "parson.hh"
#include "warp.hh"

using namespace makemore;

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
  double *tmp_tgt = new double[64 * 64 * 3];
  double *tmp_ctx = new double[72];

  fprintf(stderr, "learnmore mu=%lf nu=%lf pi=%lf xi=%lf yo=%lf wu=%lf\n", mu, nu, pi, xi, yo, wu);

  unsigned int i = 0;
  while (1) {
    double *ctx, *out;

    ctx = pipe->ctxbuf;
    out = pipe->outbuf;
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      Parson *p = parsons->pick();

      for (unsigned int j = 0; j < Parson::nattrs; ++j)
        tmp_ctx[j] = (double)(p->attrs[j] + 0.5) / 256.0;

      tmp_ctx[40] = 0.5;
      tmp_ctx[41] = 0.5;
      tmp_ctx[42] = 0.5;
      tmp_ctx[43] = 0.5;

      if (0) {
        int dx = -3 + (randuint() % 7);
        int dy = -3 + (randuint() % 7);

        iwarp(p->target, dx, dy, 64, 64, tmp_tgt);

        tmp_ctx[40] += ((double)dx / 8.0);
        tmp_ctx[41] += ((double)dy / 8.0);
      } else {
        memcpy(tmp_tgt, p->target, sizeof(double) * 64 * 64 * 3);
      }

      for (unsigned int j = 0; j < Parson::nattrs; ++j)
        *ctx++ = tmp_ctx[j];
      for (unsigned int j = 0; j < Parson::dim * Parson::dim * 3; ++j)
        *out++ = tmp_tgt[j];
    }

    pipe->ctrlock = 0;
    pipe->tgtlock = -1;
    
//    pipe->reencode();
    if (nu > 0 || pi > 0) 
      pipe->burn(-1, nu, pi);

    if (yo > 0 || wu > 0) {
      pipe->reencode();
      pipe->recombine();
      pipe->generate();

      pipe->condition(-1, yo, wu);
    }

    if (i % 100 == 0) {
      pipe->report("learnmore");
      pipe->save();
    }
    ++i;
  }

  return 0;
}
