#include <math.h>

#include "project.hh"
#include "topology.hh"
#include "random.hh"
#include "cudamem.hh"
#include "project.hh"
#include "pipeline.hh"
#include "ppm.hh"
#include "warp.hh"
#include "parson.hh"

using namespace makemore;

int usage() {
  fprintf(stderr,
    "Usage: makemore\n"
    "  dir.proj\n"
  );
  return 1;
}

Pipeline *open_pipeline(unsigned int mbn) {
  Pipeline *pipe = new Pipeline(mbn);
  pipe->add_stage(new Project("new8.proj", mbn));
  pipe->add_stage(new Project("new16.proj", mbn));
  pipe->add_stage(new Project("new32.proj", mbn));
  pipe->add_stage(new Project("new64.proj", mbn));
  return pipe;
}

int main(int argc, char **argv) {
  seedrand();

  unsigned int mbn = 4;
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

  unsigned int i = 0;
  unsigned int dim = 64;
  assert(Parson::dim == dim);
  assert(mbn == 4);
  PPM ppm(dim * 4, dim * 4);
  Parson **pbuf = new Parson *[mbn];

  while (1) {
    pipe->load();

    double *ctx, *out;

    ctx = pipe->ctxbuf;
    out = pipe->outbuf;
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      Parson *p = parsons->pick();
      pbuf[mbi] = p;

      for (unsigned int j = 0; j < Parson::nattrs; ++j)
        *ctx++ = (double)(p->attrs[j] + 0.5) / 256.0;
      for (unsigned int j = 0; j < Parson::dim * Parson::dim * 3; ++j)
        *out++ = p->target[j];
    }

    for (unsigned int i = 0; i < 4; ++i)
      ppm.pastelab(pipe->outbuf + (dim*dim*3*i), dim, dim, dim * i, 0);

    pipe->ctrlock = 0;
    pipe->tgtlock = -1;
    pipe->reencode();

    pipe->ctrlock = -1;
    pipe->tgtlock = 0;
    pipe->generate();

    for (unsigned int i = 0; i < 4; ++i)
      ppm.pastelab(pipe->outbuf + (dim*dim*3*i), dim, dim, dim * i, dim);

#if 1
    pipe->recombine();
    pipe->generate();


    for (unsigned int i = 0; i < 4; ++i)
      ppm.pastelab(pipe->outbuf + (dim*dim*3*i), dim, dim, dim * i, dim*2);

    pipe->ctxbuf[69] = 1.0;
    pipe->ctxbuf[70] = 1.0;
    pipe->ctxbuf[71] = 1.0;

    pipe->generate();
    for (unsigned int i = 0; i < 4; ++i)
      ppm.pastelab(pipe->outbuf + (dim*dim*3*i), dim, dim, dim * i, dim*3);
#endif


#if 0
    out = pipe->outbuf;
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      Parson *p = pbuf[mbi];
      for (unsigned int j = 0; j < Parson::dim * Parson::dim * 3; ++j)
        *out++ = p->target[j];
    }

    for (unsigned int i = 0; i < 4; ++i) {
      double tmp[dim*dim*3];
      double wx = sigmoid(randgauss() * 1.0);
      double wy = sigmoid(randgauss() * 1.0);
      double wz = sigmoid(randgauss() * 1.0);
      double wr = sigmoid(randgauss() * 1.0);

      pipe->ctxbuf[40 + i * pipe->ctxlay->n] = wx;
      pipe->ctxbuf[41 + i * pipe->ctxlay->n] = wy;
      pipe->ctxbuf[42 + i * pipe->ctxlay->n] = wz;
      pipe->ctxbuf[43 + i * pipe->ctxlay->n] = wr;

      warp(pipe->outbuf + dim*dim*3*i, wx, wy, wz, wr, 64, 64, tmp);
      ppm.pastelab(tmp, dim, dim, dim * i, dim*2);

      memcpy(pipe->outbuf + dim*dim*3*i, tmp, sizeof(double) * dim*dim*3);
    }

    pipe->ctrlock = 0;
    pipe->tgtlock = -1;
    pipe->reencode();

    pipe->ctrlock = -1;
    pipe->tgtlock = 0;
    pipe->generate();

    for (unsigned int i = 0; i < 4; ++i)
      ppm.pastelab(pipe->outbuf + (dim*dim*3*i), dim, dim, dim * i, dim*3);
#endif

    ppm.write(stdout);
    break;
  }

  return 0;
}
