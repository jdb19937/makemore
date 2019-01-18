#include "project.hh"
#include "pipeline.hh"
#include "topology.hh"
#include "random.hh"
#include "cudamem.hh"
#include "ppm.hh"

#include <math.h>

int usage() {
  fprintf(stderr, "Usage: makemore dir.proj ...\n");
  return 1;
}

int main(int argc, char **argv) {
  assert(argc > 1);
  seedrand();

  unsigned int mbn = 1;
  Pipeline *pipe = new Pipeline(1);
  for (unsigned int i = 1; i < argc; ++i) {
    Project *proj = new Project(argv[i], mbn);
    pipe->add_stage(proj);
  }

  unsigned int iters = 1;
  if (argc > 2)
    iters = (unsigned)atoi(argv[2]);

  unsigned int i = 0;

  unsigned int labn = pipe->outlay->n;
  unsigned int dim = lround(sqrt(labn / 3));
  assert(dim * dim * 3 == labn);
  PPM ppm(dim * 3, dim, 0);

  while (1) {
    pipe->load_ctx_bytes(stdin);
pipe->tgtlock = -1;

    for (unsigned int j = 0, jn = pipe->ctrlay->n; j < jn; ++j)
      pipe->ctrbuf[j] = sigmoid(randgauss() * 0.1);
    pipe->generate();
pipe->reencode();
pipe->generate();
    ppm.pastelab(pipe->outbuf, dim, dim, dim * 0, 0);

    for (unsigned int j = 0, jn = pipe->ctrlay->n; j < jn; ++j)
        pipe->ctrbuf[j] = sigmoid(randgauss() * 0.2);
    pipe->generate();
pipe->reencode();
pipe->generate();
    ppm.pastelab(pipe->outbuf, dim, dim, dim * 1, 0);

    for (unsigned int j = 0, jn = pipe->ctrlay->n; j < jn; ++j)
        pipe->ctrbuf[j] = sigmoid(randgauss());
    pipe->generate();
pipe->reencode();
pipe->generate();
    ppm.pastelab(pipe->outbuf, dim, dim, dim * 2, 0);

#if 0
    proj->load_ctxtgt(stdin);

    proj->passgenerate();
    ppm.pastelab(proj->outbuf, dim, dim, dim * 0, 0);

    proj->regenerate();
    ppm.pastelab(proj->outbuf, dim, dim, dim * 1, 0);

    proj->scramble(0, 0);
    proj->generate();
    ppm.pastelab(proj->outbuf, dim, dim, dim * 2, 0);

    proj->scramble(0, 0.5);
    proj->generate();
    ppm.pastelab(proj->outbuf, dim, dim, dim * 3, 0);

    proj->scramble(0, 1);
    proj->generate(iters);
    ppm.pastelab(proj->outbuf, dim, dim, dim * 4, 0);
#endif

    ppm.write(stdout);

    ++i;
    if (i % 100 == 0) {
      pipe->load();
    }
  }

  return 0;
}
