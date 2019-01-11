#include "scrambler.hh"
#include "topology.hh"
#include "random.hh"
#include "cudamem.hh"
#include "ppm.hh"

#include <math.h>

int usage() {
  fprintf(stderr, "Usage: makemore dir.proj\n");
  return 1;
}

int main(int argc, char **argv) {
  assert(argc > 1);
  seedrand();

  unsigned int mbn = 1;
  const char *project_dir = argv[1];
  Scrambler *proj = new Scrambler(project_dir, mbn);

  unsigned int iters = 1;
  if (argc > 2)
    iters = (unsigned)atoi(argv[2]);

  unsigned int i = 0;

  unsigned int labn = proj->outlay->n;
  unsigned int dim = lround(sqrt(labn / 3));
  assert(dim * dim * 3 == labn);
  PPM ppm(dim * 5, dim, 0);

  while (1) {
    proj->load_ctxtgt(stdin);

    proj->passgenerate();
    ppm.pastelab(proj->outbuf, dim, dim, dim * 0, 0);

    proj->regenerate();
    ppm.pastelab(proj->outbuf, dim, dim, dim * 1, 0);

    proj->generate(0, iters);
    ppm.pastelab(proj->outbuf, dim, dim, dim * 2, 0);

    proj->generate(0.5, iters);
    ppm.pastelab(proj->outbuf, dim, dim, dim * 3, 0);

    proj->generate(1, iters);
    ppm.pastelab(proj->outbuf, dim, dim, dim * 4, 0);

    ppm.write(stdout);

    ++i;
    if (i % 100 == 0) {
      proj->load();
    }
  }

  return 0;
}
