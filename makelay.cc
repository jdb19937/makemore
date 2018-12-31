#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "layout.hh"
#include "random.hh"

void usage() {
  fprintf(stderr, "Usage: makelay [\n");
  fprintf(stderr, "   -grid dim [coverage [chans]] |\n");
  fprintf(stderr, "  -rand n [coverage] |\n");
  fprintf(stderr, "  -center n [radius]\n");
  fprintf(stderr, "]\n");
  exit(1);
}

int main(int argc, char **argv) {
  if (argc < 3)
    usage();

  enum { GRID, RAND, CENTER } type;

  if (!strcmp(argv[1], "-grid")) {
    type = GRID;
  } else if (!strcmp(argv[1], "-rand")) {
    type = RAND;
  } else if (!strcmp(argv[1], "-center")) {
    type = CENTER;
  } else {
    usage();
  }

  unsigned int chan = 1;

  Layout *lay;
  if (type == GRID) {
    unsigned int dim = atoi(argv[2]);
    double cov = 1.0;
    if (argc > 3) cov = strtod(argv[3], NULL);
    if (argc > 4) chan = atoi(argv[4]);
    lay = Layout::new_square_grid(dim, cov, chan);
  } else if (type == RAND) {
    unsigned int n = atoi(argv[2]);
    double cov = 1.0;
    if (argc > 3) cov = strtod(argv[3], NULL);
    lay = Layout::new_square_random(n, cov);
  } else if (type == CENTER) {
    unsigned int n = atoi(argv[2]);
    double r = 1.0;
    if (argc > 3) r = strtod(argv[3], NULL);
    lay = Layout::new_square_center(n, r);
  } else {
    assert(0);
  }
    
  lay->save(stdout);
  return 0;
}

