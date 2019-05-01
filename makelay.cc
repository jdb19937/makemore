#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "layout.hh"
#include "random.hh"

using namespace makemore;

void usage() {
  fprintf(stderr, "Usage: makelay [\n");
  fprintf(stderr, "  -rect w h |\n");
  fprintf(stderr, "  -grid dim [coverage [chans]] |\n");
  fprintf(stderr, "  -text chans |\n");
  fprintf(stderr, "  -rand n [coverage] |\n");
  fprintf(stderr, "  -center n [radius]\n");
  fprintf(stderr, "]\n");
  exit(1);
}

int main(int argc, char **argv) {
  seedrand();
  if (argc < 2)
    usage();

  enum { GRID, RECT, RAND, CENTER, TEXT, LINE } type;

  if (!strcmp(argv[1], "-grid")) {
    type = GRID;
  } else if (!strcmp(argv[1], "-rect")) {
    type = RECT;
  } else if (!strcmp(argv[1], "-rand")) {
    type = RAND;
  } else if (!strcmp(argv[1], "-center")) {
    type = CENTER;
  } else if (!strcmp(argv[1], "-text")) {
    type = TEXT;
  } else if (!strcmp(argv[1], "-line")) {
    type = LINE;
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
  } else if (type == RECT) {
    unsigned int w = atoi(argv[2]);
    assert(argc > 3);
    unsigned int h = atoi(argv[3]);
    assert(argc > 4);
    unsigned int c = atoi(argv[4]);
    lay = Layout::new_rect(w, h, c);
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
  } else if (type == TEXT) {
    chan = 4;
    if (argc > 2) chan = atoi(argv[2]);
    lay = Layout::new_text(chan);
  } else if (type == LINE) {
    unsigned int dim = atoi(argv[2]);
    double cov = 1.0;
    if (argc > 3) cov = strtod(argv[3], NULL);
    if (argc > 4) chan = atoi(argv[4]);
    lay = Layout::new_line(dim, cov, chan);
  } else {
    assert(0);
  }
    
  lay->save(stdout);
  return 0;
}

