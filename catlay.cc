#include "layout.hh"

int main(int argc, char **argv) {
  Layout x;

  for (int i = 1; i < argc; ++i) {
    const char *fn = argv[i];
    Layout y;
    y.load_file(fn);
    x += y;
  }

  x.save(stdout);
}
