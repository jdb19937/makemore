#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "mork.hh"

using namespace makemore;

int main(int argc, char **argv) {
  ++argv;
  --argc;

  int w = 0, h = 0;

  while (argc > 0 && !strncmp(argv[0], "--", 2)) {
    if (!strcmp(argv[0], "--width")) {
      ++argv;
      --argc;
      assert(argc > 0);
      w = atoi(argv[0]);
      ++argv;
      --argc;
    } else if (!strcmp(argv[0], "--height")) {
      ++argv;
      --argc;
      assert(argc > 0);
      h = atoi(argv[0]);
      ++argv;
      --argc;
    } else {
      assert(0);
    }
  }

  assert(w > 0);
  assert(h > 0);

  uint8_t *rgb = new uint8_t[w * h * 3]();
  int x = 0, y = 0;

  while (1) {
    int c = getc(stdin);
    if (c == EOF)
      break;
    if (c == '\n') {
      x = 0;
      y += 3;
      continue;
    }

    assert(c >= 0 && c < 256);
    int cx = 2 * (int)(c % 16);
    int cy = 3 * (int)(c / 16);

    for (int dy = 0; dy < 3; ++dy) {
      if (y + dy >= h) continue;
      for (int dx = 0; dx < 2; ++dx) {
        if (x + dx >= w) continue;
        for (int q = 0; q < 3; ++q) {
          rgb[(y + dy) * w * 3 + (x + dx) * 3 + q] = mork_rgb[(cy + dy) * 32 * 3 + (cx + dx) * 3 + q];
        }
      }
    }

    x += 2;
  }

  printf("P6\n%d %d\n255\n", w, h);
  fwrite(rgb, w * h * 3, 1, stdout);

  return 0;
}
