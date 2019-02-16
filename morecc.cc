#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#include "rule.hh"
#include "wildmap.hh"
#include "script.hh"

using namespace makemore;

void usage() {
  fprintf(stderr, "Usage: morecc [-o|--output script.more] [script.txt]\n");
  exit(1);
}

int main(int argc, char **argv) {
  const char *infn = "-";
  const char *outfn = "-";

  if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))
    usage();

  if (!strcmp(argv[1], "-o") || !strcmp(argv[1], "--output")) {
    if (argc < 3)
      usage();
    outfn = argv[2];

    if (argc >= 4)
      infn = argv[3];
  } else {
    if (argc > 2)
      usage();
    if (argc >= 2)
      infn = argv[1];
  }

  FILE *outfp;
  if (!strcmp(outfn, "-")) {
    outfp = stdout;
  } else {
    outfp = fopen(outfn, "w");
    if (!outfp) {
      fprintf(stderr, "morecc: %s: %s\n", outfn, strerror(errno));
      exit(1);
    }
  }

  FILE *infp;
  if (!strcmp(infn, "-")) {
    infp = stdin;
  } else {
    infp = fopen(infn, "r");
    if (!infp) {
      fprintf(stderr, "morecc: %s: %s\n", infn, strerror(errno));
      exit(1);
    }
  }

  char buf[4096];
  while (1) {
    *buf = 0;
    char *unused = fgets(buf, sizeof(buf) - 1, infp);
    buf[sizeof(buf) - 1] = 0;
    char *p = strchr(buf, '\n');
    if (!p)
      break;
    *p = 0;

    p = strchr(buf, '#');
    if (p)
      *p = 0;
    const char *q = buf;
    while (*q == ' ')
      ++q;
    if (!*q)
      continue;

    Rule r;
    r.parse(q);
    r.save(outfp);
  }


  fclose(infp);
  fclose(outfp);
  return 0;
}
