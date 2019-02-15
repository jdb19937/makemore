#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#include "rule.hh"
#include "wildmap.hh"
#include "script.hh"

using namespace makemore;

void usage() {
  fprintf(stderr, "Usage: morecc [-o script.more] script.txt\n");
  exit(1);
}

int main(int argc, char **argv) {
  if (argc < 2)
    usage();

  const char *infn;
  const char *outfn = "-";

  if (!strcmp(argv[1], "-o")) {
    if (argc < 3)
      usage();
    outfn = argv[2];

    if (argc < 4)
      usage();
    infn = argv[3];
  } else {
    if (argc != 2)
      usage();
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

  Script *script = new Script(infn);

  for (auto i = script->rules.begin(); i != script->rules.end(); ++i) {
    const Rule *rule = &*i;
    rule->save(outfp);
  }

  fclose(outfp);
  delete script;
  return 0;
}
