#include <stdio.h>

#include "vocab.hh"
#include "shibboleth.hh"
#include "script.hh"
#include "confab.hh"
#include "brane.hh"

using namespace makemore;

int main() {
  seedrand();

  unsigned int mbn = 8;
  Confab confab("test.confab", mbn);
  confab.load();

  Script script;
  script.load("test.confab/script.more");
  Brane brane(&confab);

  assert(sizeof(Shibboleth) * 3 == confab.ctxlay->n * sizeof(double));
  assert(sizeof(Shibboleth) * 6 == confab.tgtlay->n * sizeof(double));
  assert(confab.mbn == mbn);

  Shibboleth req, rsp;
  Rule rules[mbn];

  int i = 0;
  while (1) {
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      const Rule *r = script.pick();
      rules[mbi] = *r;
      rules[mbi].prepare();
    }
    brane.burn(rules, mbn, 0.001);

    if (i % 1000 == 0) {
      confab.report("learnfab");
      confab.save();
    }

    ++i;
  }

  return 0;
}
