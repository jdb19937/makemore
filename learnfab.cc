#include <stdio.h>

#include "vocab.hh"
#include "shibboleth.hh"
#include "script.hh"
#include "confab.hh"

using namespace makemore;

int main() {
  seedrand();

  unsigned int mbn = 8;
  Confab confab("test.confab", mbn);
  confab.load();

  Script scr("script.txt");

  assert(confab.ctxlay->n == 512);
  assert(confab.tgtlay->n == 512);
  assert(sizeof(Shibboleth) >= 512 * sizeof(double));
  assert(confab.mbn == mbn);

  Shibboleth req, rsp;

  int i = 0;
  while (1) {
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      scr.pick(&req, &rsp);

      memcpy(confab.ctxbuf + mbi * 512, (double *)&req, sizeof(double) * 512);
      memcpy(confab.tgtbuf + mbi * 512, (double *)&rsp, sizeof(double) * 512);
    }
 
    confab.burn(0.002, 0.002);

    if (i % 1000 == 0) {
      confab.report("learnfab");
      confab.save();
    }

    ++i;
  }

  return 0;
}
