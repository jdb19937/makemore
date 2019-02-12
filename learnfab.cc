#include <stdio.h>

#include "vocab.hh"
#include "tagbag.hh"
#include "script.hh"
#include "confab.hh"

using namespace makemore;

int main() {
  seedrand();

  unsigned int mbn = 8;
  Confab confab("test.confab", mbn);
  confab.load();

  Script scr("script.txt");

  unsigned int tbn = 4;
  assert(confab.ctxlay->n == 256 * tbn);
  assert(confab.tgtlay->n == 256 * tbn);
  assert(confab.mbn == mbn);

  Tagbag *req = new Tagbag[tbn];
  Tagbag *rsp = new Tagbag[tbn];

  int i = 0;
  while (1) {
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      scr.pick(req, rsp, tbn);

#if 0
unsigned int aa = randuint() % 1000;
unsigned int bb = randuint() % 1000;
unsigned int cc = aa + bb;
char buf[256];
req[0].encode("+");
sprintf(buf, "%d %d %d", (aa / 100) % 10, (aa / 10) % 10, aa % 10);
//sprintf(buf, "%d %d", (aa / 10) % 10, aa % 10);
req[1].encode(buf);
sprintf(buf, "%d %d %d", (bb / 100) % 10, (bb / 10) % 10, bb % 10);
//sprintf(buf, "%d %d", (bb / 10) % 10, bb % 10);
req[2].encode(buf);
req[3].clear();
sprintf(buf, "%d %d %d", (cc / 100) % 10, (cc / 10) % 10, cc % 10);
//sprintf(buf, "%d %d", (cc / 10) % 10, cc % 10);
rsp[0].encode(buf);
rsp[1].clear();
rsp[2].clear();
rsp[3].clear();
#endif

      memcpy(confab.ctxbuf + mbi * 256 * tbn, (double *)req, sizeof(double) * 256 * tbn);
      memcpy(confab.tgtbuf + mbi * 256 * tbn, (double *)rsp, sizeof(double) * 256 * tbn);
    }
 
    confab.burn(0.002, 0.002);

    if (i % 1000 == 0) {
      confab.report("learnfab");
      confab.save();
    }

    ++i;
  }

#if 0
  Vocab &v = scr.vocab;

  std::string str;
  v.decode(req, &str);
  printf("%s\n", str.c_str());

  v.decode(rsp, &str);
  printf("%s\n", str.c_str());
#endif

  return 0;
}
