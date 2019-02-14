#include <stdio.h>

#include "vocab.hh"
#include "shibboleth.hh"
#include "script.hh"
#include "confab.hh"
#include "strutils.hh"

using namespace makemore;

  unsigned int tbn = 4;

std::string normpart(const std::string &x) {
  std::vector<std::string> xp;
  std::string y;
  split(x.c_str(), ' ', &xp);
  for (unsigned int i = 0; i < xp.size(); ++i) {
    if (i > 0)
      y += " ";
    y += xp[i];
  }
  return y;
}


std::string ask(Confab &confab, Vocab &vocab, std::string reqstr) {
  Shibboleth req, rsp;

  std::string rspstr;
  req.encode(reqstr.c_str(), &vocab);

  memcpy(confab.ctxbuf, (double *)&req, sizeof(double) * 512);

  confab.scramble(0, 0);
  confab.generate();

  memcpy((double *)&rsp, confab.outbuf, sizeof(double) * 512);

  rspstr = rsp.decode(vocab);
  return rspstr;
}

int main() {
  setbuf(stdout, NULL);
  seedrand();

  Vocab vocab;
  Confab confab("test.confab", 1);
  {
    Script scr("script.txt", &vocab);
  }

  char buf[4096];

  while (1) {
    *buf = 0;
    char *unused = fgets(buf, 4095, stdin);
    buf[4095] = 0;
    if (char *p = strchr(buf, '\n'))
      *p = 0;
    else
      break;

    confab.load();

    int iters = 0;
    std::string reqstr = buf;
    std::string rspstr = ask(confab, vocab, reqstr);

    printf("%s\n", rspstr.c_str());
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
