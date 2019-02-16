#include <stdio.h>

#include "vocab.hh"
#include "shibboleth.hh"
#include "script.hh"
#include "confab.hh"
#include "strutils.hh"
#include "brane.hh"

using namespace makemore;

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
  vocab.add(reqstr.c_str());
  req.encode(reqstr.c_str());

  memcpy(confab.ctxbuf, (double *)&req, sizeof(Shibboleth));

  confab.scramble(0, 0);
  confab.generate();

  memcpy((double *)&rsp, confab.outbuf, sizeof(Shibboleth));

  rspstr = rsp.decode(vocab);
  return rspstr;
}

static void parsereqmem(const char *in, std::string *req, std::string *mem) {
  if (const char *p = strchr(in, '(')) {
    ++p;

    const char *q = strchr(p, ')');
    if (!q)
      q = p + strlen(p);

    *mem = std::string(p, q - p);
    *req = std::string(in, p - in - 1);
  } else {
    *req = in;
    *mem = "";
  }
}

int main() {
  setbuf(stdout, NULL);
  seedrand();

  Confab confab("test.confab", 1);
  Brane brane(&confab);

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
    confab.vocab.add(buf);

    std::string reqstr, memstr;
    parsereqmem(buf, &reqstr, &memstr);
    Shibboleth reqshib;
    reqshib.encode(reqstr.c_str());
    Shibboleth memshib;
    memshib.encode(memstr.c_str());
    Shibboleth rspshib = brane.ask(reqshib, &memshib);

    std::string rspstr = rspshib.decode(confab.vocab);
    printf("%s\n", rspstr.c_str());
  }

  return 0;
}
