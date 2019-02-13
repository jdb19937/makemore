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


std::string ask(Confab &confab, Vocab &vocab, std::string reqstr, int &iters, int depth = 0) {

  if (iters > 256)
    return "...";
  if (depth > 64)
    return "...";
  ++iters;

  Shibboleth req, rsp;

fprintf(stderr, "ask reqstr=%s depth=%d\n", reqstr.c_str(), depth);


  std::string rspstr;
  req.encode(reqstr, &vocab);

  memcpy(confab.ctxbuf, (double *)&req, sizeof(double) * 512);

  confab.scramble(0, 0);
  confab.generate();

  memcpy((double *)&rsp, confab.outbuf, sizeof(double) * 512);

  rspstr = rsp.decode(vocab);
fprintf(stderr, "rspstr=%s\n", rspstr.c_str());

  {
    int nexec = 0;
    std::string strexec;

    std::vector<std::string> partwords;
    split(rspstr.c_str(), ' ', &partwords);
    rspstr = "";

    for (int j = 0; j < partwords.size(); ++j) {
      if (!strncmp(partwords[j].c_str(), "!", 1) && partwords[j] != "!!") {
        nexec = atoi(partwords[j].c_str() + 1);
        if (nexec < 0)
          nexec = 0;
        continue;
      }

      if (nexec == 0) {
        if (rspstr.length()) rspstr += " ";
        rspstr += partwords[j];
        continue;
      }

      if (strexec.length()) strexec += " ";
      strexec += partwords[j];
      --nexec;

      if (nexec == 0) {
        std::string ans = ask(confab, vocab, strexec, iters, depth + 1);
        if (rspstr.length()) rspstr += " ";
        rspstr += ans;
      }
    }


fprintf(stderr, "rspstr1=%s\n", rspstr.c_str());
    split(rspstr.c_str(), ' ', &partwords);
    rspstr = "";

    bool spell = 0, join = 0;
    for (auto j = partwords.begin(); j != partwords.end(); ++j) {
      std::string rw = *j;

      if (rw == "@") {
        spell = 1;
        continue;
      }
      if (rw == "%") {
        if (rspstr.length())
          rspstr += " ";
        join = 1;
        continue;
      }

      if (join) {
        rspstr += rw;
      } else if (spell) {
        std::string letters;

        for (auto k = 0; k < rw.length(); ++k) {
          if (k)
            letters += " ";
          letters += rw[k];
        }

        if (rspstr.length())
          rspstr += " ";
        rspstr += letters;
      } else {
        if (rspstr.length())
          rspstr += " ";
        rspstr += rw;
      }

      spell = 0;
    }
  }
fprintf(stderr, "rspstr2=%s\n", rspstr.c_str());

  vocab.add(rspstr);

  if (rspstr.length() >= 4 && !memcmp(rspstr.c_str(), "!! ", 3))
    rspstr = ask(confab, vocab, std::string(rspstr.c_str() + 3), iters, depth + 1);


// fprintf(stderr, "rsp=%s\n", rspstr.c_str());
  return rspstr;
}

int main() {
  setbuf(stdout, NULL);
  seedrand();

  Vocab vocab;
  Confab confab("test.confab", 1);
  Script scr("script.txt", &vocab);

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
    std::string rspstr = ask(confab, vocab, reqstr, iters);

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
