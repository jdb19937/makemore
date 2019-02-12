#include <stdio.h>

#include "vocab.hh"
#include "tagbag.hh"
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

  if (iters > 128)
    return "...";
  if (depth > 32)
    return "...";
  ++iters;

  Tagbag *req = new Tagbag[tbn];
  Tagbag *rsp = new Tagbag[tbn];

//fprintf(stderr, "ask reqstr=%s depth=%d\n", reqstr.c_str(), depth);

  std::string rspstr;

  vocab.add(reqstr);

  std::vector<std::string> reqparts;
  split(reqstr.c_str(), ',', &reqparts);

  std::vector<std::string> newreqparts;

  for (unsigned int i = 0; i < tbn; ++i) {
    if (i >= reqparts.size())
      break;

    std::string reqpart = normpart( reqparts[i] );
//fprintf(stderr, "depth=%d reqpart=%s\n", depth, reqpart.c_str());

    if (!strncmp(reqpart.c_str(), "! ", 2)) {
      std::string partreqstr = "";

      std::vector<std::string> partwords;
      split(reqpart.c_str() + 2, ' ', &partwords);
      for (unsigned int j = 0; j < 4 && j < partwords.size(); ++j) {
        if (j > 0)
          partreqstr += ", ";
        partreqstr += partwords[j];
      }

      reqpart = ask(confab, vocab, partreqstr, iters, depth + 1);
    }

    std::vector<std::string> tmp;
    split(reqpart.c_str(), ',', &tmp);
    for (unsigned int k = 0; k < tmp.size(); ++k)
      newreqparts.push_back(normpart( tmp[k]));
  }

  for (unsigned int i = 0; i < tbn; ++i) {
    if (i >= newreqparts.size()) {
      req[i].clear();
      continue;
    }
// fprintf(stderr, "encoding reqpart=%s\n", newreqparts[i].c_str());
    req[i].encode(newreqparts[i]);
  }



  memcpy(confab.ctxbuf, (double *)req, sizeof(double) * 256 * tbn);

  confab.scramble(0, 0);
  confab.generate();

  memcpy((double *)rsp, confab.outbuf, sizeof(double) * 256 * tbn);

  for (unsigned int i = 0; i < tbn; ++i) {
    std::string rsppart;
    vocab.decode(rsp[i], &rsppart);
    if (!rsppart.length())
      break;

// fprintf(stderr, "rsppart=%s\n", rsppart.c_str());

    bool spell = 0, join = 0;
    std::vector<std::string> rsppartwords;
    split(rsppart.c_str(), ' ', &rsppartwords);
    std::string newrsppart;
    for (auto j = rsppartwords.begin(); j != rsppartwords.end(); ++j) {
      std::string rw = *j;

      if (rw == "@") {
        join = 0;
        spell = 1;
        continue;
      }

      if (rw == "%") {
        join = 1;
        spell = 0;
        continue;
      }

      if (spell) {
        std::string letters;
        for (auto k = 0; k < rw.length(); ++k) {
          if (k)
            letters += " ";
          letters += rw[k];
        }
        if (!join && newrsppart.length())
          newrsppart += " ";
        newrsppart += letters;
      } else {
        if (!join && newrsppart.length())
          newrsppart += " ";
        newrsppart += rw;
      }

      spell = 0;
    }
    rsppart = newrsppart;
    if (!rsppart.length())
      break;

    if (rspstr.length())
      rspstr += ", ";
    rspstr += rsppart;
  }

  if (rspstr.length() >= 4 && !memcmp(rspstr.c_str(), "!! ", 3))
    rspstr = ask(confab, vocab, std::string(rspstr.c_str() + 3), iters, depth + 1);


  delete[] req;
  delete[] rsp;

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
