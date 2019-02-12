#define __MAKEMORE_SCRIPT_CC__ 1

#include <assert.h>

#include "script.hh"
#include "strutils.hh"

namespace makemore {
using namespace std;

Script::Script(const char *_fn, Vocab *vocab) {
  fn = _fn;
  assert(fp = fopen(fn.c_str(), "r"));

  char buf[4096];
  while (1) {
    *buf = 0;
    char *unused = fgets(buf, sizeof(buf) - 1, fp);
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

    vector<string> wv;
    split(q, ' ', &wv);

    std::string req, rsp;
    vector<string>::iterator wvi;
    for (wvi = wv.begin(); wvi != wv.end() && *wvi != "->"; ++wvi) {
      if (vocab && (*wvi)[0] != '$')
        vocab->add(*wvi);
      req += *wvi;
      req += " ";
    }
    if (req.length())
      req.erase(req.length() - 1);
    assert(wvi != wv.end());

    ++wvi;
    for (; wvi != wv.end(); ++wvi) {
      if (vocab && (*wvi)[0] != '$')
        vocab->add(*wvi);
      rsp += *wvi;
      rsp += " ";
    }
    if (rsp.length())
      rsp.erase(rsp.length() - 1);

    templates.push_back(make_pair(req, rsp));
  }

  fclose(fp);
  fp = NULL;
}

Script::~Script() {
}

void Script::pick(Tagbag *req, Tagbag *rsp, unsigned int ntb) {
  assert(templates.size());
  unsigned int i = randuint() % templates.size();

  unsigned int seed = randuint();

  const Template &tpl = templates[i];

//fprintf(stderr, "%s -> %s\n", tpl.first.c_str(), tpl.second.c_str());

  std::string reqstr = tpl.first;
  std::vector<std::string> reqparts;
  split(reqstr.c_str(), ',', &reqparts);
  for (unsigned int i = 0; i < ntb; ++i) {
    if (i < reqparts.size())
      req[i].encode(reqparts[i], seed);
    else
      req[i].clear();
  }

  std::string rspstr = tpl.second;
  std::vector<std::string> rspparts;
  split(rspstr.c_str(), ',', &rspparts);
  for (unsigned int i = 0; i < ntb; ++i) {
    if (i < rspparts.size())
      rsp[i].encode(rspparts[i], seed);
    else
      rsp[i].clear();
  }
}

}
