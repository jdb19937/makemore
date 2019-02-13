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
    for (wvi = wv.begin(); wvi != wv.end() && strncmp(wvi->c_str(), "->", 2); ++wvi) {
      if (vocab && (*wvi)[0] != '$')
        vocab->add(*wvi);
      req += *wvi;
      req += " ";
    }
    if (req.length())
      req.erase(req.length() - 1);
    assert(wvi != wv.end());

    unsigned int copies = 1;
    if (!strncmp(wvi->c_str(), "->x", 3))
      copies = atoi(wvi->c_str() + 3);
    if (copies > 32)
      copies = 32;

    ++wvi;
    for (; wvi != wv.end(); ++wvi) {
      if (vocab && (*wvi)[0] != '$')
        vocab->add(*wvi);
      rsp += *wvi;
      rsp += " ";
    }
    if (rsp.length())
      rsp.erase(rsp.length() - 1);

    for (unsigned int copy = 0; copy < copies; ++copy)
      templates.push_back(make_pair(req, rsp));
  }

  fclose(fp);
  fp = NULL;
}

Script::~Script() {
}

void Script::pick(Shibboleth *req, Shibboleth *rsp) {
  assert(templates.size());
  unsigned int i = randuint() % templates.size();
  unsigned int seed = randuint();

  const Template &tpl = templates[i];
  req->encode(tpl.first, NULL, seed);
  rsp->encode(tpl.second, NULL, seed);
}

}
