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
    vector<string>::iterator sepi;

    for (sepi = wv.begin(); sepi != wv.end(); ++sepi) {
      if (!strncmp(sepi->c_str(), "->", 2))
        break;
      if (*sepi == "-:")
        break;
    }

    if (sepi == wv.end()) {
      fprintf(stderr, "malformed rule case 1: %s\n", q);
      continue;
    }

    vector<string>::iterator wvi;
    if (*sepi == "-:") {
      wvi = wv.begin();
      const std::string &key = *wvi;

      ++wvi;
      if (sepi != wvi) {
        fprintf(stderr, "malformed rule case 2: %s\n", q);
        continue;
      }

      ++wvi;
      if (wvi == wv.end()) {
        fprintf(stderr, "malformed rule case 3: %s\n", q);
        continue;
      }

      for (; wvi != wv.end(); ++wvi) {
        const std::string &val = *wvi;
//fprintf(stderr, "define [%s -> %s]\n", key.c_str(), val.c_str());
        defines.insert(make_pair(key, val));
      }
    } else {
      std::string req, rsp;
      for (wvi = wv.begin(); wvi != sepi; ++wvi) {
        if (vocab && (*wvi)[0] != '$')
          vocab->add(*wvi);
        if (req.length())
          req += " ";
        req += *wvi;
      }
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
        if (rsp.length())
          rsp += " ";
        rsp += *wvi;
      }
//fprintf(stderr, "template [%s -> %s]\n", req.c_str(), rsp.c_str());
  
      for (unsigned int copy = 0; copy < copies; ++copy)
        templates.push_back(make_pair(req, rsp));
    }
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

  map<string, string> assign;
  req->encode(tpl.first.c_str(), NULL, seed, &defines, &assign);
  rsp->encode(tpl.second.c_str(), NULL, seed, &defines, &assign);
}

}
