#define __MAKEMORE_VOCAB_CC__ 1
#include <assert.h>
#include <stdlib.h>
#include <ctype.h>

#include "vocab.hh"
#include "closest.hh"

namespace makemore {  

using namespace std;

Vocab::Vocab() {
  n = 1;

  tags.resize(1);
  tags[0] = "";

  bags.resize(1);
  bags[0].clear();
}

void Vocab::add(const char *tag) {
  tags.resize(n + 1);
  tags[n] = string(tag);

  bags.resize(n + 1);
  bags[n].add(tag);

  ++n;
}

void Vocab::decode(const Tagbag &_tb, string *strp) {
  *strp = "";
  unsigned int nw = 16, iw;
  Tagbag tb = _tb;

  for (iw = 0; iw < nw; ++iw) {
    const double *x = tb.vec;
    const double *m = (const double *)bags.data();
    unsigned int k = 256;

    unsigned int i = closest(x, m, k, n);
    assert(i >= 0 && i < n);

    if (i == 0)
      break;

    const char *w = tags[i].c_str();
    if (iw > 0)
      *strp += " ";
    *strp += w;

    tb.sub(bags[i]);
    tb.mul((double)(iw + 2) / (double)(iw + 1));
  }

  if (iw == nw)
    *strp += " ...";
}

static void split(const char *str, vector<string> *words) {
  words->clear();

  const char *p = str;

  while (const char *q = strchr(p, ' ')) {
    words->push_back(string(p, q - p));

    p = q + 1;
    while (*p == ' ')
      p++;
  }

  if (*p)
    words->push_back(string(p));
}

void Vocab::encode(const char *str, Tagbag *tbp) {
  vector<string> strv;
  split(str, &strv);

  unsigned int nw = strv.size(), iw;
  for (iw = 0; iw < nw; ++iw) {
    Tagbag tw(strv[iw].c_str());
    tw.mul(1.0 / (1.0 + (double)iw));
    tbp->add(tw);
  }
}
  
}
