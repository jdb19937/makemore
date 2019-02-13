#define __MAKEMORE_SHIBBOLETH_CC__ 1
#include <vector>
#include <string>
#include <algorithm>

#include "shibboleth.hh"
#include "strutils.hh"
#include "closest.hh"
#include "vocab.hh"

#include "sha256.c"

namespace makemore {
using namespace std;

static void vecadd(double *x, const double *y) {
  for (unsigned int i = 0; i < 256; ++i)
    x[i] += y[i];
}

static void vecmul(double *x, double m) {
  for (unsigned int i = 0; i < 256; ++i)
    x[i] *= m;
}

static void vecsub(double *x, const double *y) {
  for (unsigned int i = 0; i < 256; ++i)
    x[i] -= y[i];
}

static void addstars(vector<string> &words) {
  set<string> seen;

  for (auto wi = words.begin(); wi != words.end(); ++wi) {
    if (*wi->c_str() == '$' || *wi->c_str() == '*')
      continue;
    while (seen.count(*wi))
      *wi = std::string("*") + *wi;
    seen.insert(*wi);
  }
}

static void delstars(vector<string> &words) {
  for (auto wi = words.begin(); wi != words.end(); ++wi) {
    const char *p = wi->c_str();
    while (*p == '*' && p[1])
      p++;
    *wi = p;
  }
}

static void varsubst(vector<string> *words, unsigned int seed) {
  char buf[32];

  for (auto wi = words->begin(); wi != words->end(); ++wi) {
    const char *w = wi->c_str();
    if (*w == '$') {
      sprintf(buf, "[%08X]", seed);
      *wi += buf;
    }
  }
}

void Shibboleth::push(const char *word) {
  Wordvec wvec(word);

  ovec -= avec;
  avec += wvec;
  wvec *= (double)wn;
  ovec += wvec;
  ++wn;
}

void Shibboleth::unshift(const char *word) {
  Wordvec wvec(word);

  ovec += avec;
  avec += wvec;
  wvec *= (double)wn;
  ovec -= wvec;
  ++wn;
}
  
  
void Shibboleth::encode(const char *str, Vocab *vocab, unsigned int seed) {
  vector<string> strv;
  split(str, ' ', &strv);
  addstars(strv);
  varsubst(&strv, seed);

  if (vocab)
    vocab->add(join(strv, ' '));

//fprintf(stderr, "enc %s\n", join(strv, ' ').c_str());

  clear();
  for (auto i = strv.begin(); i != strv.end(); ++i)
    push(*i);

  ovec *= omul;
}

std::string Shibboleth::decode(const Vocab &vocab) {
  Wordvec tvec = avec;
  const Wordvec *uvecp = NULL;

  vector<string> out;

  unsigned int outn = 8;
  for (unsigned int outi = 0; outi < outn; ++outi) {
    const char *w = vocab.closest(tvec, &uvecp);
    if (!w)
      break;

    out.push_back(w);
    tvec -= *uvecp;
  }

  Wordvec pvec = ovec;
  pvec *= (1.0 / omul);

  struct cmp_t {
    const Wordvec *ovecp;

    cmp_t(const Wordvec *_ovecp) : ovecp(_ovecp) { }

    bool operator () (const std::string &x, const std::string &y) {
      Wordvec xvec(x.c_str()), yvec(y.c_str());

      Wordvec tvec = yvec;
      tvec -= xvec;
      Wordvec uvec = tvec;
      uvec.mul(-1);

      tvec -= *ovecp;
      uvec -= *ovecp;

      return (tvec.abs() < uvec.abs());
    }
  } cmp(&pvec);

  std::sort(out.begin(), out.end(), cmp);

  delstars(out);

  return join(out, ' ');
}

}
