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

static void varsubst(vector<string> *words, unsigned int seed, multimap<string, string> *defines, map<string, string> *assign) {
  char buf[32];

  for (auto wi = words->begin(); wi != words->end(); ++wi) {
    std::string wstr = *wi;

    if (wstr[0] == '$') {
      if (assign) {
        auto kvi = assign->find(wstr);
        if (kvi != assign->end()) {
          *wi = kvi->second;
           continue;
        }
      }

      vector<string> vals;
      if (defines) {
        const char *p = wstr.c_str() + 1;
        const char *q = strchr(p, ':');
        if (!q)
          q = p + strlen(p);
        std::string key(p, q - p);

        auto r = defines->equal_range(key);
        for (auto ri = r.first; ri != r.second; ++ri)
          vals.push_back(ri->second);
      }

      if (vals.size()) {
        string val = vals[randuint() % vals.size()];
        if (assign)
          assign->insert(make_pair(wstr, val));
        *wi = val;
      } else {
        sprintf(buf, "[%08X]", seed);
        if (assign)
          assign->insert(make_pair(wstr, wstr + buf));
        *wi = wstr + buf;
      }
    }
  }
}

void Shibboleth::push(const char *word) {
  Hashbag wvec(word);

  ovec -= avec;
  avec += wvec;
  wvec *= (double)wn;
  ovec += wvec;
  ++wn;
}

void Shibboleth::unshift(const char *word) {
  Hashbag wvec(word);

  ovec += avec;
  avec += wvec;
  wvec *= (double)wn;
  ovec -= wvec;
  ++wn;
}
  
  
void Shibboleth::encode(const char *str, Vocab *vocab, unsigned int seed, multimap<string, string> *defines, map<string, string> *assign) {
  vector<string> strv;
  split(str, ' ', &strv);
  varsubst(&strv, seed, defines, assign);
  addstars(strv);

  if (vocab)
    vocab->add(join(strv, ' '));

//fprintf(stderr, "enc %s\n", join(strv, ' ').c_str());

  clear();
  for (auto i = strv.begin(); i != strv.end(); ++i)
    push(*i);

  ovec *= omul;
}

std::string Shibboleth::decode(const Vocab &vocab) {
  Hashbag tvec = avec;
  const Hashbag *uvecp = NULL;

  vector<string> out;

  unsigned int outn = 8;
  for (unsigned int outi = 0; outi < outn; ++outi) {
    const char *w = vocab.closest(tvec, &uvecp);
    if (!w)
      break;

    out.push_back(w);
    tvec -= *uvecp;
  }

  Hashbag pvec = ovec;
  pvec *= (1.0 / omul);

  struct cmp_t {
    const Hashbag *ovecp;

    cmp_t(const Hashbag *_ovecp) : ovecp(_ovecp) { }

    bool operator () (const std::string &x, const std::string &y) {
      Hashbag xvec(x.c_str()), yvec(y.c_str());

      Hashbag tvec = yvec;
      tvec -= xvec;
      Hashbag uvec = tvec;
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
