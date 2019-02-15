#define __MAKEMORE_SHIBBOLETH_CC__ 1
#include <vector>
#include <string>
#include <algorithm>

#include "shibboleth.hh"
#include "strutils.hh"
#include "closest.hh"
#include "vocab.hh"
#include "wildmap.hh"

#include "sha256.c"

namespace makemore {
double pairmul = 0.1;

using namespace std;

static void addstars(vector<string> &words) {
  set<string> seen;

  for (auto wi = words.begin(); wi != words.end(); ++wi) {
    if (*wi->c_str() == '*')
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

static void varsubst(vector<string> *words, multimap<string, string> *defines) {
  char buf[32];

  for (auto wi = words->begin(); wi != words->end(); ++wi) {
    std::string wstr = *wi;

    if (wstr == "*") {
      sprintf(buf, "[%08X]", randuint());
      *wi = buf;
    } else if (defines) {
      vector<string> vals;
      std::string key(wstr);

      auto r = defines->equal_range(wstr);
      for (auto ri = r.first; ri != r.second; ++ri)
        vals.push_back(ri->second);

      if (vals.size()) {
        string val = vals[randuint() % vals.size()];
        *wi = val;
      }
    }
  }
}

void Shibboleth::append(const Hashbag &bag) {
  unsigned int n = lround(size()); 

  if (n == 0) {
    head = bag;
    return;
  } else if (n == 1) {
    rear = bag;
    pairs.add(head * rear * pairmul);
    return;
  }

  Hashbag prev = rear;
  torso.add(rear);
  rear = bag;

  prev *= bag;
  pairs.add(prev * pairmul);
}

void Shibboleth::append(const char *word) {
  append(Hashbag(word));
}

void Shibboleth::append(const Shibboleth &shib) {
  unsigned int n = lround(size()); 
  unsigned int shibn = lround(shib.size());

  if (shibn == 0)
    return;

  if (n == 0) {
    copy(shib);
    return;
  } else if (n == 1) {
    if (shibn == 1) {
      rear = shib.head;
      pairs.add(head * rear * pairmul);
      return;
    }

    Hashbag tmp = head;
    copy(shib);
    torso.add(head);
    pairs.add(head * tmp * pairmul);
    head = tmp;
    return;
  }

  torso.add(rear);
  torso.add(shib.head);
  torso.add(shib.torso);
  pairs.add(rear * shib.head * pairmul);
  rear = shib.rear;
}

void Shibboleth::encode(const char *str) {
  vector<string> words;
  split(str, ' ', &words);
//  addstars(words);

//fprintf(stderr, "enc %s\n", join(words, ' ').c_str());

  clear();

  unsigned int wn = words.size();
  if (wn == 0) {
    return;
  }
  head.add(words[0].c_str());

  if (wn == 1)
    return;
  rear.add(words[wn - 1].c_str());
  
  for (unsigned int wi = 1; wi < wn - 1; ++wi) {
    torso.add(words[wi].c_str());
  }

  char buf[1024];
  for (unsigned int wi = 0, wj = wi + 1; wj < wn; ++wi, ++wj) {
    const char *wa = words[wi].c_str();
    const char *wb = words[wj].c_str();

    Hashbag pairbag(wa);
    pairbag *= Hashbag(wb);

    pairs.add(pairbag);
  }

//fprintf(stderr, "pairsize=%lf\n", pairs.size());
  pairs *= pairmul;

}

std::string Shibboleth::decode(const Vocab &vocab) {
  Hashbag tvec = torso;
  const Hashbag *uvecp = NULL;

  const char *headword = vocab.closest(head, NULL);
  if (!headword)
    return "";

  const char *rearword = vocab.closest(rear, NULL);
  if (!rearword)
    return headword;

  map<string, unsigned int> whb;
  for (unsigned int outi = 0; outi < 16; ++outi) {
    const char *w = vocab.closest(tvec, &uvecp);
    if (!w)
      break;
    ++whb[w];
    tvec -= *uvecp;
  }

  vector<string> front, back;

  Hashbag tpairs = pairs;
  tpairs *= (1.0 / pairmul);
  std::string prevword0 = headword;
  std::string prevword1 = rearword;

  while (whb.begin() != whb.end()) {
    double bestd = -1;
    int bestdir = -1;
    std::string bestnextword;
    Hashbag bestpairbag;

    for (auto whi = whb.begin(); whi != whb.end(); ++whi) {
      std::string nextword = whi->first;

      Hashbag pairbag0(prevword0.c_str());
      pairbag0 *= Hashbag(nextword.c_str());
      double d0 = (pairbag0 - tpairs).abs();
      if (bestd < 0 || d0 < bestd) {
        bestnextword = nextword;
        bestpairbag = pairbag0;
        bestdir = 0;
        bestd = d0;
      }
     
#if 1
      Hashbag pairbag1(prevword1.c_str());
      pairbag1 *= Hashbag(nextword.c_str());
      double d1 = (pairbag1 - tpairs).abs();
      if (bestd < 0 || d1 < bestd) {
        bestnextword = nextword;
        bestpairbag = pairbag1;
        bestdir = 1;
        bestd = d1;
      }
#endif
    }

    --whb[bestnextword];
    if (whb[bestnextword] == 0)
      whb.erase(bestnextword);

    tpairs -= bestpairbag;

    if (bestdir == 0) {
      front.push_back(bestnextword);
      prevword0 = bestnextword;
    } else if (bestdir == 1) {
      back.push_back(bestnextword);
      prevword1 = bestnextword;
    }
  }

  std::vector<std::string> out;
  out.push_back(headword);
  for (auto i = 0; i < front.size(); ++i)
    out.push_back(front[i]);
  for (int i = back.size() - 1; i >= 0; --i)
    out.push_back(back[i]);
  out.push_back(rearword);

//  delstars(out);
  return join(out, ' ');
}

}
