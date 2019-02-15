#define __MAKEMORE_WILDMAP_CC__
#include "wildmap.hh"
#include "hashbag.hh"
#include "strutils.hh"

#include <vector>
#include <string>

namespace makemore {

extern double pairmul;

using namespace std;

void Wildmap::parse(const char *str) {
  vector<string> words;
  split(str, ' ', &words);
  parse(words);
}

void Wildmap::parse(const std::string &str) {
  parse(str.c_str());
}

void Wildmap::parse(const vector<string> &words) {
  unsigned int wn = words.size();
  unsigned int prevwi = 0;

  map.clear();
  for (unsigned int wi = 0; wi < wn; ++wi) {
    if (words[wi] != "*")
      continue;

    Entry ent;
    ent.consec_prev = false;
    ent.consec_next = false;
    ent.is_head = (wi == 0);
    ent.is_rear = (wi == wn - 1 && wn >= 2);

    if (map.size()) {
      if (prevwi + 1 == wi) {
        Entry &prev = *map.rbegin();
        prev.consec_next = true;
        ent.consec_prev = true;
      }
    }

    ent.ctx.clear();
    if (wi > 0)
      ent.ctx += Hashbag(words[wi - 1].c_str());
    if (wi < wn - 1)
      ent.ctx += Hashbag(words[wi + 1].c_str());

    map.push_back(ent);
    prevwi = wi;
  }
}

void Wildmap::mutate(Shibboleth *shib) {
  Hashbag negstar("*");
  negstar *= -1.0;

  if (Wildmap::Entry *wh = wild_head())
    shib->head = wh->tmp = Hashbag::random();
  if (Wildmap::Entry *wr = wild_rear())
    shib->rear = wr->tmp = Hashbag::random();

  for (unsigned int w = 0; w < map.size(); ++w) {
    Wildmap::Entry *wt = &map[w];
    if (wt->is_head || wt->is_rear)
      continue;
    wt->tmp = Hashbag::random();
    shib->torso.add(wt->tmp);
    shib->torso.add(negstar);
  }

//fprintf(stderr, "ps0=%lf\n", shib->pairs.size());
  unsigned int wmn = map.size();
  for (unsigned int wmi = 0; wmi < wmn; ++wmi) {
    const Wildmap::Entry *wme = &map[wmi];

    if (wme->consec_prev) {
      shib->pairs.add((wme->ctx + negstar) * negstar * pairmul);
    } else {
      shib->pairs.add(wme->ctx * negstar * pairmul);
    }
  }

  for (unsigned int wmi = 0; wmi < wmn; ++wmi) {
    const Wildmap::Entry *wme = &map[wmi];

    Hashbag tmp = wme->ctx;
    if (wme->consec_prev) {
      assert(wmi > 0);
      tmp += negstar;
    }

    if (wme->consec_next) {
      assert(wmi < wmn - 1);
      tmp += map[wmi + 1].tmp;
      tmp += negstar;
    }

    tmp *= wme->tmp;
    tmp *= pairmul;

    shib->pairs.add(tmp);
  }

#if 0
//fprintf(stderr, "pst=%lf\n", shib->pairs.size());

    Hashbag tmp = wme->ctx;
    if (wme->consec_prev) {
      assert(wmi > 0);
      tmp += map[wmi - 1].tmp;
      tmp += negstar;
    }
    if (wme->consec_next) {
      assert(wmi < wmn - 1);
      tmp += map[wmi + 1].tmp;
      tmp += negstar;
    }

    tmp *= wme->tmp;
    tmp *= pairmul;
    shib->pairs.add(tmp);
  }
#endif

//fprintf(stderr, "ps1=%lf\n", shib->pairs.size());
}

}