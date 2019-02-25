#define __MAKEMORE_WILDLETH_CC__

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <netinet/in.h>

#include <vector>
#include <string>

#include "wildleth.hh"
#include "hashbag.hh"
#include "strutils.hh"

namespace makemore {

extern double pairmul;

using namespace std;

void Wildleth::parse(const char *str) {
  vector<string> words;
  split(str, ' ', &words);
  parse(words);
}

void Wildleth::parse(const std::string &str) {
  parse(str.c_str());
}

static void randfix(const vector<string> &words, vector<string> *fixated) {
  for (auto word : words) {
    if (word == "?") {
      if (randbit()) {
        fixated->push_back("*");
      }
    } else if (word == "**") {
      unsigned int nstars = randuint() % 3;
      for (unsigned int i = 0; i < nstars; ++i)
        fixated->push_back("*");
    } else if (word == "***") {
      unsigned int nstars = randuint() % 4;
      for (unsigned int i = 0; i < nstars; ++i)
        fixated->push_back("*");
    } else {
      fixated->push_back(word);
    }
  }
}

void Wildleth::parse(const vector<string> &_words) {
  vector<string> words;
  randfix(_words, &words);

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

void Wildleth::mutate(Shibboleth *shib) {
  Hashbag negstar("*");
  negstar *= -1.0;

  if (Wildleth::Entry *wh = wild_head())
    shib->head = wh->tmp = Hashbag::random();
  if (Wildleth::Entry *wr = wild_rear())
    shib->rear = wr->tmp = Hashbag::random();

  for (unsigned int w = 0; w < map.size(); ++w) {
    Wildleth::Entry *wt = &map[w];
    if (wt->is_head || wt->is_rear)
      continue;
    wt->tmp = Hashbag::random();
    shib->torso.add(wt->tmp);
    shib->torso.add(negstar);
  }

//fprintf(stderr, "ps0=%lf\n", shib->pairs.size());
  unsigned int wmn = map.size();
  for (unsigned int wmi = 0; wmi < wmn; ++wmi) {
    const Wildleth::Entry *wme = &map[wmi];

    if (wme->consec_prev) {
      shib->pairs.add((wme->ctx + negstar) * negstar * pairmul);
    } else {
      shib->pairs.add(wme->ctx * negstar * pairmul);
    }
  }

  for (unsigned int wmi = 0; wmi < wmn; ++wmi) {
    const Wildleth::Entry *wme = &map[wmi];

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


void Wildleth::save(FILE *fp) const {
  size_t ret;
  unsigned int n = map.size();

  uint32_t hn = htonl(n);
  ret = fwrite(&hn, 1, 4, fp);
  assert(ret == 4);

  if (n > 0) {
    ret = fwrite(map.data(), sizeof(Entry), n, fp);
    assert(ret == n);
  }
}

void Wildleth::load(FILE *fp) {
  size_t ret;

  uint32_t hn;
  ret = fread(&hn, 1, 4, fp);
  assert(ret == 4);

  unsigned int n = ntohl(hn);
  map.resize(n);

  ret = fread(map.data(), sizeof(Entry), n, fp);
  assert(ret == n);
}

}
