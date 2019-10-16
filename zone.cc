#define __MAKEMORE_ZONE_CC__ 1
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/mman.h>

#include <set>
#include <list>

#include "tmutils.hh"
#include "random.hh"
#include "zone.hh"

namespace makemore {

using namespace std;

void Zone::fill_fam(const char *nom, Parson::Nom *fam) {
  set<string> seen;

  Parson *p = find(nom);
  assert(p);

  seen.insert(nom);
  seen.insert(p->parens[0]);
  seen.insert(p->parens[1]);

  memset(fam, 0, sizeof(Parson::Nom) * nfam);
  unsigned int fami = 0;

  Parson *pp0 = find(p->parens[0]);
  Parson *pp1 = find(p->parens[1]);
  if (pp0 == pp1)
    pp1 = NULL;

  for (unsigned int i = 0; i < Parson::nfrens && fami < nfam; ++i) {
    const char *pfnom = p->frens[i];
    if (seen.count(pfnom))
      continue;
    Parson *pf = find(p->frens[i]);
    if (!pf)
      continue;
    if (p->fraternal(pf) || !strcmp(pf->parens[0], nom) || !strcmp(pf->parens[1], nom)) {
      strcpy(fam[fami++], pf->nom);
      seen.insert(pf->nom);
    }
  }

  if (pp0) {
    for (unsigned int i = 0; i < Parson::nfrens && fami < nfam; ++i) {
      const char *pfnom = pp0->frens[i];
      if (seen.count(pfnom))
        continue;
      Parson *pf = find(pfnom);
      if (!pf)
        continue;
      if (p->fraternal(pf)) {
        strcpy(fam[fami++], pf->nom);
        seen.insert(pf->nom);
      }
    }
  }

  if (pp1) {
    for (unsigned int i = 0; i < Parson::nfrens && fami < nfam; ++i) {
      const char *pfnom = pp1->frens[i];
      if (seen.count(pfnom))
        continue;
      Parson *pf = find(pfnom);
      if (!pf)
        continue;
      if (p->fraternal(pf)) {
        strcpy(fam[fami++], pf->nom); 
        seen.insert(pf->nom);
      }
    }
  }
}

void Zone::create(const char *_fn, unsigned int n) {
  assert(strlen(_fn) < 4000);
  string fn = _fn;

  int fd = ::open(fn.c_str(), O_RDWR | O_CREAT | O_EXCL, 0777);
  assert(fd != -1);

  int ret;
  ret = ::ftruncate(fd, n * sizeof(Parson));
  assert(ret == 0);

  ret = ::close(fd);
  assert(ret == 0);

  Zone *pdb = new Zone(fn.c_str());
  assert(pdb->n == n);
  memset((uint8_t *)pdb->db, 0, n * sizeof(Parson));
  delete pdb;
}

Zone::Zone(const std::string &_fn) {
  assert(_fn.length() < 4000);
  fn = _fn;

  fd = ::open(fn.c_str(), O_RDWR | O_CREAT, 0777);
  assert(fd != -1);

  off_t size = ::lseek(fd, 0, SEEK_END);
  assert(size > 0);
  assert(size % sizeof(Parson) == 0);
  n = size / sizeof(Parson);

  assert(0 == ::lseek(fd, 0, SEEK_SET));

  size_t map_size = (size + 4095) & ~4095;
  void *map = ::mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  assert(map != NULL);
  assert(map != MAP_FAILED);

  db = (Parson *)map;

  ac = new Autocompleter;
  for (unsigned int i = 0; i < n; ++i) {
    Parson *prs = db + i;
    if (!prs->revised)
      continue;
    ac->add(prs->nom);
  }

  actup();
  scrup();
  onlup();
}

void Zone::actup() {
  act_nom.clear();
  for (unsigned int i = 0; i < n; ++i) {
    Parson *prs = db + i;
    if (!prs->revised)
      continue;
    if (prs->flags & Parson::FLAG_HIDETOP)
      continue;
    act_nom.insert(std::pair<double,std::string>(prs->activity(), prs->nom));
  }
}

void Zone::onlup() {
  onl_nom.clear();
  for (unsigned int i = 0; i < n; ++i) {
    Parson *prs = db + i;
    if (!prs->acted)
      continue;
    if (!*prs->owner)
      continue;
    onl_nom.insert(std::pair<double,std::string>(prs->acted, prs->nom));
  }
}

void Zone::scrup() {
  scr_nom.clear();
  for (unsigned int i = 0; i < n; ++i) {
    Parson *prs = db + i;
    if (!prs->revised)
      continue;
    if (prs->flags & Parson::FLAG_HIDETOP)
      continue;
    scr_nom.insert(std::make_pair(prs->score, std::string(prs->nom)));
  }
}

Zone::~Zone() {
  size_t map_size = ((n * sizeof(Parson)) + 4095) & ~4095;
  ::munmap((void *)db, map_size);
  ::close(fd);
}

Parson *Zone::pick() {
  return pick(64);
}

Parson *Zone::pick(unsigned int max_tries) {
  for (unsigned int tries = 0; tries < max_tries; ++tries) {
    Parson *cand = db + randuint() % n;
    if (cand->created)
      return cand;
  }

  return find("synthetic_dan_brumleve");
}


Parson *Zone::pick(const char *tag, unsigned int max_tries) {
  for (unsigned int tries = 0; tries < max_tries; ++tries) {
    Parson *cand = db + randuint() % n;
    if (cand->created && cand->has_tag(tag)) {
      return cand;
    }
  }

  return NULL;
}

Parson *Zone::pick(const char *tag1, const char *tag2, unsigned int max_tries) {
  for (unsigned int tries = 0; tries < max_tries; ++tries) {
    Parson *cand = db + randuint() % n;
    if (cand->created && cand->has_tag(tag1) && cand->has_tag(tag2)) {
      return cand;
    }
  }

  return find("synthetic_dan_brumleve");
}

Parson *Zone::find(const std::string &nom) const {
  if (!Parson::valid_nom(nom.c_str()))
    return NULL;

  for (unsigned int j = 0; j < nvariants; ++j) {
    Parson *cand = db + Parson::hash_nom(nom.c_str(), j) % n;
    if (nom == cand->nom)
      return cand;
  }

  return NULL;
}

Parson *Zone::left_naybor(Parson *p, unsigned int max_tries) {
  unsigned long i = p - db;
  if (i >= n)
    return NULL;

  for (unsigned int tries = 0; tries < max_tries; ++tries) {
    if (i == 0)
      i = n;
    --i;

    Parson *q = db + i;
    if (*q->nom && q->created)
      return q;
  }
  return NULL;
}

Parson *Zone::right_naybor(Parson *p, unsigned int max_tries) {
  unsigned long i = p - db;
  if (i >= n)
    return NULL;

  for (unsigned int tries = 0; tries < max_tries; ++tries) {
    ++i;
    if (i == n)
      i = 0;

    Parson *q = db + i;
    if (*q->nom && q->created)
      return q;
  }
  return NULL;
}

Parson *Zone::make(const Parson &x, bool *evicted, Parson *evictee) {
  if (evicted)
    *evicted = false;
  assert(Parson::valid_nom(x.nom));

  multimap<double, Parson*> act_cand;

  for (unsigned int j = 0; j < nvariants; ++j) {
    Parson *cand = db + Parson::hash_nom(x.nom, j) % n;

    if (!strcmp(cand->nom, x.nom)) {
      memcpy(cand, &x, sizeof(Parson));
      return cand;
    }

    double act = cand->activity();
    act_cand.insert(make_pair(act, cand));
  }

  auto i = act_cand.begin(); 
  Parson *p = i->second;

  while (i != act_cand.end()) {
    if (!i->second->created) {
      p = i->second;
      break;
    }
    ++i;
  }

  if (p->created) {
    if (evicted)
      *evicted = true;
    if (evictee)
      memcpy(evictee, p, sizeof(Parson));
  }

  memcpy(p, &x, sizeof(Parson));

  const char *dan_nom = "synthetic_dan_brumleve";
  p->add_fren(dan_nom);
  if (Parson *dan = find(dan_nom)) {
    dan->add_fren(p->nom);
  }

  if (Parson *q = left_naybor(p)) {
    p->add_fren(q->nom);
    q->add_fren(p->nom);
  }

  if (Parson *q = right_naybor(p)) {
    p->add_fren(q->nom);
    q->add_fren(p->nom);
  }

  return p;
}

void Zone::scan_kids(const std::string &nom, std::list<Parson*> *kids, int m) {
  int k = 0;
  for (unsigned int i = 0; i < n; ++i) {
    Parson *prs = db + i;
    if (nom == prs->parens[0] || nom == prs->parens[1]) {
      kids->push_back(prs);
      ++k;
      if (k >= m)
        break;
    }
  }
}

}
