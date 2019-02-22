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

  Parson *pp0 = import(p->parens[0]);
  Parson *pp1 = import(p->parens[1]);
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

Zone::Zone(const char *_fn) {
  assert(strlen(_fn) < 4000);
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
}

Zone::~Zone() {
  size_t map_size = ((n * sizeof(Parson)) + 4095) & ~4095;
  ::munmap((void *)db, map_size);
  ::close(fd);
}

Parson *Zone::pick() {
  unsigned int tries = 0;

  while (1) {
    Parson *cand = db + randuint() % n;
    if (!cand->created)
      continue;
    return cand;
  }
}


Parson *Zone::pick(const char *tag, unsigned int max_tries) {
  for (unsigned int tries = 0; tries < max_tries; ++tries) {
    Parson *cand = db + randuint() % n;
    if (!cand->created)
      continue;

    if (cand->has_tag(tag)) {
      return cand;
    }
  }

  return NULL;
}

Parson *Zone::pick(const char *tag1, const char *tag2, unsigned int max_tries) {
  for (unsigned int tries = 0; tries < max_tries; ++tries) {
    Parson *cand = db + randuint() % n;
    if (!cand->created)
      continue;

    if (cand->has_tag(tag1) && cand->has_tag(tag2)) {
      return cand;
    }
  }

  return NULL;
}

Parson *Zone::find(const char *nom) const {
  if (!Parson::valid_nom(nom))
    return NULL;

  for (unsigned int j = 0; j < nvariants; ++j) {
    Parson *cand = db + Parson::hash_nom(nom, j) % n;
    if (!strcmp(cand->nom, nom))
      return cand;
  }

  return NULL;
}

Parson *Zone::import(const char *nom, Parson *evicted) {
  assert(Parson::valid_nom(nom));

  Parson *p = NULL;
  multimap<double, Parson*> act_cand;

  for (unsigned int j = 0; j < nvariants; ++j) {
    Parson *cand = db + Parson::hash_nom(nom, j) % n;

    if (!strcmp(cand->nom, nom)) {
      p = cand;
      break;
    }

    double act = cand->activity();
    act_cand.insert(make_pair(act, cand));
  }

  if (!p) {
    auto i = act_cand.begin(); 
    p = i->second;

    while (i != act_cand.end()) {
      if (!i->second->created) {
        p = i->second;
        break;
      }
      ++i;
    }

    if (evicted)
      memcpy(evicted, p, sizeof(Parson));

    p->initialize(nom, 0, 1);

    const char *dan_nom = "synthetic_dan_brumleve";
    if (strcmp(nom, dan_nom)) {
      p->add_fren(dan_nom);
      Parson *dan = import(dan_nom);
      dan->add_fren(nom);

      memcpy(dan->parens[0], dan->nom, 32);
      memcpy(dan->parens[1], dan->nom, 32);
    }
  }

  return p;
}

}