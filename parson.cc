#define __MAKEMORE_PARSON_CC__ 1

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>

#include <map>
#include <set>
#include <string>

#include "sha256.c"
#include "prenoms.c"
#include "surnoms.c"

#include "parson.hh"
#include "random.hh"

namespace makemore {

uint64_t Parson::hash_nom(const char *nom, unsigned int variant) {
  uint8_t hash[32];
  SHA256_CTX sha;
  sha256_init(&sha);
  sha256_update(&sha, (const uint8_t *)nom, strlen(nom));

  if (variant) {
    char buf[32];
    sprintf(buf, "/%u", variant);
    sha256_update(&sha, (const uint8_t *)buf, strlen(buf));
  }

  sha256_final(&sha, hash);

  uint64_t h;
  memcpy(&h, hash, 8);
  return h;
}

uint64_t Parson::hash_tag(const char *tag) {
  uint8_t hash[32];
  SHA256_CTX sha;
  sha256_init(&sha);
  sha256_update(&sha, (const uint8_t *)"#", 1);
  sha256_update(&sha, (const uint8_t *)tag, strlen(tag));
  sha256_final(&sha, hash);

  uint64_t h;
  memcpy(&h, hash, 8);
  return h;
}


bool Parson::valid_nom(const char *nom) {
  if (strlen(nom) > 31)
    return false;
  if (!nom[0])
    return false;

  if (nom[0] >= '0' && nom[0] <= '9')
    return false;

  for (unsigned int i = 0; i < 32; ++i) {
    if (!nom[i])
      break;
    if (!(nom[i] >= 'a' && nom[i] <= 'z' || nom[i] == '_' || nom[i] >= '0' && nom[i] <= '9')) {
      return false;
    }
  }
  return true;
}

static std::map<std::string, bool> _gender_map;

static void _make_gender_map() {
  for (unsigned int i = 0; prenoms[i]; ++i) {
    _gender_map[prenoms[i]] = prenom_gender[i];
  }
}

bool Parson::female_nom(const char *nom) {
  if (!valid_nom(nom))
    return false;
  if (_gender_map.empty())
    _make_gender_map();

  const char *p = nom;
  while (*p == '_')
    ++p;

  Nom prenom;
  strcpy(prenom, p);
  if (char *q = strchr(prenom, '_'))
    *q = 0;
  auto i = _gender_map.find(prenom);
  if (i != _gender_map.end())
    return !i->second;
  
  unsigned int l = strlen(prenom);
  if (l >= 1) {
    if (prenom[l - 1] == 'a' || prenom[l - 1] == 'i')
      return true;
  }

  if (l >= 1) {
    if (prenom[l - 1] == 'o')
      return false;
  }
  if (l >= 2) {
    if (prenom[l - 1] == 's' && prenom[l - 2] == 'u')
      return false;
  }
      
      
  return (randuint() % 2 == 0);
}

void Parson::paren_noms(const char *nom, char *mnom, char *fnom) {
  assert(valid_nom(nom));

  unsigned int mprenomid;
  do {
    mprenomid = randuint() % ((sizeof(prenoms) / sizeof(*prenoms)) - 1);
  } while (prenom_gender[mprenomid] != 1);
  const char *mprenom = prenoms[mprenomid];

  Nom surnom;
  if (const char *p = strrchr(nom, '_')) {
    strcpy(surnom, p + 1);
  } else {
    strcpy(surnom, nom);
  }
  surnom[16] = '\0';
  const char *msurnom = surnom;

  unsigned int fprenomid;
  do {
    fprenomid = randuint() % ((sizeof(prenoms) / sizeof(*prenoms)) - 1);
  } while (prenom_gender[fprenomid] != 0);
  const char *fprenom = prenoms[fprenomid];

  unsigned int fsurnomid = randuint() % ((sizeof(surnoms) / sizeof(*surnoms)) - 1);
  const char *fsurnom = surnoms[fsurnomid];

  sprintf(mnom, "%s_%s", mprenom, msurnom);
  sprintf(fnom, "%s_%s", fprenom, fsurnom);
}

std::string Parson::bread(const char *nom0, const char *nom1, uint8_t gender) {
  std::string newnom;

  unsigned int prenomid;
  do {
    prenomid = randuint() % ((sizeof(prenoms) / sizeof(*prenoms)) - 1);
  } while (prenom_gender[prenomid] != gender);

  const char *prenom = prenoms[prenomid];

  newnom = prenom;

  const char *suf;
  if (const char *p = strrchr(nom0, '_')) {
    newnom += p;
  } else {
    newnom += "_";
    newnom += nom0;
  }

  if (newnom.length() > 31)
    newnom = std::string(newnom.c_str(), 31);

  return newnom;
}
  
void Parson::add_fren(const char *fnom) {
  assert(valid_nom(fnom));
  for (int i = 0; i < nfrens; ++i)
    if (!strcmp(frens[i], fnom))
      return;
  memmove(frens + 1, frens, sizeof(Nom) * (nfrens - 1));
  memset(frens[0], 0, sizeof(Nom));
  strcpy(frens[0], fnom);
}

void Parson::set_parens(const char *anom, const char *bnom) {
  if (anom)
    assert(valid_nom(anom));
  if (bnom)
    assert(valid_nom(anom));
  if (!anom)
    anom = "";
  if (!bnom)
    bnom = "";

  memset(parens, 0, sizeof(parens));
  strcpy(parens[0], anom);
  strcpy(parens[1], bnom);
}

void Parson::initialize(const char *_nom, double mean, double dev) {
  assert(valid_nom(_nom));
  if (!strcmp(nom, _nom)) {
    return;
  }

  memset(nom, 0, sizeof(Nom));
  strcpy(nom, _nom);
  hash = hash_nom(_nom);

  seedrand(hash);

  for (unsigned int i = 0; i < ncontrols; ++i)
    controls[i] = sigmoid(mean + randgauss() * dev);

  memset(tags, 0, sizeof(tags));

  for (unsigned int i = 0; i < nattrs; ++i)
    attrs[i] = (randuint() % 2) ? 255 : 0;
  if (nattrs >= 21) {
    if (female_nom(nom))
      attrs[20] = 0;
    else
      attrs[20] = 255;
  }

  created = 0;
  revised = 0;
  creator = 0;
  revisor = 0;

  visits = 0;
  visited = 0;
  _activity = 0;

  target_lock = 0;
  control_lock = 0xFF;
  memset(parens, 0, sizeof(parens));
  paren_noms(nom, parens[0], parens[1]);
  memset(frens, 0, sizeof(frens));
  memset(target, 0, sizeof(target));

  seedrand();
}

void ParsonDB::fill_fam(const char *nom, Parson::Nom *fam) {
  std::set<std::string> seen;

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

void ParsonDB::create(const char *_fn, unsigned int n) {
  assert(strlen(_fn) < 4000);
  std::string fn = _fn;

  int fd = ::open(fn.c_str(), O_RDWR | O_CREAT | O_EXCL, 0777);
  assert(fd != -1);

  int ret;
  ret = ::ftruncate(fd, n * sizeof(Parson));
  assert(ret == 0);

  ret = ::close(fd);
  assert(ret == 0);

  ParsonDB *pdb = new ParsonDB(fn.c_str());
  assert(pdb->n == n);
  memset((uint8_t *)pdb->db, 0, n * sizeof(Parson));
  delete pdb;
}

ParsonDB::ParsonDB(const char *_fn) {
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

ParsonDB::~ParsonDB() {
  size_t map_size = ((n * sizeof(Parson)) + 4095) & ~4095;
  ::munmap((void *)db, map_size);
  ::close(fd);
}

Parson *ParsonDB::pick() {
  unsigned int tries = 0;

  while (1) {
    Parson *cand = db + randuint() % n;
    if (!cand->created)
      continue;
    return cand;
  }
}


Parson *ParsonDB::pick(bool male) {
  unsigned int tries = 0;

  while (1) {
    Parson *cand = db + randuint() % n;
    if (!cand->created)
      continue;

    if (cand->attrs[20] == 255 * male) {
      return cand;
    }

    ++tries;
    assert(tries < 65536);
  }
}

Parson *ParsonDB::pick(bool male, bool old) {
  unsigned int tries = 0;

  while (1) {
    Parson *cand = db + randuint() % n;
    if (!cand->created)
      continue;

    if (cand->attrs[20] == 255 * male && cand->attrs[39] == 255 * !old) {
      return cand;
    }

    ++tries;
    assert(tries < 65536);
  }
}

Parson *ParsonDB::find(const char *nom) {
  if (!Parson::valid_nom(nom))
    return NULL;

  Parson *p = NULL;
  std::multimap<double, Parson*> act_cand;

  for (unsigned int j = 0; j < nvariants; ++j) {
    Parson *cand = db + Parson::hash_nom(nom, j) % n;

    if (!strcmp(cand->nom, nom)) {
      p = cand;
      break;
    }

    double act = cand->activity();
    act_cand.insert(std::make_pair(act, cand));
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

    p->initialize(nom, 0, 1);

    const char *dan_nom = "synthetic_dan_brumleve";
    if (strcmp(nom, dan_nom)) {
      p->add_fren(dan_nom);
      Parson *dan = find(dan_nom);
      dan->add_fren(nom);

      memcpy(dan->parens[0], dan->nom, 32);
      memcpy(dan->parens[1], dan->nom, 32);
    }
  }

  return p;
}

}

//int main(){
//  Parson p;
//  printf("%lu\n", (uint8_t *)p.frens - (uint8_t *)&p);
//}
