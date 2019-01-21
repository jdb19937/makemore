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

#include "sha256.c"
#include "prenoms.c"
#include "surnoms.c"

#include "parson.hh"
#include "random.hh"

uint64_t Parson::hash_nom(const char *nom) {
  uint8_t hash[32];
  SHA256_CTX sha;
  sha256_init(&sha);
  sha256_update(&sha, (const uint8_t *)nom, strlen(nom));
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
  
  if (unsigned int l = strlen(prenom)) {
    if (prenom[l - 1] == 'a' || prenom[l - 1] == 'i')
      return true;
  }
      
  return (randuint() % 3 == 0);
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

std::string Parson::bread(const char *nom0, const char *nom1, uint8_t *new_gender) {
  std::string newnom;

  unsigned int prenomid = randuint() % ((sizeof(prenoms) / sizeof(*prenoms)) - 1);
  const char *prenom = prenoms[prenomid];
  if (new_gender)
    *new_gender = prenom_gender[prenomid];

  newnom = prenom;

  const char *suf;
  if (const char *p = strrchr(nom1, '_')) {
    newnom += p;
    suf = p + 1;
  } else {
    newnom += "_";
    newnom += nom1;
    suf = nom1;
  }

  if (const char *p = strrchr(nom0, '_')) {
    if (strcmp(p + 1, suf)) {
      newnom += p;
    }
  } else {
    if (strcmp(nom0, suf)) {
      newnom += "_";
      newnom += nom0;
    }
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
  assert(!created);

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

  for (unsigned int i = 0; i < ntags; ++i)
    tags[i] = 0;

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

  target_lock = 0;
  control_lock = 0xFF;
  memset(parens, 0, sizeof(parens));
  paren_noms(nom, parens[0], parens[1]);
  memset(frens, 0, sizeof(frens));
  memset(target, 0, sizeof(target));

  seedrand();
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

Parson *ParsonDB::find(const char *nom) {
  if (!Parson::valid_nom(nom))
    return NULL;

  uint64_t h = Parson::hash_nom(nom);

  unsigned int i0 = h % n;
  unsigned int i = i0;

  while (db[i].created && strcmp(db[i].nom, nom)) {
    ++i;
    i %= n;
    assert(i != i0);
  }

  Parson *p = db + i;

  if (!p->created) {
    p->initialize(nom, 0, 1);

    const char *dan_nom = "synthetic_dan_brumleve";
    if (strcmp(nom, dan_nom)) {
      p->add_fren(dan_nom);
      Parson *dan = find(dan_nom);
      dan->add_fren(nom);
    }
  }

  return p;
}

//int main(){
//  Parson p;
//  printf("%lu\n", (uint8_t *)p.frens - (uint8_t *)&p);
//}
