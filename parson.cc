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

bool Parson::female_nom(const char *nom) {
  if (!valid_nom(nom))
    return false;

  const char *p = nom;
  while (*p == '_')
    ++p;

  p = strchr(nom, '_');
  if (!p)
    p = nom + strlen(nom);
  --p;
  assert(p > nom);
  assert(p[1] == 0 || p[1] == '_');

  return (*p == 'a' || *p == 'i');
}
  

void Parson::initialize(const char *_nom, double mean, double dev) {
  assert(!created);

  assert(valid_nom(_nom));
  // if (!strcmp(nom, _nom))
  //   return;

  memcpy(nom, _nom, sizeof(nom));
  hash = hash_nom(_nom);

  seedrand(hash);

  for (unsigned int i = 0; i < ncontrols; ++i) {
    double z = sigmoid(mean + randgauss() * dev);

    z *= 256.0;
    long lz = lround(z);
    if (lz > 255) lz = 255;
    if (lz < 0) lz = 0;
    controls[i] = lz;
  }

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

  seedrand();

  created = 0;
  revised = 0;
  creator = 0;
  revisor = 0;

  memset(frens, 0, sizeof(frens));
  memset(target, 0, sizeof(target));
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
    p->initialize(nom, 0, 0.5);
  }

  return p;
}

//int main(){printf("%lu\n", sizeof(Parson)); }
