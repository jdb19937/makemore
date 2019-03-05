#define __MAKEMORE_IPDB_CC__ 1

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

#include <openssl/sha.h>

#include <map>
#include <set>
#include <string>

#include "ipdb.hh"
#include "random.hh"

namespace makemore {

void IPDB::create(const char *_fn, unsigned int n) {
  assert(strlen(_fn) < 4000);
  std::string fn = _fn;

  int fd = ::open(fn.c_str(), O_RDWR | O_CREAT | O_EXCL, 0777);
  assert(fd != -1);

  int ret;
  ret = ::ftruncate(fd, n * sizeof(Entry));
  assert(ret == 0);

  ret = ::close(fd);
  assert(ret == 0);

  IPDB *pdb = new IPDB(fn.c_str());
  assert(pdb->n == n);
  memset((uint8_t *)pdb->db, 0, n * sizeof(Entry));
  delete pdb;
}

IPDB::IPDB(const char *_fn) {
  assert(strlen(_fn) < 4000);
  fn = _fn;

  fd = ::open(fn.c_str(), O_RDWR | O_CREAT, 0777);
  assert(fd != -1);

  off_t size = ::lseek(fd, 0, SEEK_END);
  assert(size > 0);
  assert(size % sizeof(Entry) == 0);
  n = size / sizeof(Entry);

  assert(0 == ::lseek(fd, 0, SEEK_SET));

  size_t map_size = (size + 4095) & ~4095;
  void *map = ::mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  assert(map != NULL);
  assert(map != MAP_FAILED);

  db = (Entry *)map;
}

IPDB::~IPDB() {
  size_t map_size = ((n * sizeof(Entry)) + 4095) & ~4095;
  ::munmap((void *)db, map_size);
  ::close(fd);
}

IPDB::Entry *IPDB::find(uint32_t ip) {
  uint8_t hash[32];
  SHA256_CTX sha;
  SHA256_Init(&sha);
  SHA256_Update(&sha, (const uint8_t *)&ip, 4);
  SHA256_Update(&sha, (const uint8_t *)"asdf", 4);
  SHA256_Final(hash, &sha);
  uint64_t h;
  memcpy(&h, hash, 8);

  return (db + h % n);
}

}
