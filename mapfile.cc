#define __MAKEMORE_MAPFILE_CC__ 1
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <assert.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include "mapfile.hh"

#include "cudamem.hh"

namespace makemore {

Mapfile::Mapfile(const std::string &_fn) {
  fn = _fn;
  fd = -1;

  fd = ::open(fn.c_str(), O_RDWR | O_CREAT, 0777);
  if (fd < 0) {
    fprintf(stderr, "%s: %s\n", fn.c_str(), strerror(errno));
    assert(!(fd < 0));
  }

  top = 0;
  size = 0;
  base = NULL;

  struct stat stbuf;
  int ret = ::fstat(fd, &stbuf);
  assert(ret == 0);
  grow(stbuf.st_size);
}

Mapfile::~Mapfile() {
  ::munmap(base, size);
  ::close(fd);
}

void Mapfile::grow(unsigned long new_size) {
  new_size = (new_size + 4095) & ~4095;

  if (new_size <= size)
    return;

  int ret = ::ftruncate(fd, new_size);
  assert(ret == 0);

  void *new_base;
  if (size == 0) {
    new_base = ::mmap(NULL, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  } else {
    new_base = ::mremap(base, size, new_size, MREMAP_MAYMOVE);
  }
  assert(new_base != MAP_FAILED);
  assert(new_base != NULL);

  size = new_size;
  base = new_base;
}

void Mapfile::mapv(void *cudata, unsigned long n) {
  assert(ptrofflen.find(cudata) == ptrofflen.end());

  auto offlen = std::make_pair(top, n);
  ptrofflen.insert(std::make_pair(cudata, offlen));

  top += n;
  grow(top);
}

void Mapfile::load(void *ptr, unsigned long verify_len) {
  auto i = ptrofflen.find(ptr);
  assert(i != ptrofflen.end());
  unsigned long off = i->second.first;
  unsigned long len = i->second.second;
  assert(verify_len == len);
  encude((const uint8_t *)base + off, len, (uint8_t *)ptr);
}

void Mapfile::load(void *ptr) {
  auto i = ptrofflen.find(ptr);
  assert(i != ptrofflen.end());
  unsigned long off = i->second.first;
  unsigned long len = i->second.second;
  encude((const uint8_t *)base + off, len, (uint8_t *)ptr);
}

void Mapfile::load() {
  for (auto i = ptrofflen.begin(); i != ptrofflen.end(); ++i) {
    void *ptr = i->first;
    unsigned long off = i->second.first;
    unsigned long len = i->second.second;

    encude((const uint8_t *)base + off, len, (uint8_t *)ptr);
  }
}

void Mapfile::save(void *ptr) {
  auto i = ptrofflen.find(ptr);
  assert(i != ptrofflen.end());
  unsigned long off = i->second.first;
  unsigned long len = i->second.second;
  decude((const uint8_t *)ptr, len, (uint8_t *)base + off);
}

void Mapfile::save(void *ptr, unsigned long verify_len) {
  auto i = ptrofflen.find(ptr);
  assert(i != ptrofflen.end());
  unsigned long off = i->second.first;
  unsigned long len = i->second.second;
  assert(len == verify_len);
  decude((const uint8_t *)ptr, len, (uint8_t *)base + off);
}

void Mapfile::save() {
  for (auto i = ptrofflen.begin(); i != ptrofflen.end(); ++i) {
    void *ptr = i->first;
    unsigned long off = i->second.first;
    unsigned long len = i->second.second;

    decude((const uint8_t *)ptr, len, (uint8_t *)base + off);
  }
}


}
