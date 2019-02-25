#define __MAKEMORE_BUS_CC__ 1
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/file.h>
#include <sys/fcntl.h>

#include "bus.hh"
#include "random.hh"
#include "numutils.hh"

namespace makemore {

Bus::Bus() {
  fn = "";
  fp = NULL;
  fd = -1;
  locked = false;
}

Bus::Bus(const std::string &_fn) {
  fp = NULL;
  open(_fn);
}

Bus::~Bus() {
  if (fp)
    close();
}

void Bus::close() {
  unlock();
  ::fclose(fp);
  fp = NULL;
  fd = -1;
  fn = "";
}

void Bus::open(const std::string &_fn) {
  fn = _fn;
  
  assert(!fp);
  fp = fopen(fn.c_str(), "r+");
  assert(fp);
  setbuf(fp, NULL);

  fd = fileno(fp);
  locked = false;
}

void Bus::lock() {
  if (locked)
    return;
  assert(fd >= 0);
  int ret = ::flock(fd, LOCK_EX);
  assert(ret == 0);
}

void Bus::unlock() {
  if (!locked)
    return;
  assert(fd >= 0);
  int ret = ::flock(fd, LOCK_UN);
  assert(ret == 0);
}


void Bus::_seek_end() {
  int ret = fseek(fp, 0, SEEK_END); 
  assert(ret == 0);

  long off = ftell(fp);
  assert(off >= 0);

  if (unsigned int rem = off % sizeof(Parson)) {
    off += (sizeof(Parson) - rem);
    ret = fseek(fp, off, SEEK_SET);
    assert(ret == 0);
  }
  // assert(ftell(fp) == off);
  assert(off % sizeof(Parson) == 0);
}


void Bus::add(const Parson *p, unsigned int n) {
  lock();

  _seek_end();

  size_t ret = fwrite(p, sizeof(Parson), n, fp);
  assert(ret == n);

  unlock();
}

void Bus::add(const Parson **pp, unsigned int n) {
  lock();

  _seek_end();

  for (unsigned int i = 0; i < n; ++i) {
    const Parson *p = pp[i];
    size_t ret = fwrite(p, sizeof(Parson), 1, fp);
    assert(ret == 1);
  }

  unlock();
}

}
