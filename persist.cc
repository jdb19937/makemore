#define __MAKEMORE_PERSIST_CC__ 1

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>

#include <typeinfo>

#include "persist.hh"

#define PERROR(method, fn) \
  fprintf(stderr, "%s::%s: %s: %s\n", \
    typeid(*this).name(), method, fn, strerror(errno));


void Persist::load(const char *fn) {
  FILE *fp = fopen(fn, "r");
  if (!fp) {
    PERROR("load", fn);
    assert(fp);
  }
  load(fp);
  fclose(fp);
}

void Persist::save(const char *fn) {
  char tfn[4096] = {0};
  assert(strlen(fn) < 4000);
  sprintf(tfn, "%s.%d.tmp", fn, getpid());
  FILE *fp = fopen(tfn, "w");
  if (!fp) {
    PERROR("save", tfn);
    assert(fp);
  }
  save(fp);
  fclose(fp);
  int ret = rename(tfn, fn);
  if (ret != 0) {
    PERROR("save", fn);
    assert(ret == 0);
  }
}

