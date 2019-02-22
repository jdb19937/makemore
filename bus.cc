#define __MAKEMORE_BUS_CC__ 1
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "bus.hh"
#include "random.hh"
#include "numutils.hh"

namespace makemore {

Bus::Bus() {
  n = 0;
}

Bus::Bus(const char *fn) {
  n = 0;
  load(fn);
}

Bus::~Bus() {
}


void Bus::load(const char *fn) {
  assert(strlen(fn) < 4000);
  FILE *fp = fopen(fn, "r");
  assert(fp);

  load(fp);

  fclose(fp);
}

void Bus::load(FILE *fp) {
  while (1) {
    Parson p;
    size_t ret;
    ret = fread(&p, sizeof(Parson), 1, fp);
    if (ret != 1) {
      assert(feof(fp));
      break;
    }

    add(p);
  }
}

void Bus::save(FILE *fp) {
  size_t ret;
  ret = fwrite(seat.data(), sizeof(Parson), seat.size(), fp);
  assert(ret == seat.size());
}

void Bus::add(const Parson &p) {
  seat.resize(n + 1);
  memcpy(&seat[n], &p, sizeof(Parson));
  ++n;
}

Parson *Bus::pick() {
  if (!seat.size())
    return NULL;

  while (1) {
    Parson *cand = seat.data() + randuint() % seat.size();
    if (!cand->created)
      continue;
    return cand;
  }
}


Parson *Bus::pick(const char *tag, unsigned int max_tries) {
  if (!seat.size())
    return NULL;

  for (unsigned int tries = 0; tries < max_tries; ++tries) {
    Parson *cand = seat.data() + randuint() % seat.size();
    if (!cand->created)
      continue;

    if (cand->has_tag(tag)) {
      return cand;
    }
  }

  return NULL;
}

Parson *Bus::pick(const char *tag1, const char *tag2, unsigned int max_tries) {
  if (!seat.size())
    return NULL;

  for (unsigned int tries = 0; tries < max_tries; ++tries) {
    Parson *cand = seat.data() + randuint() % seat.size();
    if (!cand->created)
      continue;

    if (cand->has_tag(tag1) && cand->has_tag(tag2)) {
      return cand;
    }
  }

  return NULL;
}

void Bus::generate(Pipeline *pipe, long min_age) {
  unsigned int mbn = pipe->mbn;
  unsigned int mbi = 0;
  unsigned long dd3 = Parson::dim * Parson::dim * 3;
  std::vector<unsigned int> todo;
  time_t now = time(NULL);

  assert(dd3 == sizeof(Parson::target));

  for (unsigned int i = 0, n = seat.size(); i < n; ++i) {
    Parson *p = &seat[i];

    if (min_age > 0) {
      if (now < p->generated + min_age) {
        continue;
      }
    } else if (min_age < 0) {
      if (p->generated > 0) {
        continue;
      }
    }

    todo.push_back(i);
  }

  unsigned long hashlen = pipe->ctxlay->n / mbn;
  assert(hashlen * mbn == pipe->ctxlay->n);
  assert(hashlen <= Hashbag::n);

  assert(pipe->outlay->n == mbn * dd3);
  for (unsigned int j = 0; j < todo.size(); j += mbn) {
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      Parson *p = &seat[todo[j + mbi]];
      p->load_pipe(pipe, mbi);
    }

    pipe->ctrlock = 0;
    pipe->tgtlock = -1;
    pipe->reencode();

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      Parson *p = &seat[todo[j + mbi]];
      p->save_pipe(pipe, mbi);
      p->generated = now;
    }
  }
}


}
