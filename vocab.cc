#define __MAKEMORE_VOCAB_CC__ 1
#include <assert.h>
#include <stdlib.h>
#include <ctype.h>

#include "vocab.hh"
#include "closest.hh"
#include "strutils.hh"

namespace makemore {  

using namespace std;

Vocab::Vocab() {
  clear();

#if 0
  for (unsigned int i = 0; i < 1000; ++i) {
    char buf[32];
    sprintf(buf, "huh%03u", i);
    add(buf);
  }
#endif
}

void Vocab::clear() {
  seen_tag.clear();

  n = 1;
  tags.resize(1);
  tags[0] = new char[1];
  strcpy(tags[0], "");

  bags.resize(1);
  bags[0].clear();
}

void Vocab::add(const char *str, const char *desc) {
  vector<string> words;
  split(str, ' ', &words);

  for (auto wi = words.begin(); wi != words.end(); ++wi) {
    std::string word = "";
    word += wi->c_str();

    const char *tag = word.c_str();
    if (seen_tag.count(tag))
      continue;

//fprintf(stderr, "adding tag [%s]\n", tag);

    tags.resize(n + 1);
    tags[n] = new char[strlen(tag) + 1];
    strcpy(tags[n], tag);

    bags.resize(n + 1);
    bags[n].clear();
    bags[n].add(tag);

    descs.resize(n + 1);
    descs[n] = NULL;
    if (desc) {
      descs[n] = new char[strlen(desc) + 1];
      strcpy(descs[n], desc);
    }

    seen_tag.insert(tag);

    ++n;
  }
}

const char *Vocab::closest(const Hashbag &x, const Hashbag **y, bool force) const {
  const double *m = (const double *)bags.data();
  unsigned int k = Hashbag::n;

  unsigned int i;
  if (force) {
    i = makemore::closest(x.vec, m + k, k, n - 1) + 1;
  } else {
    i = makemore::closest(x.vec, m, k, n);
  }
  assert(i >= 0 && i < n);

  const char *w = tags[i];
  if (i == 0)
    return NULL;

  if (y)
    *y = &bags[i];

  return w;
}

void Vocab::load(const char *fn) {
  FILE *fp;
  assert(fp = fopen(fn, "r"));
  load(fp);
  fclose(fp);
}

void Vocab::load(FILE *fp) {
  char buf[4096];

  while (1) {
    *buf = 0;
    char *unused = fgets(buf, sizeof(buf) - 1, fp);
    buf[sizeof(buf) - 1] = 0;
    char *p = strchr(buf, '\n');
    if (!p)
      break;
    *p = 0;

    p = strchr(buf, '#');
    if (p)
      *p = 0;
    char *q = buf;
    while (*q == ' ')
      ++q;
    if (!*q)
      continue;

    char *desc = strchr(q, '\t');
    if (desc) {
      *desc = 0;
      ++desc;
    }

    add(q, desc);
  }
}
 

}
