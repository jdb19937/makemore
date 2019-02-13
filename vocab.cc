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
  n = 1;
  tags.resize(1);
  tags[0] = new char[1];
  strcpy(tags[0], "");

  bags.resize(1);
  bags[0].clear();

#if 0
  for (unsigned int i = 0; i < 1000; ++i) {
    char buf[32];
    sprintf(buf, "huh%03u", i);
    add(buf);
  }
#endif
}

void Vocab::add(const char *str) {
  vector<string> phrases;
  split(str, ',', &phrases);

  for (auto pi = phrases.begin(); pi != phrases.end(); ++pi) {
    vector<string> words;
    split(pi->c_str(), ' ', &words);

    for (auto wi = words.begin(); wi != words.end(); ++wi) {
      const char *tag = wi->c_str();
      if (seen_tag.count(tag))
        continue;

      tags.resize(n + 1);
      tags[n] = new char[strlen(tag) + 1];
      strcpy(tags[n], tag);

      bags.resize(n + 1);
      bags[n].clear();
      bags[n].add(tag);

      seen_tag.insert(tag);

      ++n;
    }
  }
}

const char *Vocab::closest(const Wordvec &x, const Wordvec **y) const {
  const double *m = (const double *)bags.data();
  unsigned int k = 256;

  unsigned int i = makemore::closest(x.vec, m, k, n);
  assert(i >= 0 && i < n);

  const char *w = tags[i];
  if (i == 0)
    return NULL;

  if (y)
    *y = &bags[i];

  return w;
}

}
