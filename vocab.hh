#ifndef __MAKEMORE_VOCAB_HH__
#define __MAKEMORE_VOCAB_HH__ 1

#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "hashbag.hh"

#include <string>
#include <vector>
#include <set>

namespace makemore {

struct Vocab {
  Vocab();

  unsigned int n;
  std::vector<char *> tags;
  std::vector<Hashbag> bags;
  std::set<std::string> seen_tag;
  std::vector<double> tag_weight;

  std::vector<char *> descs;

  void clear();

  void add(const char *tag, const char *desc = NULL, double weight = 1.0);
  void add(const std::string &tagstr, const char *desc = NULL, double weight = 1.0) {
    add(tagstr.c_str(), desc, weight);
  }

  const char *closest(const Hashbag &x, const Hashbag **y = NULL) const;

  void dump() {
    for (unsigned int i = 1; i < n; ++i) {
      if (descs[i]) {
        printf("%s\t%s\n", tags[i], descs[i]);
      } else {
        printf("%s\n", tags[i]);
      }
    }
  }

  void load(FILE *fp);
  void load(const char *fn);
};

}

#endif
