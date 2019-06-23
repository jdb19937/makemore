#define __MAKEMORE_CATALOG_CC__ 1

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>

#include <vector>
#include <string>
#include <algorithm>

#include "catalog.hh"
#include "partrait.hh"

namespace makemore {

Catalog::Catalog() {
}

Catalog::~Catalog() {
}

Catalog::Catalog(const std::string &dir) {
  add_dir(dir);
}

void Catalog::add_dir(const std::string &dir) {
  struct dirent *de;
  DIR *dp = opendir(dir.c_str());
  assert(dp);
  while ((de = readdir(dp))) {
    if (*de->d_name == '.')
      continue;
    size_t len = strlen(de->d_name);
    if (len < 4 || strcmp(de->d_name + len - 4, ".png"))
      continue;
    fn.push_back(dir + "/" + de->d_name);
  }
  closedir(dp);
  std::sort(fn.begin(), fn.end());
}

void Catalog::pick(Partrait *par, unsigned int n, bool randreflect) {
  assert(fn.size());
  for (unsigned int i = 0; i < n; ++i) {
    unsigned int j = randuint() % fn.size();
    par[i].load(fn[j]);

    if (randreflect && randuint() % 2)
      par[i].reflect();
  }
}

};

