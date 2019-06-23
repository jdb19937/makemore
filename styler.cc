#define __MAKEMORE_STYLER_CC__ 1

#include <stdio.h>
#include <dirent.h>

#include <string>
#include <map>

#include "cholo.hh"
#include "parson.hh"
#include "styler.hh"

namespace makemore {

Styler::Styler(const std::string &dir) : Project(dir) {
  dim = (unsigned int)strtoul(config["dim"].c_str(), NULL, 0);
  assert(dim);

  assert(config["type"] == "styler");

  tmp = new double[dim];

  DIR *dp = ::opendir(dir.c_str());
  assert(dp);
  struct dirent *de;
  while ((de = readdir(dp))) {
    if (*de->d_name == '.')
      continue;
    if (strlen(de->d_name) < 6)
      continue;
    if (strcmp(de->d_name + strlen(de->d_name) - 6, ".cholo"))
      continue;

    std::string tag(de->d_name, strlen(de->d_name) - 6);
fprintf(stderr, "tag=%s\n", tag.c_str());
    add_cholo(tag, dir + "/" + std::string(de->d_name));
  }
  closedir(dp);

  assert(tag_cholo["base"]);
}

void Styler::encode(const double *ctr, Parson *prs) {
  Cholo *base = tag_cholo["base"];
  assert(base);

  memcpy(tmp, ctr, sizeof(double) * dim);

  for (unsigned int i = 0; i < Parson::ntags; ++i) {
    const char *tag = prs->tags[i];
    if (!*tag)
      break;
    Cholo *cholo = tag_cholo[tag];
    if (!cholo)
      continue;

    cholo->port(base, tmp, tmp);
  }

  assert(dim == Parson::ncontrols);
  base->encode(tmp, prs->controls);
fprintf(stderr, "here1 ctr=%lf,%lf,...\n", prs->controls[0], prs->controls[1]);
}

void Styler::generate(const Parson &prs, double *ctr, unsigned int m) {
  Cholo *base = tag_cholo["base"];
  assert(base);

  base->generate(prs.controls, ctr);

  for (unsigned int i = 0; i < Parson::ntags; ++i) {
    const char *tag = prs.tags[i];
    if (!*tag)
      break;
    Cholo *cholo = tag_cholo[tag];
    if (!cholo)
      continue;

    for (unsigned int j = 0; j < m; ++j)
      base->port(cholo, ctr, ctr);
  }
}

}
