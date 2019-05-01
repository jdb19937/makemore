#define __MAKEMORE_PROJECT_CC__ 1

#include <string>
#include <netinet/in.h>

#include "project.hh"

namespace makemore {

using namespace std;

static string read_word(FILE *fp, char sep) {
  int c = getc(fp);
  if (c == EOF)
    return "";

  char buf[2];
  buf[0] = (char)c;
  buf[1] = 0;
  string word(buf);

  while (1) {
    c = getc(fp);
    if (c == EOF)
      return "";
    if (c == sep)
      break;
    buf[0] = (char)c;
    word += buf;
  }

  return word;
}

Project::Project(const std::string &_dir, unsigned int _mbn) {
  mbn = _mbn;
  assert(mbn > 0);

  dir = _dir;
  assert(strlen(dir.c_str()) < 4000);

  char configfn[4096];
  sprintf(configfn, "%s/config.tsv", dir.c_str());
  FILE *configfp = fopen(configfn, "r");
fprintf(stderr, "%s\n", configfn);
  assert(configfp);

  while (1) {
    string k = read_word(configfp, '\t');
    if (!k.length())
      break;
    string v = read_word(configfp, '\n');

    assert(config.find(k) == config.end());
    assert(v.length());
    config[k] = v;
  }
  fclose(configfp);
}

}
