#define __MAKEMORE_STRUTILS_CC
#include "strutils.hh"

#include <string.h>

#include <vector>
#include <string>

namespace makemore {
using namespace std;

void split(const char *str, char sep, vector<string> *words) {
  words->clear();

  const char *p = str;
  while (*p == sep)
    ++p;

  while (const char *q = strchr(p, sep)) {
    words->push_back(string(p, q - p));

    p = q + 1;
    while (*p == sep)
      p++;
  }

  if (*p)
    words->push_back(string(p));
}

bool read_line(FILE *fp, std::string *line) {
  char buf[4096];

  int c = getc(fp);
  if (c == EOF)
    return false;
  ungetc(c, fp);

  *buf = 0;
  char *unused = fgets(buf, sizeof(buf) - 1, fp);
  char *p = strchr(buf, '\n');
  if (!p)
    return false;
  *p = 0;

  *line = buf;
}

}
