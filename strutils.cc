#define __MAKEMORE_STRUTILS_CC
#include "strutils.hh"

#include <string.h>

#include <map>
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

void splitwords(const std::string &str, vector<string> *words) {
  words->clear();

  const char *p = str.c_str();
  while (isspace(*p))
    ++p;

  while (*p) {
    const char *q = p;
    while (*q && !isspace(*q)) 
      ++q;

    words->push_back(string(p, q - p));

    p = q;
    while (isspace(*p))
      p++;
  }
}

std::string varsubst(const std::string &str, const std::map<std::string, std::string>& dict) {
  vector<string> words;
  splitwords(str, &words);

  for (unsigned int wi = 0, wn = words.size(); wi < wn; ++wi) {
    auto di = dict.find(words[wi]);
    if (di != dict.end())
      words[wi] = di->second;
  }

  return joinwords(words);
}


void splitparts(const string &str, vector<string> *parts) {
  parts->clear();

  const char *p = str.c_str();

  while (const char *q = strchr(p, ',')) {
    while (isspace(*p))
      ++p;
    const char *r = q;
    while (r > p && isspace(*(r - 1)))
      --r;

    parts->push_back(string(p, r - p));

    p = q + 1;
  }

  while (isspace(*p))
    ++p;
  const char *r = p + strlen(p);
  while (r > p && isspace(*(r - 1)))
    --r;

  if (r > p)
    parts->push_back(string(p, r - p));
}

void splitlines(const string &str, vector<string> *lines) {
  lines->clear();

  const char *p = str.c_str();

  while (const char *q = strchr(p, '\n')) {
    while (isspace(*p))
      ++p;
    const char *r = q;
    while (r > p && isspace(*(r - 1)))
      --r;

    lines->push_back(string(p, r - p));

    p = q + 1;
  }

  while (isspace(*p))
    ++p;
  const char *r = p + strlen(p);
  while (r > p && isspace(*(r - 1)))
    --r;

  if (r > p)
    lines->push_back(string(p, r - p));
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
