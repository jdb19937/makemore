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

std::string refsubst(const std::string &rsp, const std::string &req) {
  const char *reqstr;
  if ((reqstr = strrchr(req.c_str(), ',')))
    ++reqstr;
  else
    reqstr = req.c_str();
  vector<string> reqwords;
  split(reqstr, ' ', &reqwords);

  const char *rspstr = rsp.c_str();
  vector<string> rspwords;
  split(rspstr, ' ', &rspwords);

  for (auto i = rspwords.begin(); i != rspwords.end(); ++i) { 
    if (*i->c_str() == '\\') {
      int which = atoi(i->c_str() + 1) - 1;
      if (which >= 0 && which < reqwords.size()) {
        *i = reqwords[which];
      }
    }
  }

  return join(rspwords, ' ');
}

void splitthread(const vector<string> &words, vector<vector<string> > *threadp, const std::string &sep) {
  unsigned int tn = 0;
  vector<vector<string> > &thread = *threadp;

  thread.resize(1);
  thread[0].clear();
  ++tn;

  for (unsigned int wi = 0, wn = words.size(); wi < wn; ++wi) {
    if (words[wi] == sep) {
      thread.resize(tn + 1);
      thread[tn].clear();
      ++tn;
      continue;
    }
    thread[tn - 1].push_back(words[wi]);
  }
}

void jointhread(const vector<vector<string> > &thread, vector<string> *wordsp, const std::string &sep) {
  vector<string> &words = *wordsp;

  for (auto twordsi = thread.begin(); twordsi != thread.end(); ++twordsi) {
    const vector<string> &twords = *twordsi;

    if (twordsi != thread.begin())
      words.push_back(sep);
    for (unsigned int wi = 0, wn = twords.size(); wi < wn; ++wi) 
      words.push_back(twords[wi]);
  }
}

}
