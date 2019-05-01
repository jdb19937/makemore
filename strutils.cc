#define __MAKEMORE_STRUTILS_CC
#include "strutils.hh"

#include <string.h>

#include <map>
#include <vector>
#include <string>
#include <regex>

namespace makemore {
using namespace std;

bool match(const std::string &regexstr, const std::string &str) {
  bool ret = 0;

  try {
    const std::regex rx(regexstr);
    ret = std::regex_match(str, rx);
  } catch (std::regex_error) { }

  return ret;
}

void split(const std::string &sstr, char sep, vector<string> *words) {
  const char *str = sstr.c_str();
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

bool parsecolspec(const std::string &colspec, std::vector<int> *offp) {
  offp->clear();

  strvec parts;
  split(colspec.c_str(), ',', &parts);

  for (auto part : parts) {
    strvec range;
    split(part.c_str(), '-', &range);

    assert(range.size());
    int x = strtoul(range[0].c_str(), NULL, 0);
    if (range.size() == 1) {
      offp->push_back(x);
    } else {
      int y = strtoul(range[1].c_str(), NULL, 0);
      if (x <= y) {
        for (int z = x; z <= y; ++z) {
          offp->push_back(z);
        }
      } else {
        for (int z = y; z >= x; --z) {
          offp->push_back(z);
        }
      }
    }
  }
}

std::string moretpenc(const strvec &words, char sep) {
  strvec nwords = words;
  string extra;

  assert(isspace(sep) && sep != '\n');

  for (auto wordi = nwords.begin(); wordi != nwords.end(); ++wordi) {
    string &word = *wordi;
    unsigned int wordn = word.length();
    if (wordn == 0 || wordn > 255 || word[0] == '<' || hasspace(word) || hasnull(word)) {
      extra += word;
      char buf[64];
      sprintf(buf, "<%u", wordn);
      word = buf;
    }
  }

  return join(nwords, sep) + "\n" + extra;
}



/* 
   base64.cpp and base64.h

   base64 encoding and decoding with C++.

   Version: 1.01.00

   Copyright (C) 2004-2017 René Nyffenegger

   This source code is provided 'as-is', without any express or implied
   warranty. In no event will the author be held liable for any damages
   arising from the use of this software.

   Permission is granted to anyone to use this software for any purpose,
   including commercial applications, and to alter it and redistribute it
   freely, subject to the following restrictions:

   1. The origin of this source code must not be misrepresented; you must not
      claim that you wrote the original source code. If you use this source code
      in a product, an acknowledgment in the product documentation would be
      appreciated but is not required.

   2. Altered source versions must be plainly marked as such, and must not be
      misrepresented as being the original source code.

   3. This notice may not be removed or altered from any source distribution.

   René Nyffenegger rene.nyffenegger@adp-gmbh.ch

*/

static const std::string base64_chars = 
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

static inline bool is_base64(unsigned char c) {
  return (isalnum(c) || (c == '+') || (c == '/'));
}

std::string base64_encode(const uint8_t* bytes_to_encode, unsigned int in_len) {
  std::string ret;
  int i = 0;
  int j = 0;
  unsigned char char_array_3[3];
  unsigned char char_array_4[4];

  while (in_len--) {
    char_array_3[i++] = *(bytes_to_encode++);
    if (i == 3) {
      char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
      char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
      char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
      char_array_4[3] = char_array_3[2] & 0x3f;

      for(i = 0; (i <4) ; i++)
        ret += base64_chars[char_array_4[i]];
      i = 0;
    }
  }

  if (i)
  {
    for(j = i; j < 3; j++)
      char_array_3[j] = '\0';

    char_array_4[0] = ( char_array_3[0] & 0xfc) >> 2;
    char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
    char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

    for (j = 0; (j < i + 1); j++)
      ret += base64_chars[char_array_4[j]];

    while((i++ < 3))
      ret += '=';

  }

  return ret;

}

std::string base64_decode(const std::string &encoded_string) {
  int in_len = encoded_string.size();
  int i = 0;
  int j = 0;
  int in_ = 0;
  unsigned char char_array_4[4], char_array_3[3];
  std::string ret;

  while (in_len-- && ( encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
    char_array_4[i++] = encoded_string[in_]; in_++;
    if (i ==4) {
      for (i = 0; i <4; i++)
        char_array_4[i] = base64_chars.find(char_array_4[i]);

      char_array_3[0] = ( char_array_4[0] << 2       ) + ((char_array_4[1] & 0x30) >> 4);
      char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
      char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

      for (i = 0; (i < 3); i++)
        ret += char_array_3[i];
      i = 0;
    }
  }

  if (i) {
    for (j = 0; j < i; j++)
      char_array_4[j] = base64_chars.find(char_array_4[j]);

    char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
    char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);

    for (j = 0; (j < i - 1); j++) ret += char_array_3[j];
  }

  return ret;
}


// https://stackoverflow.com/a/29962178
std::string urldecode(const std::string &str){
  std::string ret;
  char ch;
  int i, ii, len = str.length();

  for (i = 0; i < len; i++){
    if (str[i] != '%') {
      if (str[i] == '+')
        ret += ' ';
      else
        ret += str[i];
    } else {
      sscanf(str.substr(i + 1, 2).c_str(), "%x", &ii);
      ch = static_cast<char>(ii);
      ret += ch;
      i = i + 2;
    }
  }

  return ret;
}

void cgiparse(const std::string &cgistr, std::map<std::string, std::string> *cgimap) {
  strvec cgiparts;
  split(cgistr, '&', &cgiparts);

  cgimap->clear();
  for (auto ekv : cgiparts) {
    const char *ekvp = ekv.data();
    std::string ek, ev;
    const char *p = strchr(ekvp, '=');
    if (!p)
      continue;

    ek = std::string(ekvp, p - ekvp);
    ev = p + 1;

    std::string k = urldecode(ek);
    std::string v = urldecode(ev);
    cgimap->insert(std::make_pair(k, v));
  }
}

}
