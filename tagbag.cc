#define __MAKEMORE_TAGBAG_CC__ 1
#include <vector>
#include <string>

#include "tagbag.hh"

#include "sha256.c"

namespace makemore {
using namespace std;

Tagbag::Tagbag(const char *tag, double w) {
  uint8_t hash[32];
  SHA256_CTX sha;
  sha256_init(&sha);
  sha256_update(&sha, (const uint8_t *)"#", 1);
  sha256_update(&sha, (const uint8_t *)tag, strlen(tag));
  sha256_final(&sha, hash);

  assert(n >= 256);
  memset(vec, 0, sizeof(vec));
  for (unsigned int i = 0; i < 256; ++i) {
    unsigned int j = (i >> 3);
    unsigned int k = (i & 7);
    vec[i] = ((hash[j] >> k) & 1) ? w : -w;
  }
}

static void split(const char *str, vector<string> *words) {
  words->clear();

  const char *p = str;

  while (const char *q = strchr(p, ' ')) {
    words->push_back(string(p, q - p));

    p = q + 1;
    while (*p == ' ')
      p++;
  }

  if (*p)
    words->push_back(string(p));
}

void varsubst(vector<string> *words, unsigned int seed) {
  char buf[32];

  for (auto wi = words->begin(); wi != words->end(); ++wi) {
    const char *w = wi->c_str();
    if (*w == '$') {
      *wi = string(w + 1);
      sprintf(buf, "[%08X]", seed);
      *wi += buf;
    }
  }
}


void Tagbag::encode(const char *str, unsigned int seed) {
  clear();
  vector<string> strv;
  split(str, &strv);
  varsubst(&strv, seed);

  unsigned int nw = strv.size(), iw;
  for (iw = 0; iw < nw; ++iw) {
    Tagbag tw(strv[iw].c_str());
    tw.mul(1.0 / (1.0 + (double)iw));
    add(tw);
  }
}
  
}
