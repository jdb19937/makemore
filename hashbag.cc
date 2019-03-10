#define __MAKEMORE_HASHBAG_CC__ 1

#include <openssl/sha.h>

#include <vector>
#include <string>
#include <algorithm>

#include "hashbag.hh"
#include "strutils.hh"
#include "closest.hh"
#include "vocab.hh"

namespace makemore {
using namespace std;

static int hexnum(char c) {
  if (c >= '0' && c <= '9')
    return c - '0';
  if (c >= 'A' && c <= 'F')
    return c - 'A' + 10;
  return -1;
}

static bool is_selfhash(const char *tag) {
  if (strlen(tag) != 33)
    return false;
  if (*tag != '~')
    return false;
  for (unsigned int i = 1; i < 33; ++i)
    if (hexnum(tag[i]) < 0)
      return false;
  return true;
}

void Hashbag::add(const char *tag, double w) {
  uint8_t hash[64];

  if (!*tag)
    return;

  if (*tag == '-') {
    unsigned int m = 1;
    ++tag;
    while (*tag == '-') {
      ++m;
      ++tag;
    }

    add(tag, -w * (double)m);
    return;
  }

  if (*tag == '+') {
    unsigned int m = 1;
    ++tag;
    while (*tag == '+') {
      ++m;
      ++tag;
    }

    add(tag, w * (double)m);
    return;
  }

  if (strchr(tag, '/')) {
    vector<string> tagparts;
    split(tag, '/', &tagparts);
    unsigned int ntagparts = tagparts.size();
    assert(ntagparts >= 2);

    double z = 1.0 / sqrt((double)ntagparts);
    for (unsigned int i = 0; i < ntagparts; ++i)
      add(tagparts[i].c_str(), w * z);

    return;
  }

  if (is_selfhash(tag)) {
    ++tag;
    for (unsigned int i = 0; i < 32; ++i) {
      int h = hexnum(tag[i]);
      vec[i * 4 + 0] += (h & 1) ? 1.0 : -1.0;
      vec[i * 4 + 1] += (h & 2) ? 1.0 : -1.0;
      vec[i * 4 + 2] += (h & 4) ? 1.0 : -1.0;
      vec[i * 4 + 3] += (h & 8) ? 1.0 : -1.0;
    }
    return;
  }

  SHA256_CTX sha;
  SHA256_Init(&sha);
  SHA256_Update(&sha, (const uint8_t *)"#", 1);
  SHA256_Update(&sha, (const uint8_t *)tag, strlen(tag));
  SHA256_Final(hash, &sha);

  SHA256_Init(&sha);
  SHA256_Update(&sha, (const uint8_t *)hash, 32);
  SHA256_Final(hash + 32, &sha);

#if 0
  assert(n >= 256);

  for (unsigned int i = 0; i < 32; ++i) {
    unsigned int j = hash[i];
    unsigned int k = hash[i + 32];
    int bit = (k & 1);
    vec[j] += bit ? w : -w;
  }
#endif

  assert(n == 256);

  for (unsigned int i = 0; i < 256; ++i) {
    unsigned int j = (i >> 3);
    unsigned int k = (i & 7);
    int bit = (hash[j] >> k) & 1;
    vec[i] += bit ? w : -w;
  }

}

std::string Hashbag::guesstract(const Vocab &v, double nfloor) {
  double sump = 0;
  vector< pair<double, string> > ps;

  for (auto vi = v.tags.begin(); vi != v.tags.end(); ++vi) {
    const char *word = *vi;

    double p = (Hashbag(word) * *this).sum();
    p -= nfloor * (double)Hashbag::n;
    if (p > 0)
      ps.push_back(make_pair(p, string(word)));
  }

  std::sort(ps.begin(), ps.end(), std::greater<>());
  double total = 0;
  for (auto i = ps.begin(); i != ps.end(); ++i) {
    total += i->first;
  }
  if (total <= 0)
    return "?";
  for (auto i = ps.begin(); i != ps.end(); ++i) {
    i->first /= total;
  }

//for (auto i = ps.begin(); i != ps.end(); ++i) {
//fprintf(stderr, "word=%s p=%lf\n", i->second.c_str(), i->first);
//}

  string *wordp = NULL;
  double r = randrange(0, 1.0);

  for (auto i = ps.begin(); i != ps.end() && r > 0; ++i) {
    r -= i->first;
    wordp = &i->second;
  }
  assert(wordp);

  return *wordp;
}

std::string Hashbag::decode(const vector<string> &words) {
  int best_i;
  double best_z;
  assert(words.size());

  for (int i = 0, n = words.size(); i < n; ++i) {
    Hashbag h;
    h.add(words[i].c_str());
    double z = (*this * h).sum();

    if (i == 0 || z > best_z) {
      best_i = i;
      best_z = z;
    }
  }

  return words[best_i];
}

std::string Hashbag::pick(const vector<string> &words, double phi) {
  Hashbag s;
  for (int i = 0, n = words.size(); i < n; ++i) {
    Hashbag h;
    h.add(words[i].c_str());

    double p = (*this * h).sum();
    p /= (double)Hashbag::n;

    p -= phi;
    if (p < 0.0)
      continue;
fprintf(stderr, "w=%s p=%lf\n", words[i].c_str(), p);

    h *= p;
    h *= randexp();
    s += h;
  }
fprintf(stderr, "\n");

  return s.decode(words);
}

}
