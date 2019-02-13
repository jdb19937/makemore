#define __MAKEMORE_HASHBAG_CC__ 1
#include <vector>
#include <string>

#include "hashbag.hh"
#include "strutils.hh"
#include "closest.hh"
#include "vocab.hh"

#include "sha256.c"

namespace makemore {
using namespace std;

void Hashbag::add(const char *tag, double w) {
  uint8_t hash[64];

  SHA256_CTX sha;
  sha256_init(&sha);
  sha256_update(&sha, (const uint8_t *)"#", 1);
  sha256_update(&sha, (const uint8_t *)tag, strlen(tag));
  sha256_final(&sha, hash);

  sha256_init(&sha);
  sha256_update(&sha, (const uint8_t *)hash, 32);
  sha256_final(&sha, hash + 32);

  assert(n >= 256);

  for (unsigned int i = 0; i < 32; ++i) {
    unsigned int j = hash[i];
    unsigned int k = hash[i + 32];
    vec[j] += (k % 2) ? w : -w;
  }
}

}
