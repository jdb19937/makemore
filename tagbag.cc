#define __MAKEMORE_TAGBAG_CC__ 1
#include "tagbag.hh"

#include "sha256.c"

namespace makemore {

Tagbag::Tagbag(const char *tag) {
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
    vec[i] = ((hash[j] >> k) & 1) ? 1.0 : -1.0;
  }
}

}


