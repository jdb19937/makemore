  uint8_t hash[32];
  SHA256_CTX sha;
  sha256_init(&sha);
  sha256_update(&sha, (const uint8_t *)"#", 1);
  sha256_update(&sha, (const uint8_t *)tag, strlen(tag));
  sha256_final(&sha, hash);

  uint64_t h;
  memcpy(&h, hash, 8);
  return h;
}
