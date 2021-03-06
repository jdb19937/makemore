#define __MAKEMORE_PARSON_CC__ 1

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>

#include <openssl/sha.h>

#include <map>
#include <set>
#include <string>

#include "prenoms.c"
#include "surnoms.c"

#include "parson.hh"
#include "hashbag.hh"
#include "numutils.hh"
#include "random.hh"
#include "pipeline.hh"
#include "ppm.hh"

namespace makemore {

uint64_t Parson::hash_nom(const char *nom, unsigned int variant) {
  uint8_t hash[32];
  SHA256_CTX sha;
  SHA256_Init(&sha);
  SHA256_Update(&sha, (const uint8_t *)nom, strlen(nom));

  if (variant) {
    char buf[32];
    sprintf(buf, "/%u", variant);
    SHA256_Update(&sha, (const uint8_t *)buf, strlen(buf));
  }

  SHA256_Final(hash, &sha);

  uint64_t h;
  memcpy(&h, hash, 8);
  return h;
}

bool Parson::valid_nom(const char *nom) {
  if (strlen(nom) > 31)
    return false;
  if (!nom[0])
    return false;

  for (unsigned int i = 0; i < 32; ++i) {
    if (!nom[i])
      break;
    if (!(
      nom[i] >= 'a' && nom[i] <= 'z' ||
      nom[i] == '.' ||
      nom[i] == '_' ||
      nom[i] >= '0' && nom[i] <= '9'
    )) {
      return false;
    }
  }
  return true;
}

bool Parson::valid_tag(const char *tag) {
  if (strlen(tag) > 31)
    return false;
  if (!tag[0])
    return false;

  for (unsigned int i = 0; i < 32; ++i) {
    if (!tag[i])
      break;
    if (!(
      tag[i] >= 'a' && tag[i] <= 'z' ||
      tag[i] == ':' ||
      tag[i] == ',' ||
      tag[i] == '.' ||
      tag[i] == '-' ||
      tag[i] == '_' ||
      tag[i] >= '0' && tag[i] <= '9'
    )) {
      return false;
    }
  }
  return true;
}

void Parson::set_pass(const std::string &password) {
  if (password == "") {
    memset(pass, 0, sizeof(pass));
    memset(salt, 0, sizeof(salt));
    return;
  }

  for (unsigned int i = 0; i < sizeof(salt); ++i)
    salt[i] = randuint() & 0xFF;

  SHA256_CTX sha;
  SHA256_Init(&sha);
  SHA256_Update(&sha, (const uint8_t *)password.c_str(), password.length() + 1);
  SHA256_Update(&sha, (const uint8_t *)salt, sizeof(salt));
  SHA256_Final(pass, &sha);
}

bool Parson::check_pass(const std::string &password) const {
  assert(sizeof(pass) == 32);

  uint8_t check_pass[32];
  memset(check_pass, 0, sizeof(check_pass));
  if (!memcmp(check_pass, pass, 32))
    return true;

  SHA256_CTX sha;
  SHA256_Init(&sha);
  SHA256_Update(&sha, (const uint8_t *)password.c_str(), password.length() + 1);
  SHA256_Update(&sha, (const uint8_t *)salt, sizeof(salt));
  SHA256_Final(check_pass, &sha);

  return (0 == memcmp(check_pass, pass, 32));
}

static std::map<std::string, bool> _gender_map;

static void _make_gender_map() {
  for (unsigned int i = 0; prenoms[i]; ++i) {
    _gender_map[prenoms[i]] = prenom_gender[i];
  }
}

std::string Parson::gen_nom() {
  return gen_nom((bool *)NULL);
}

std::string Parson::gen_nom(bool *gender) {
  unsigned int prenomid;
  prenomid = randuint() % ((sizeof(prenoms) / sizeof(*prenoms)) - 1);

  if (gender)
    *gender = prenom_gender[prenomid];

  const char *prenom = prenoms[prenomid];

  unsigned int surnomid;
  surnomid = randuint() % ((sizeof(surnoms) / sizeof(*surnoms)) - 1);

  const char *surnom = surnoms[surnomid];

  std::string nomstr;
  nomstr = prenom;
  nomstr += "_";
  nomstr += surnom;

  return nomstr;
}

std::string Parson::gen_nom(bool gender) {
  unsigned int prenomid;
  do {
    prenomid = randuint() % ((sizeof(prenoms) / sizeof(*prenoms)) - 1);
  } while (prenom_gender[prenomid] != gender);

  const char *prenom = prenoms[prenomid];

  unsigned int surnomid;
  surnomid = randuint() % ((sizeof(surnoms) / sizeof(*surnoms)) - 1);

  const char *surnom = surnoms[surnomid];

  std::string nomstr;
  nomstr = prenom;
  nomstr += "_";
  nomstr += surnom;

  return nomstr;
}

bool Parson::female_nom(const char *nom) {
  if (!valid_nom(nom))
    return false;
  if (_gender_map.empty())
    _make_gender_map();

  const char *p = nom;
  while (*p == '_')
    ++p;

  Nom prenom;
  strcpy(prenom, p);
  if (char *q = strchr(prenom, '_'))
    *q = 0;
  auto i = _gender_map.find(prenom);
  if (i != _gender_map.end())
    return !i->second;
  
  unsigned int l = strlen(prenom);
  if (l >= 1) {
    if (prenom[l - 1] == 'a' || prenom[l - 1] == 'i')
      return true;
  }

  if (l >= 1) {
    if (prenom[l - 1] == 'o')
      return false;
  }
  if (l >= 2) {
    if (prenom[l - 1] == 's' && prenom[l - 2] == 'u')
      return false;
  }
      
      
  return (randuint() % 2 == 0);
}

void Parson::paren_noms(const char *nom, char *mnom, char *fnom) {
  assert(valid_nom(nom));

  unsigned int mprenomid;
  do {
    mprenomid = randuint() % ((sizeof(prenoms) / sizeof(*prenoms)) - 1);
  } while (prenom_gender[mprenomid] != 1);
  const char *mprenom = prenoms[mprenomid];

  Nom surnom;
  if (const char *p = strrchr(nom, '_')) {
    strcpy(surnom, p + 1);
  } else {
    strcpy(surnom, nom);
  }
  surnom[16] = '\0';
  const char *msurnom = surnom;

  unsigned int fprenomid;
  do {
    fprenomid = randuint() % ((sizeof(prenoms) / sizeof(*prenoms)) - 1);
  } while (prenom_gender[fprenomid] != 0);
  const char *fprenom = prenoms[fprenomid];

  unsigned int fsurnomid = randuint() % ((sizeof(surnoms) / sizeof(*surnoms)) - 1);
  const char *fsurnom = surnoms[fsurnomid];

  sprintf(mnom, "%s_%s", mprenom, msurnom);
  sprintf(fnom, "%s_%s", fprenom, fsurnom);
}

std::string Parson::bread_nom(const char *nom0, const char *nom1, uint8_t gender) {
  std::string newnom;

  unsigned int prenomid;
  do {
    prenomid = randuint() % ((sizeof(prenoms) / sizeof(*prenoms)) - 1);
  } while (prenom_gender[prenomid] != gender);

  const char *prenom = prenoms[prenomid];

  newnom = prenom;

  const char *suf;
  if (const char *p = strrchr(nom0, '_')) {
    newnom += p;
  } else {
    newnom += "_";
    newnom += nom0;
  }

  if (newnom.length() > 31)
    newnom = std::string(newnom.c_str(), 31);

  return newnom;
}

bool Parson::has_tag(const char *tag) const {
  for (unsigned int i = 0; i < ntags; ++i)
    if (!strcmp(tags[i], tag))
      return true;
  return false;
}

bool Parson::has_fren(const char *nom) {
  for (unsigned int i = 0; i < nfrens; ++i)
    if (!strcmp(frens[i], nom))
      return true;
  return false;
}

void Parson::add_tag(const char *tag) {
  assert(valid_tag(tag));

  for (unsigned int i = 0; i < ntags; ++i)
    if (!strcmp(tags[i], tag))
      return;
  memmove(tags + 1, tags, sizeof(Tag) * (ntags - 1));
  memset(tags[0], 0, sizeof(Tag));
  strcpy(tags[0], tag);
}


void Parson::add_fren(const char *nom) {
  assert(valid_nom(nom));

  for (unsigned int i = 0; i < nfrens; ++i)
    if (!strcmp(frens[i], nom))
      return;
  memmove(frens + 1, frens, sizeof(Nom) * (nfrens - 1));
  memset(frens[0], 0, sizeof(Nom));
  strcpy(frens[0], nom);
}

void Parson::del_tag(const char *tag) {
  assert(valid_tag(tag));
  for (unsigned int i = 0; i < ntags; ++i) {
    if (!strcmp(tags[i], tag)) {
      memmove(tags + i, tags + i + 1, sizeof(Tag) * (ntags - i - 1));
      memset(tags + ntags - 1, 0, sizeof(Tag));
    }
  }
}

void Parson::del_fren(const char *nom) {
  assert(valid_nom(nom));
  for (unsigned int i = 0; i < nfrens; ++i) {
    if (!strcmp(frens[i], nom)) {
      memmove(frens + i, frens + i + 1, sizeof(Nom) * (nfrens - i - 1));
      memset(frens + nfrens - 1, 0, sizeof(Nom));
    }
  }
}

void Parson::push_paren(const std::string &pnom) {
  assert(valid_nom(pnom));

  if (pnom == parens[0] || pnom == parens[1])
    return;

  const char *styles[] = {
    "human", "dog", "cat",
    "young", "old",
    "male", "female",
    "white", "black", "hispanic", "asian",
    "blonde_hair", "black_hair", "brown_hair", 
    "bald", "mustache", "glasses", "smiling",
    ""
  };
  for (unsigned int i = 0; *styles[i]; ++i)
    if (pnom == styles[i])
      return;

  memcpy(parens[1], parens[0], sizeof(Nom));
  strcpy(parens[0], pnom.c_str());
}

void Parson::set_parens(const char *anom, const char *bnom) {
  if (anom)
    assert(valid_nom(anom));
  if (bnom)
    assert(valid_nom(anom));
  if (!anom)
    anom = "";
  if (!bnom)
    bnom = "";

  memset(parens, 0, sizeof(parens));
  strcpy(parens[0], anom);
  strcpy(parens[1], bnom);
}

#if 0
void Parson::initialize(const char *_nom, double mean, double dev) {
  assert(valid_nom(_nom));
  if (!strcmp(nom, _nom)) {
    return;
  }

  memset(nom, 0, sizeof(Nom));
  strcpy(nom, _nom);
  uint64_t hash = hash_nom(_nom);

  seedrand(hash);

  for (unsigned int i = 0; i < ncontrols; ++i)
    controls[i] = sigmoid(mean + randgauss() * dev);

  memset(tags, 0, sizeof(tags));

  if (female_nom(nom))
    strcpy(tags[0], "female");
  else
    strcpy(tags[0], "male");

  created = 0;
  revised = 0;
  creator = 0;
  revisor = 0;

  visits = 0;
  visited = 0;
  last_activity = 0;

  generated = 0;
  acted = 0;
  memset(parens, 0, sizeof(parens));
  paren_noms(nom, parens[0], parens[1]);
  memset(frens, 0, sizeof(frens));
  memset(partrait, 0, sizeof(partrait));

  seedrand();
}
#endif

#if 0
void Parson::_from_pipe(Pipeline *pipe, unsigned int mbi) {
  assert(mbi < pipe->mbn);
  assert(pipe->ctrlay->n == ncontrols);
  unsigned long dd3 = dim * dim * 3;
  assert(pipe->outlay->n == dd3);

  dtobv(pipe->ctrbuf + mbi * ncontrols, controls, ncontrols);
  dtobv(pipe->outbuf + mbi * dd3, partrait, dd3);
}

void Parson::generate(Pipeline *pipe, long min_age) {
  Parson *me = this;
  pipe->generate(&me, 1, min_age);
}
#endif

bool Parson::load(FILE *fp) {
  size_t ret;
  ret = fread((uint8_t *)this, 1, sizeof(Parson), fp);

  if (ret != sizeof(Parson)) {
    assert(feof(fp));
    return false;
  }

  return true;
}

void Parson::save(FILE *fp) {
  size_t ret;
  ret = fwrite((uint8_t *)this, 1, sizeof(Parson), fp);
  assert(ret == sizeof(Parson));
}

}
