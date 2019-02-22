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

#include <map>
#include <set>
#include <string>

#include "sha256.c"
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
  sha256_init(&sha);
  sha256_update(&sha, (const uint8_t *)nom, strlen(nom));

  if (variant) {
    char buf[32];
    sprintf(buf, "/%u", variant);
    sha256_update(&sha, (const uint8_t *)buf, strlen(buf));
  }

  sha256_final(&sha, hash);

  uint64_t h;
  memcpy(&h, hash, 8);
  return h;
}

bool Parson::valid_nom(const char *nom) {
  if (strlen(nom) > 31)
    return false;
  if (!nom[0])
    return false;

  if (nom[0] >= '0' && nom[0] <= '9')
    return false;

  for (unsigned int i = 0; i < 32; ++i) {
    if (!nom[i])
      break;
    if (!(nom[i] >= 'a' && nom[i] <= 'z' || nom[i] == '_' || nom[i] >= '0' && nom[i] <= '9')) {
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

  if (tag[0] >= '0' && tag[0] <= '9')
    return false;

  for (unsigned int i = 0; i < 32; ++i) {
    if (!tag[i])
      break;
    if (!(tag[i] >= 'a' && tag[i] <= 'z' || tag[i] == '_' || tag[i] >= '0' && tag[i] <= '9')) {
      return false;
    }
  }
  return true;
}

static std::map<std::string, bool> _gender_map;

static void _make_gender_map() {
  for (unsigned int i = 0; prenoms[i]; ++i) {
    _gender_map[prenoms[i]] = prenom_gender[i];
  }
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

bool Parson::has_tag(const char *tag) {
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
  target_lock = 0;
  control_lock = 0xFF;
  memset(parens, 0, sizeof(parens));
  paren_noms(nom, parens[0], parens[1]);
  memset(frens, 0, sizeof(frens));
  memset(target, 0, sizeof(target));
  memset(partrait, 0, sizeof(partrait));

  seedrand();
}

void Parson::_to_pipe(Pipeline *pipe, unsigned int mbi) {
  assert(pipe->ctrlay->n == ncontrols * pipe->mbn);
  unsigned long dd3 = dim * dim * 3;
  assert(pipe->outlay->n == dd3 * pipe->mbn);

  dtobv(pipe->ctrbuf + mbi * ncontrols, controls, ncontrols);
  dtobv(pipe->outbuf + mbi * dd3, partrait, dd3);
}

void Parson::_from_pipe(Pipeline *pipe, unsigned int mbi) {
  unsigned long dd3 = dim * dim * 3;
  assert(pipe->outlay->n == dd3 * pipe->mbn);
  assert(sizeof(target) == dd3 * sizeof(double));

  unsigned long hashlen = pipe->ctxlay->n / pipe->mbn;
  assert(hashlen * pipe->mbn == pipe->ctxlay->n);
  assert(hashlen <= Hashbag::n);
  assert(hashlen * sizeof(double) <= sizeof(Hashbag));

  Hashbag ph;
  bagtags(&ph);
  memcpy(pipe->ctxbuf + mbi * hashlen, &ph, hashlen * sizeof(double));
  btodv(target, pipe->outbuf + mbi * dd3, dd3 * sizeof(double));

}

void Parson::generate(Pipeline *pipe, long min_age) {
  time_t now = time(NULL);
  if (min_age > 0) {
    if (now < generated + min_age) {
      return;
    }
  } else if (min_age < 0) {
    if (generated > 0) {
      return;
    }
  }
  
  assert(pipe->mbn == 1);
  _to_pipe(pipe, 0);

  pipe->ctrlock = 0;
  pipe->tgtlock = -1;
  pipe->reencode();

  _from_pipe(pipe, 0);
  generated = now;
}

void Parson::paste_target(PPM *ppm, unsigned int x0, unsigned int y0) {
  assert(x0 + dim < ppm->w);
  assert(y0 + dim < ppm->h);
  ppm->pastelab(target, dim, dim, x0, y0);
}

void Parson::paste_partrait(PPM *ppm, unsigned int x0, unsigned int y0) {
  assert(x0 + dim < ppm->w);
  assert(y0 + dim < ppm->h);
  ppm->pastelab(partrait, dim, dim, x0, y0);
}


}
