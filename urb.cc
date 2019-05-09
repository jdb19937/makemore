#define __MAKEMORE_URB_CC__ 1
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "urb.hh"
#include "pipeline.hh"
#include "strutils.hh"
#include "imgutils.hh"

namespace makemore {

using namespace std;

Urb::Urb(const char *_dir, unsigned int _mbn) {
  mbn = _mbn;

  assert(strlen(_dir) < 4000);
  dir = _dir;

  Zone *main = new Zone(dir + "/main.zone");
  zones.push_back(main);

  outgoing = new Bus(dir + "/outgoing.bus");

  egd = new Encgendis(dir + "/egd.proj", 1);

  cholo = new Cholo(egd->ctrlay->n);
  cholo->load(dir + "/egd.proj/cholo.dat");


  images.clear();
  DIR *dp = opendir((dir + "/images").c_str());
  assert(dp);
  struct dirent *de;
  while ((de = readdir(dp))) {
    if (*de->d_name == '.')
      continue;
    images.push_back(dir + "/images/" + de->d_name);
  }
  closedir(dp);
  sort(images.begin(), images.end());
  assert(images.size());

  srcimages.clear();
  dp = opendir((dir + "/srcimages").c_str());
  assert(dp);
  while ((de = readdir(dp))) {
    if (*de->d_name == '.')
      continue;
    srcimages.push_back(dir + "/srcimages/" + de->d_name);
  }
  closedir(dp);
  sort(srcimages.begin(), srcimages.end());
  assert(srcimages.size());

  {
    std::string home = dir + "/home";
    int ret = ::mkdir(home.c_str(), 0700);
    assert(ret == 0 || ret == -1 && errno == EEXIST);
  }

#if 1
  {
    std::string sampfn = dir + "/egd.proj/samp.dat";
    FILE *sampfp = fopen(sampfn.c_str(), "r");
    assert(sampfp);
    assert(0 == fseek(sampfp, 0, 2));

    nsamp = ftell(sampfp);
    assert(nsamp % (sizeof(double) * egd->ctrlay->n) == 0);
    nsamp /= egd->ctrlay->n * sizeof(double);
    assert(0 == fseek(sampfp, 0, 0));

    samp = new double[egd->ctrlay->n * nsamp];
    int ret = fread(samp, egd->ctrlay->n * sizeof(double), nsamp, sampfp);
    assert(ret == nsamp);
    fclose(sampfp);
  }
#endif

  {
    std::string framefn = dir + "/egd.proj/frame.dat";
    FILE *framefp = fopen(framefn.c_str(), "r");
    assert(framefp);
    assert(0 == fseek(framefp, 0, 2));

    nframe = ftell(framefp);
    assert(nframe % (sizeof(double) * 6) == 0);
    nframe /= 6 * sizeof(double);
    assert(0 == fseek(framefp, 0, 0));

    frame = new double[6 * nframe];
    int ret = fread(frame, 6 * sizeof(double), nframe, framefp);
    assert(ret == nframe);
    fclose(framefp);
  }

}

Urb::~Urb() {
  delete egd;

  for (Zone *zone : zones) {
    delete zone;
  }

  delete outgoing;
}

Parson *Urb::make(const std::string &nom, unsigned int tier, unsigned int gens, Parson *child, unsigned int which) {

  if (!Parson::valid_nom(nom))
    return NULL;

  if (Parson *x = find(nom)) {
fprintf(stderr, "found %s\n", nom.c_str());
    if (!*x->parens[0] || !*x->parens[1]) {
      seedrand(Parson::hash_nom(nom.c_str()));
      x->paren_noms(nom.c_str(), x->parens[0], x->parens[1]);
    }

    if (gens) {
      (void) make(x->parens[0], tier, gens - 1, x, 0);
      (void) make(x->parens[1], tier, gens - 1, x, 1);
    }
    return x;
  }

  Parson parson;
  memset(&parson, 0, sizeof(Parson));
  strcpy(parson.nom, nom.c_str());

  seedrand(Parson::hash_nom(nom.c_str()));
  parson.paren_noms(nom.c_str(), parson.parens[0], parson.parens[1]);

  if (nom.length() >= 5 && !strncmp(nom.c_str(), "anti", 4)) {
    Parson *x = this->make(nom.c_str() + 4, tier);
    for (unsigned int j = 0; j < egd->ctrlay->n; ++j) {
      parson.controls[j] = -x->controls[j];
    }
  } else {
    for (unsigned int j = 0; j < egd->ctrlay->n; ++j) {
      unsigned int k = randuint() % nsamp;
      parson.controls[j] = samp[k * egd->ctrlay->n + j] * 1.0;
    }
  }

  if (child) {
    seedrand(Parson::hash_nom(child->nom, 93));
    for (unsigned int j = 0; j < egd->ctrlay->n; ++j) {
      if (randuint() % 2 == which) {
        parson.controls[j] = child->controls[j];
      }
    }
  }

  if (gens) {
    (void) make(parson.parens[0], tier, gens - 1, &parson, 0);
    (void) make(parson.parens[1], tier, gens - 1, &parson, 1);
  }

fprintf(stderr, "%s -> %s, %s\n", nom.c_str(), parson.parens[0], parson.parens[1]);

#if 0
  cholo->generate(parson.controls, egd->ctrbuf);

  for (unsigned int j = 0; j < egd->ctrlay->n; ++j)
    egd->ctrbuf[j] = sigmoid(egd->ctrbuf[j]);

  egd->generate();

  assert(egd->tgtlay->n == Parson::dim * Parson::dim * 3);
  labquant(egd->tgtbuf, Parson::dim * Parson::dim * 3, parson.target);
#endif

#if 0
  string imagefn = images[randuint() % images.size()];
  string png = slurp(imagefn);

  vector<string> tags;
  // imglab("png", png, 64, 64, parson.target, &tags);
  pnglab(png, 64, 64, parson.target, &tags);
  for (auto tag : tags) {
    parson.add_tag(tag.c_str());
  }
#endif

  time_t now = time(NULL);
  parson.creator = 0x7F000001;
  parson.revisor = 0x7F000001;
  parson.created = now;
  parson.revised = now;
  parson.visited = now;

  Parson *x = this->make(parson, tier);

  return x;
}


Parson *Urb::make(unsigned int tier) {
  string imagefn = images[randuint() % images.size()];
  string png = slurp(imagefn);

  Parson parson;
  memset(&parson, 0, sizeof(Parson));

  vector<string> tags;
  // imglab("png", png, 64, 64, parson.target, &tags);
  pnglab(png, 64, 64, parson.target, &tags);

  bool gender = 1;
  for (auto tag : tags) {
    if (tag == "female")
      gender = 0;
    parson.add_tag(tag.c_str());
  }

  string nom;
  do {
    nom = Parson::gen_nom(gender);
  } while (find(nom));

  time_t now = time(NULL);
  parson.creator = 0x7F000001;
  parson.revisor = 0x7F000001;
  parson.created = now;
  parson.revised = now;
  parson.visited = now;

  strcpy(parson.nom, nom.c_str());
  return this->make(parson, tier);
}



void Urb::restock(unsigned int n, vector<string> *noms) {
  for (unsigned int i = 0; i < n; ++i) {
    Parson *parson = make();
    assert(parson);
    if (noms)
      noms->push_back(parson->nom);
  }
}

    

void Urb::generate(Parson *p, long min_age) {
//  p->generate(pipe1, min_age);
}

Parson *Urb::find(const std::string &nom, unsigned int *tierp) const {

  for (unsigned int tier = 0, tiers = zones.size(); tier < tiers; ++tier) {
    Zone *zone = zones[tier];

    if (Parson *p = zone->find(nom)) {
      if (tierp)
        *tierp = tier;
      return p;
    }
  }

  return NULL;
}

unsigned int Urb::tier(const Zone *zone) const {
  for (unsigned int tier = 0, tiers = zones.size(); tier < tiers; ++tier) {
    if (zones[tier] == zone)
      return tier;
  }
  assert(0);
}

unsigned int Urb::tier(const Parson *p) const {
  for (unsigned int tier = 0, tiers = zones.size(); tier < tiers; ++tier) {
    Zone *zone = zones[tier];
    if (zone->has(p))
      return tier;
  }
  assert(0);
}

Zone *Urb::zone(const Parson *p) const {
  for (unsigned int tier = 0, tiers = zones.size(); tier < tiers; ++tier) {
    Zone *zone = zones[tier];
    if (zone->has(p))
      return zone;
  }
  return NULL;
}

bool Urb::deport(const char *nom) {
  if (Parson *p = find(nom)) {
    deport(p);
    return true;
  } else {
    return false;
  }
}

void Urb::deport(Parson *x) {
  assert(x->exists());
  _busout(*x);
  memset(x, 0, sizeof(Parson));
}

void Urb::_busout(const Parson &x) {
  outgoing->add(x);
}

Parson *Urb::make(const Parson &x, unsigned int tier) {
  if (tier >= zones.size()) {
    _busout(x);
    return NULL;
  }

  assert(tier < zones.size());
  Zone *zone = zones[tier];

  bool eq;
  Parson q;
  Parson *p = zone->make(x, &eq, &q);

  if (!p)
    return NULL;
  if (eq)
    (void)make(q, tier + 1);

  return p;
}

}
