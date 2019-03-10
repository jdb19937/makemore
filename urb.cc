#define __MAKEMORE_URB_CC__ 1
#include <assert.h>
#include <stdio.h>
#include <dirent.h>
#include <sys/types.h>

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

  pipe1 = new Pipeline((dir + "/partrait.proj").c_str(), 1);
  pipex = new Pipeline((dir + "/partrait.proj").c_str(), mbn);

  brane1 = new Brane((dir + "/brane.proj").c_str(), 1);

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
  assert(images.size());
}

Urb::~Urb() {
  delete pipex;
  delete pipe1;

  for (Zone *zone : zones) {
    delete zone;
  }

  delete outgoing;
}

Parson *Urb::make(const std::string &nom, unsigned int tier) {
  if (!Parson::valid_nom(nom))
    return NULL;

  string imagefn = images[randuint() % images.size()];
  string png = slurp(imagefn);

  Parson parson;
  memset(&parson, 0, sizeof(Parson));

  vector<string> tags;
  // imglab("png", png, 64, 64, parson.target, &tags);
  pnglab(png, 64, 64, parson.target, &tags);
  for (auto tag : tags) {
    parson.add_tag(tag.c_str());
  }

  time_t now = time(NULL);
  parson.creator = 0x7F000001;
  parson.revisor = 0x7F000001;
  parson.created = now;
  parson.revised = now;
  parson.visited = now;

  strcpy(parson.nom, nom.c_str());
  return this->make(parson, tier);
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
  p->generate(pipe1, min_age);
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
