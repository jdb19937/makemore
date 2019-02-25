#define __MAKEMORE_URB_CC__ 1
#include <assert.h>
#include <stdio.h>

#include "urb.hh"
#include "pipeline.hh"

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
}

Urb::~Urb() {
  delete pipex;
  delete pipe1;

  for (Zone *zone : zones) {
    delete zone;
  }

  delete outgoing;
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

Parson *Urb::import(const Parson &x, unsigned int tier) {
  if (tier >= zones.size()) {
    _busout(x);
    return NULL;
  }

  assert(tier < zones.size());
  Zone *zone = zones[tier];

  bool eq;
  Parson q;
  Parson *p = zone->import(x, &eq, &q);

  if (!p)
    return NULL;
  if (eq)
    (void)import(q, tier + 1);

  return p;
}

}
