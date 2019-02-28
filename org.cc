#define __MAKEMORE_ORG_CC__ 1
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "org.hh"
#include "random.hh"
#include "numutils.hh"
#include "zone.hh"

namespace makemore {

Org::Org() {
  clear();
}

Org::~Org() {

}

void Org::clear() {
  n = 0;
  member.clear();
}

void Org::add(Parson *p) {
  member.resize(n + 1);
  member[n] = p;
  ++n;
}

void Org::pick(Zone *zone, unsigned int k) {
  member.resize(k);
  n = k;
  
  for (unsigned int i = 0; i < n; ++i) {
    member[i] = zone->pick(16);
  }
}

}
