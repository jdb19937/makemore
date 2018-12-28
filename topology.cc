#define __MAKEMORE_TOPOLOGY_CC__ 1
#include "topology.hh"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <netinet/in.h>
  
void Topology::save(FILE *fp) const {
  uint32_t tmp = htonl(wirings.size());
  assert(1 == fwrite(&tmp, 4, 1, fp));

  for (auto wi = wirings.begin(); wi != wirings.end(); ++wi) {
    Wiring *wire = *wi;
    wire->save(fp);
  }
}

void Topology::load(FILE *fp) {
  wirings.clear();

  uint32_t tmp;
  assert(1 == fread(&tmp, 4, 1, fp));
  tmp = ntohl(tmp);
  assert(tmp >= 1);
  wirings.resize(ntohl(tmp));

  Wiring *prev = NULL;
  for (auto wi = wirings.begin(); wi != wirings.end(); ++wi) {
    Wiring *wire = new Wiring;
    wire->load(fp);
    *wi = wire;

    if (prev) {
      assert(wire->inn == prev->outn);
    }

    prev = wire;
  }
}
