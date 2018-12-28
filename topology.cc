#define __MAKEMORE_TOPOLOGY_CC__ 1
#include "topology.hh"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <netinet/in.h>
  
void Topology::save(FILE *fp) const {
  assert(layouts.size() == 1 + wirings.size());
  uint32_t tmp = htonl(layouts.size());
  assert(1 == fwrite(&tmp, 1, 4, fp));

  for (auto li = layouts.begin(); li != layouts.end(); ++li) {
    Layout *lay = *li;
    lay->save(fp);
  }

  for (auto wi = wirings.begin(); wi != wirings.end(); ++wi) {
    Wiring *wire = *wi;
    wire->save(fp);
  }
}

void Topology::load(FILE *fp) {
  layouts.clear();
  wirings.clear();

  uint32_t tmp;
  assert(1 == fread(&tmp, 1, 4, fp));
  tmp = ntohl(tmp);
  assert(tmp >= 1);
  layouts.resize(ntohl(tmp));
  wirings.resize(layouts.size() - 1);

  for (auto li = layouts.begin(); li != layouts.end(); ++li) {
    Layout *lay = new Layout;
    lay->load(fp);
    *li = lay;
  }

  auto li = layouts.begin();
  for (auto wi = wirings.begin(); wi != wirings.end(); ++wi) {
    Wiring *wire = new Wiring;
    wire->load(fp);
    *wi = wire;

    assert(wire->inn == (*li)->n);
    ++li;
    assert(wire->outn == (*li)->n);
  }
}
