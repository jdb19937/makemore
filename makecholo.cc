#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>

#include "strutils.hh"
#include "encgendis.hh"
#include "ppm.hh"
#include "urb.hh"
#include "zone.hh"
#include "cudamem.hh"
#include "numutils.hh"
#include "imgutils.hh"
#include "cholo.hh"

using namespace makemore;
using namespace std;

int main() {
  Zone *zone = new Zone("/spin/dan/celeba.dat");
  Cholo cholo(Parson::ncontrols);

  for (unsigned int i = 0; i < zone->n; ++i) {
    Parson *prs = zone->db + i;
    cholo.observe(prs->controls);
    fprintf(stderr, "i=%d\n", i);
  }

  cholo.finalize();
  cholo.save(stdout);
}
