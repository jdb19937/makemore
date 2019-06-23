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

int main(int argc, char **argv) {
  assert(argc > 1);
  Zone *zone = new Zone(argv[1]);
  Cholo cholo(Parson::ncontrols);

  double *tmp = new double[Parson::ncontrols];

  int n = 0;
  for (unsigned int i = 0; i < zone->n; ++i) {
    Parson *prs = zone->db + i;
    if (!prs->revised)
      continue;

    bool all_tags = true;
    for (unsigned int argi = 2; argi < argc; ++argi) {
      if (!prs->has_tag(argv[argi])) {
        all_tags = false;
        break;
      }
    }
    if (!all_tags)
      continue;

//if (prs->recon_err > 0.035)
//  continue;

    memcpy(tmp, prs->controls, Parson::ncontrols * sizeof(double));
    cholo.observe(tmp);
    ++n;
  }
  fprintf(stderr, "n=%d\n", n);

  cholo.finalize();
  cholo.save(stdout);
}
