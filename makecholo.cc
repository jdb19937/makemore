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

  double *tmp = new double[Parson::ncontrols];
  Cholo cholo(Parson::ncontrols);

  Zone *zone = new Zone(argv[1]);

  int n = 0;
  for (unsigned int i = 0; i < zone->n; ++i) {
    Parson *prs = zone->db + i;
    if (!prs->revised)
      continue;

#if 1
    bool all_tags = true;
    bool any_tags = false;

    for (unsigned int argi = 2; argi < argc; ++argi) {
      if (!prs->has_tag(argv[argi])) {
        all_tags = false;
        break;
      }
    }
    for (unsigned int argi = 2; argi < argc; ++argi) {
      if (prs->has_tag(argv[argi])) {
        any_tags = true;
        break;
      }
    }
//    if (!all_tags)
//      continue;
    if (argc > 2 && !any_tags)
      continue;
#endif

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
