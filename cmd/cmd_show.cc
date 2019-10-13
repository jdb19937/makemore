#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include <string>
#include <set>
#include <vector>

#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>
#include <strutils.hh>
#include <parson.hh>
#include <zone.hh>
#include <server.hh>
#include <ppm.hh>
#include <imgutils.hh>

namespace makemore {

using namespace makemore;
using namespace std;

extern "C" void mainmore(Process *);

void mainmore(
  Process *process
) {
  const strvec &arg = process->args;

  Server *server = process->system->server;
  assert(server);
  Urb *urb = server->urb;
  assert(urb);

  strvec outvec;
  outvec.resize(arg.size());

  for (unsigned int argi = 0; argi < arg.size(); ++argi) {
    string nom = arg[argi];

    Parson *parson = urb->find(nom);
    if (!parson)
      return;

    parson->visit(1);

    unsigned int dim = 64;

    Superenc *enc = server->urb->enc;
    Supergen *gen = server->urb->get_gen(parson->gen);
    Styler *sty = server->urb->get_sty(parson->sty);

    Partrait genpar(256, 256);

    double *tmp = new double[Parson::ncontrols];
    memcpy(tmp, parson->controls, sizeof(double) * Parson::ncontrols);
    sty->tag_cholo["base"]->generate(tmp, tmp);
    gen->generate(tmp, &genpar);
    delete[] tmp;

    Partrait showpar(dim, dim);
    Pose pose = Pose::STANDARD;
    pose.center *= (double)dim / 256.0;
    pose.scale *= (double)dim / 256.0;

    showpar.set_pose(pose);
    genpar.warp(&showpar);

    std::string png;
    showpar.to_png(&png);

    outvec[argi] = png;
  }

  (void) process->write(outvec);
}

}
