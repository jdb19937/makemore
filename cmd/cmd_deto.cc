#include <system.hh>
#include <process.hh>
#include <server.hh>
#include <agent.hh>
#include <unistd.h>

namespace makemore {

using namespace makemore;
using namespace std;

extern "C" void mainmore(Process *);

void mainmore(
  Process *process
) {
  Session *session = process->session;
  Server *server = process->system->server;
  Urb *urb = server->urb;

  if (process->args.size() != 1) {
    strvec outvec;
    outvec.resize(1);
    outvec[0] = "nope1";
    process->write(outvec);
    return;
  }

  if (!session->who) {
    strvec outvec;
    outvec.resize(1);
    outvec[0] = "nope2";
    process->write(outvec);
    return;
  }
  Urbite &ufrom = *session->who;

  Parson *from = ufrom.parson();
  if (!from) {
    strvec outvec;
    outvec.resize(1);
    outvec[0] = "nope3";
    process->write(outvec);
    return;
  }
  std::string fromnom = from->nom;

  from->acted = time(NULL);

  string tonom = process->args[0];
  Parson *to = urb->find(tonom);
  if (!to) {
    strvec outvec;
    outvec.resize(1);
    outvec[0] = "nope4";
    process->write(outvec);
    return;
  }
  Urbite uto(tonom, urb, to);

  char chdr[64];
  memset(chdr, 0, 64);
  memcpy(chdr, fromnom.c_str(), fromnom.length());
  memcpy(chdr + 32, tonom.c_str(), tonom.length());
  string hdr(chdr, 64);

  ufrom.make_home_dir();

  string ufromfn = urb->dir + "/home/" + ufrom.nom + "/" + uto.nom + ".dat";
  (void) unlink(ufromfn.c_str());

  strvec outvec;
  outvec.resize(1);
  outvec[0] = "ok";
  process->write(outvec);
}

}
