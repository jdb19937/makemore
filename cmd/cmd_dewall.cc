#include <system.hh>
#include <process.hh>
#include <server.hh>
#include <agent.hh>

#include <wall.hh>

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

  if (process->args.size() != 2) {
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

  string idstr = process->args[0];
  unsigned int id = strtoul(idstr.c_str(), NULL, 0);

  string hash = process->args[1];
#if 0
  if (hash.length() != 32) {
    strvec outvec;
    outvec.resize(1);
    outvec[0] = "nope4";
    process->write(outvec);
    return;
  }
#endif

  ufrom.make_home_dir();
  string wallfn = urb->dir + "/home/" + ufrom.nom + "/wall.txt";
  Wall wall;
  wall.load(wallfn);

  bool erased = wall.erase(id); // , hash
  if (!erased) {
    strvec outvec;
    outvec.resize(1);
    outvec[0] = "nope5";
    process->write(outvec);
    return;
  }

  wall.save(wallfn);

  strvec outvec;
  outvec.resize(1);
  outvec[0] = "ok";
  process->write(outvec);
}

}
