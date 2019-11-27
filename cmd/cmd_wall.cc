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

  string txt = process->args[0];
  txt += "\n";

  if (txt.length() > 16384) {
    strvec outvec;
    outvec.resize(1);
    outvec[0] = "nope4";
    process->write(outvec);
    return;
  }

  ufrom.make_home_dir();

  string wallfn = urb->dir + "/home/" + ufrom.nom + "/wall.txt";
  FILE *fp = fopen(wallfn.c_str(), "a");
  if (!fp) {
    strvec outvec;
    outvec.resize(1);
    outvec[0] = "nope7";
    process->write(outvec);
    return;
  }

  size_t ret = fwrite(txt.data(), 1, txt.length(), fp);
  if (ret != txt.length()) {
    strvec outvec;
    outvec.resize(1);
    outvec[0] = "nope8";
    process->write(outvec);
    return;
  }
  fclose(fp);

  Wall wall;
  wall.load(wallfn);
  if (wall.posts.size() > 8) {
    wall.truncate(8);
    wall.save(wallfn);
  }

  strvec outvec;
  outvec.resize(1);
  outvec[0] = "ok";
  process->write(outvec);
}

}
