#include <system.hh>
#include <process.hh>
#include <server.hh>
#include <agent.hh>

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

  if (process->args.size() != 3) {
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

  string msg0 = hdr + process->args[1];
  if (msg0.length() != 1024) {
    strvec outvec;
    outvec.resize(1);
    outvec[0] = "nope5";
    process->write(outvec);
    return;
  }
  string msg1 = hdr + process->args[2];
  if (msg1.length() != 1024) {
    strvec outvec;
    outvec.resize(1);
    outvec[0] = "nope6";
    process->write(outvec);
    return;
  }

  ufrom.make_home_dir();

  string ufromfn = urb->dir + "/home/" + ufrom.nom + "/" + uto.nom + ".dat";
  FILE *fp = fopen(ufromfn.c_str(), "a");
  if (!fp) {
    strvec outvec;
    outvec.resize(1);
    outvec[0] = "nope7";
    process->write(outvec);
    return;
  }
  size_t ret = fwrite(msg1.data(), 1, 1024, fp);
  if (ret != 1024) {
    strvec outvec;
    outvec.resize(1);
    outvec[0] = "nope8";
    process->write(outvec);
    return;
  }
  fclose(fp);

  // touch atime
  fp = fopen(ufromfn.c_str(), "r");
  char c;
  fread(&c, 1, 1, fp);
  if (fp)
    fclose(fp);

  string utofn = urb->dir + "/home/" + uto.nom + "/" + ufrom.nom + ".dat";
  if (uto.nom != ufrom.nom) {
    uto.make_home_dir();
    fp = fopen(utofn.c_str(), "a");
    if (!fp) {
      strvec outvec;
      outvec.resize(1);
      outvec[0] = "nope9";
      process->write(outvec);
      return;
    }
    ret = fwrite(msg0.data(), 1, 1024, fp);
    if (ret != 1024) {
      strvec outvec;
      outvec.resize(1);
      outvec[0] = "nope10";
      process->write(outvec);
      return;
    }
    fclose(fp);

    if (!*to->owner) {
      std::string br = "fakeresp " + ufrom.nom;
      to->pushbrief(br);
    }
  }

  {
    strvec notify;
    notify.resize(3);
    notify[0] = "from";
    notify[1] = ufrom.nom;
    notify[2] = msg0;

    if (server->notify(uto.nom, notify)) {
      FILE *fp = fopen(utofn.c_str(), "r");
      char c = getc(fp);
      if (fp)
        fclose(fp);
    } else {
      to->newcomms = 1;
    }
  }

  strvec outvec;
  outvec.resize(1);
  outvec[0] = "ok";
  process->write(outvec);
}

}
