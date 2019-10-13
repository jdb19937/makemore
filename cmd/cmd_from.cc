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
  Urbite &uto = *session->who;

  Parson *to = uto.parson();
  std::string tonom = to->nom;
  if (!to) {
    strvec outvec;
    outvec.resize(1);
    outvec[0] = "nope3";
    process->write(outvec);
    return;
  }

  string fromnom = process->args[0];
  Parson *from = urb->find(fromnom);
  if (!from) {
    strvec outvec;
    outvec.resize(1);
    outvec[0] = "nope4";
    process->write(outvec);
    return;
  }
  Urbite ufrom(fromnom, urb, from);

  string utofn = urb->dir + "/home/" + uto.nom + "/" + ufrom.nom + ".dat";
  FILE *fp = fopen(utofn.c_str(), "r+");
  if (!fp) {
    strvec outvec;
    outvec.resize(1);
    outvec[0] = "nope9";
    process->write(outvec);
    return;
  }

#if 0
  fseek(fp, 0, 2);
  off_t ftop = ftell(fp);
  if (ftop > 32768) {
    char *buf = new char[32768];
    fseek(fp, ftop - 32768, 0);
    fread(buf, 1, 32768, fp);
    fseek(fp, 0, 0);
    fwrite(buf, 1, 32768, fp);
    fflush(fp);
    ftruncate(fileno(fp), 32768);
    delete[] buf;
  }
  fseek(fp, 0, 0);
#endif

  while (1) {
    uint8_t buf[1024];
    size_t ret = fread(buf, 1, 1024, fp);
    if (ret == 0) {
      break;
    }
    if (ret != 1024) {
      strvec outvec;
      outvec.resize(1);
      outvec[0] = "nope10";
      process->write(outvec);
      return;
    }

    strvec outvec;
    outvec.resize(3);
    outvec[0] = "from";
    outvec[1] = ufrom.nom;
    outvec[2] = std::string((char *)buf, 1024);
    process->write(outvec);
  }
  fclose(fp);
}

}
