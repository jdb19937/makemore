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

fprintf(stderr, "in fakeresp\n");

  Command cmdto = find_command("to");

  time_t now = time(NULL);
  char buf[960];
  memset(buf, 0, 960);
  memcpy(buf, &now, sizeof(now));
  process->args.push_back(std::string(buf, 960));
  process->args.push_back(std::string(buf, 960));

  cmdto(process);
}

}
