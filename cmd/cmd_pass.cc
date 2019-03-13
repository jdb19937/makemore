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
  if (process->args.size() > 1) {
    strvec outvec;
    outvec.resize(1);
    outvec[0] = "nope";
    process->write(outvec);
    return;
  }

  if (!process->session->who)
    return;

  Parson *parson = process->session->who->parson();
  if (!parson)
    return;

  string pass; 
  if (process->args.size())
    pass = process->args[0];

  if (pass.length() > 79) {
    strvec outvec;
    outvec.resize(1);
    outvec[0] = "nope";
    process->write(outvec);
    return;
  }

  parson->set_pass(pass);

  strvec outvec;
  outvec.resize(1);
  outvec[0] = "ok";

  process->write(outvec);
}

}
