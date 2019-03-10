#include <server.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>

using namespace makemore;
using namespace std;

extern "C" bool mainmore(
  Server *, Urbite *, Process *, CommandState state, const strvec &
);

bool mainmore(
  Server *server,
  Urbite *who,
  Process *process,
  CommandState state,
  const strvec &args
) {
  if (state == COMMAND_STATE_RUNNING) {
    string colspec = process->pre[0];

    unsigned int col = strtoul(colspec.c_str(), NULL, 0);

    strvec outvec;
    outvec.resize(1);
    outvec[0] = col < args.size() ? args[col] : "";
    
    if (!process->put_out(outvec))
      return false;
  }

  return true;
}
