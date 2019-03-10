#include <server.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>

using namespace makemore;

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
fprintf(stderr, "here echo pre=%s args=%s state=%d\n", joinwords(process->pre).c_str(), joinwords(args).c_str(), state);

  if (state == COMMAND_STATE_RUNNING) {
    strvec allargs = process->pre;
    catstrvec(allargs, args);
    
    if (!process->put_out(allargs))
      return false;
  }

  return true;
}
