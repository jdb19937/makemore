#include <server.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>

using namespace makemore;

extern "C" bool mainmore(Server *, Urbite *, Process *, CommandState state, const strvec &);

bool mainmore(
  Server *server,
  Urbite *who,
  Process *process,
  CommandState state,
  const strvec &args
) {
  static strvec no_args;

  if (state == COMMAND_STATE_INITIALIZE) {
    Command shfunc = find_command("sh");
    assert(shfunc);

fprintf(stderr, "args=[%s]\n", joinwords(process->pre).c_str());

    Process *child = server->add_process(
      who, shfunc, process->pre, process->out_type, process->out
    );

    child->add_process_ref(process);

    process->state = (void *)child;
    return true;
  }

  if (state == COMMAND_STATE_RUNNING) {
    return process->put_out(args);
  }

  if (state == COMMAND_STATE_SHUTDOWN) {
    Process *child = (Process *)process->state;
    assert(child);

    if (child->put_in(no_args)) {
      child->remove_process_ref(process);
      process->state = NULL;
      return true;
    } else {
      return false;
    }
  }
}
