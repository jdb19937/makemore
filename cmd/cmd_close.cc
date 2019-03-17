#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>
#include <session.hh>

using namespace makemore;

extern "C" void mainmore(Process *);

void mainmore(
  Process *process
) {
  unsigned int fd = 0;
  if (process->args.size() > 0)
    fd = strtoul(process->args[0].c_str(), NULL, 0);

  Session *session = process->session;
  if (!session)
    return;

  Process *shell = session->shell;
  if (!shell)
    return;

  if (fd == 0)
    return;

  if (fd >= shell->otab.size())
    return;
  IO *to = shell->otab[fd];
  if (!to)
    return;

  to->unlink_writer(shell);
  shell->otab[fd] = NULL;
}
