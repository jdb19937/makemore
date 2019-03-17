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

  if (fd >= shell->otab.size())
    return;
  IO *to = shell->otab[fd];
  if (!to)
    return;


fprintf(stderr, "proc=%lu shell=%lu\n", (long)process, (long)shell);
  if (process != shell) {
    unsigned int otabn = process->otab.size();
    process->otab.resize(otabn + 1);
    process->otab[otabn] = shell->otab[fd];
    process->otab[otabn]->link_writer(process);
    fd = otabn;
  }

  if (process->args.size() > 1) {
    strvec line(process->args.begin() + 1, process->args.end());
    (void) process->write(line, fd);
  } else {
    while (Line *line = process->read()) {
      if (!process->write(line, fd)) {
        delete line;
        break;
      }
    }
  }

  if (process != shell) {
    process->otab[fd]->unlink_writer(process);
    process->otab[fd] = NULL;
  }
}
