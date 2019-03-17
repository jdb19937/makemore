#include <string>

#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>
#include <session.hh>

using namespace makemore;
using namespace std;

extern "C" void mainmore(Process *);


void mainmore(
  Process *process
) {
  if (process->args.size() == 0)
    return;

  Session *session = process->session;
  if (!session)
    return;

  Process *shell = session->shell;
  if (!shell) 
    return;
  unsigned int otabn = shell->otab.size();
  assert(otabn >= 1);

  string ncmd = process->args[0];
  strvec nargs = strvec(process->args.begin() + 1, process->args.end());

  Process *child = new Process(
    process->system, process->session, ncmd, nargs,
    NULL, shell->otab[0]
  );

  shell->otab.resize(otabn + 1);
  shell->otab[otabn] = child->itab[0];
  shell->otab[otabn]->link_writer(shell);

  {
    char buf[32];
    sprintf(buf, "%u", otabn);
    strvec outvec;
    outvec.resize(1);
    outvec[0] = buf;
    (void) process->write(outvec);
  }
}

