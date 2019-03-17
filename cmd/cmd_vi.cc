#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>
#include <word.hh>

using namespace makemore;

extern "C" void mainmore(Process *);

void mainmore(
  Process *process
) {

  Line *inl;
  while (1) {
    Line *inl = process->read();
    if (!inl)
      break;

    if (!process->write(inl)) {
      delete inl;
      break;
    }
  }

  fprintf(stderr, "finishing vi\n");
}
