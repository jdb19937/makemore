#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>

#include <regex>

using namespace makemore;
using namespace std;

extern "C" void mainmore(
  Process *process
);

void mainmore(
  Process *process
) {
  if (process->args.size() == 0)
    process->coro->finish();

  try {
    string rxstr = joinwords(process->args);
    const std::regex rx(rxstr);

    while (strvec *inp = process->read()) {
      const strvec &in(*inp);
      const string instr = joinwords(in);

      if (std::regex_search(instr, rx)) {
        if (!process->write(in))
          break;
      }
    }
  } catch (std::regex_error) {
  }

  process->coro->finish();
}

