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
#if 0
  if (Agent *agent = process->outagent) {
    agent->close();
  }
#endif
}

}
