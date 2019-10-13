#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include <string>
#include <set>
#include <vector>

#include <system.hh>
#include <urbite.hh>
#include <session.hh>
#include <process.hh>
#include <strutils.hh>
#include <parson.hh>
#include <zone.hh>
#include <server.hh>
#include <ppm.hh>
#include <imgutils.hh>

namespace makemore {

using namespace makemore;
using namespace std;

extern "C" void mainmore(Process *);

void mainmore(
  Process *process
) {
  strvec outvec;
  outvec.resize(1);
  outvec[0] = process->session->who->nom;
  process->write(outvec);
}

}
