#include <set>
#include <vector>

#include <process.hh>
#include <parson.hh>
#include <server.hh>
#include <urb.hh>
#include <strutils.hh>

using namespace makemore;
using namespace std;

extern "C" void mainmore(Process *);

void mainmore(
  Process *process
) {
  set<string> tags;
  for (auto tag : process->args)
    tags.insert(tag);

  Server *server = process->system->server;
  Urb *urb = server->urb;

  string nom;
  do {
    if (tags.count("female")) {
      nom = Parson::gen_nom(false);
    } else if (tags.count("male")) {
      nom = Parson::gen_nom(true);
    } else {
      nom = Parson::gen_nom();
    }
  } while (urb->find(nom));

  assert(Parson::valid_nom(nom.c_str()));

  strvec outvec;
  outvec.resize(1);
  outvec[0] = nom;
  process->write(outvec);
}

