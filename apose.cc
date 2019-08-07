#include "imgutils.hh"
#include "strutils.hh"
#include "partrait.hh"
#include "parson.hh"
#include "catalog.hh"
#include "autoposer.hh"

using namespace makemore;
using namespace std;

int main(int argc, char **argv) {
  assert(argc > 0);
  Catalog cat(argv[1]);
  Autoposer ap("fineposer.proj");

  for (auto fn : cat.fn) {
    Partrait par;
    par.load(fn);

    Triangle mark;
    mark.p = Point(192, 240);
    mark.q = Point(320, 240);
    mark.r = Point(256, 384);
    par.set_mark(mark);

    ap.autopose(&par);
    ap.autopose(&par);

    Pose pose = par.get_pose();
    par.set_tag("angle", pose.angle);
    par.set_tag("stretch", pose.stretch);
    par.set_tag("skew", pose.skew);

    par.save(fn);
    fprintf(stderr, "%s\n", fn.c_str());
  }

  return 0;
}
