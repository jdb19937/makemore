#include "brane.hh"
#include "strutils.hh"

using namespace makemore;
using namespace std;

int main(int argc, char **argv) {
  seedrand();
  setbuf(stdout, NULL);

  Brane brane("brane.proj", 1);
  Parson me("dan");
for (int i = 1; i < argc; ++i)
me.add_tag(argv[i]);

  string reqstr;
  while (read_line(stdin, &reqstr)) {
    string rspstr = brane.ask(&me, reqstr);
    if (*rspstr.c_str())
      printf("%s", rspstr.c_str());
  }

  return 0;
}
