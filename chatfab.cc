#include "brane.hh"
#include "strutils.hh"

using namespace makemore;
using namespace std;

int main() {
  seedrand();
  setbuf(stdout, NULL);

  Brane brane("brane.proj", 1);
  Parson me("dan");

  string reqstr;
  while (read_line(stdin, &reqstr)) {
    string rspstr = brane.ask(&me, reqstr);
    if (*rspstr.c_str())
      printf("%s", rspstr.c_str());
  }

  return 0;
}
