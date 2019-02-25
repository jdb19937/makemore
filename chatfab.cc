#include "brane.hh"
#include "strutils.hh"

using namespace makemore;
using namespace std;

int main() {
  seedrand();
  setbuf(stdout, NULL);

  Brane brane("brane.proj", 1);

  string reqstr;
  while (read_line(stdin, &reqstr)) {
    string rspstr = brane.ask(reqstr);
    printf("%s\n", rspstr.c_str());
  }

  return 0;
}
