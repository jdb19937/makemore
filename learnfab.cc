#include "brane.hh"

using namespace makemore;
using namespace std;

int main() {
  seedrand();

  Brane brane("brane.proj", 8);

  int i = 0;
  while (1) {
    brane.burn(0.001);

    if (i % 100 == 0) {
      brane.report("learnfab");
      brane.save();
    }

    ++i;
  }

  return 0;
}
