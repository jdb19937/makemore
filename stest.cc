#include <assert.h>
#include "server.hh"

using namespace makemore;

int main() {
  Server serv("urbs/crimea");
  serv.open();
  serv.bind(80, 443);
  serv.listen();
  serv.websockify(3333, "./certs");
  serv.setup();
  serv.main();
  serv.wait();
  return 0;
}
