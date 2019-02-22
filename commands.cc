#define __MAKEMORE_COMMANDS_CC__ 1
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include <string>
#include <vector>

#include "commands.hh"
#include "strutils.hh"
#include "parson.hh"
#include "server.hh"
#include "ppm.hh"

namespace makemore {

using namespace std;
  
bool cmd_echo(CMD_ARGS) {
  for (auto argi = args.begin(); argi != args.end(); ++argi) {
    fprintf(outfp, "%s%s", (argi == args.begin()) ? "" : " ", argi->c_str());
    fprintf(stderr, "%s%s", (argi == args.begin()) ? "" : " ", argi->c_str());
  }
  fprintf(outfp, "\n");
  fprintf(stderr, "\n");
  return true;
}

bool cmd_GET(CMD_ARGS) {
  if (args.size() != 2)
    return false;
  if (strncmp(args[1].c_str(), "HTTP/1.", 7))
    return false;

  std::string rsp = "test\n";
  bool retval = true;
  if (args[1] == "HTTP/1.0")
    retval = false;

  while (1) {
    string line;
    if (!read_line(infp, &line))
      return false;
    if (line == "\r" || line == "")
      break;
  }
  
  fprintf(outfp, "%s 200 OK\r\n", args[1].c_str());
  fprintf(outfp, "Content-Type: text/plain\r\n");
  fprintf(outfp, "Content-Length: %lu\r\n", rsp.length());
  fprintf(outfp, "\r\n");
  fprintf(outfp, "%s", rsp.c_str());

  return retval;
}

bool cmd_ppm(CMD_ARGS) {
  Urb *urb = server->urb;

  if (args.size() != 1)
    return false;
  const char *nom = args[0].c_str();

  Parson *parson = urb->zone->find(nom);
  if (!parson)
    return false;

  urb->generate(parson);

  PPM ppm(64, 64);
  parson->paste_partrait(&ppm);
  ppm.write(outfp);
}

}
