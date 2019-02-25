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
#include "org.hh"
#include "zone.hh"
#include "server.hh"
#include "ppm.hh"

namespace makemore {

int _startup_count = 0;

using namespace std;

NEW_CMD(exit) {
  return false;
}
  
NEW_CMD(echo) {
  for (auto argi = args.begin(); argi != args.end(); ++argi) {
    fprintf(outfp, "%s%s", (argi == args.begin()) ? "" : " ", argi->c_str());
    fprintf(stderr, "%s%s", (argi == args.begin()) ? "" : " ", argi->c_str());
  }
  fprintf(outfp, "\n");
  fprintf(stderr, "\n");
  return true;
}

NEW_CMD(GET) {
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

  if (args[0] == "/grid.jpg") {
    fprintf(outfp, "%s 200 OK\r\n", args[1].c_str());
    fprintf(outfp, "Content-Type: image/jpeg\r\n");
    fprintf(outfp, "\r\n");

    std::vector<std::string> gridargs;
    gridargs.push_back("8");
    gridargs.push_back("8");
bool cmd_grid(CMD_ARGS);
    cmd_grid(server, "grid", gridargs, infp, outfp);
    return false;
  }

  fprintf(outfp, "%s 200 OK\r\n", args[1].c_str());
  fprintf(outfp, "Content-Type: text/plain\r\n");
  fprintf(outfp, "Content-Length: %lu\r\n", rsp.length());
  fprintf(outfp, "\r\n");
  fprintf(outfp, "%s", rsp.c_str());

  return retval;
}

NEW_CMD(ppm) {
  Urb *urb = server->urb;

  if (args.size() != 1)
    return false;
  const char *nom = args[0].c_str();

  Parson *parson = urb->find(nom);
  if (!parson)
    return false;

  urb->generate(parson);

  PPM ppm(64, 64);
  parson->paste_partrait(&ppm);
  ppm.write(outfp);
}

NEW_CMD(jpeg) {
  Urb *urb = server->urb;

  if (args.size() != 1)
    return false;
  const char *nom = args[0].c_str();

  Parson *parson = urb->find(nom);
  if (!parson)
    return false;

  urb->generate(parson);

  PPM ppm(128, 64);
  parson->paste_partrait(&ppm, 0, 0);
  parson->paste_target(&ppm, 64, 0);

  std::string jpeg;
  ppm.write_jpeg(&jpeg);
  size_t ret = fwrite(jpeg.data(), 1, jpeg.length(), outfp);
  if (ret != jpeg.length())
    return false;
  return true;
}

NEW_CMD(import) {
  Urb *urb = server->urb;

  unsigned int n = (unsigned int)-1;
  if (args.size() != 0) {
    if (args.size() != 1)
      return false;
    n = (unsigned int)atoi(args[0].c_str());
  }

  for (unsigned int i = 0; i < n; ++i) {
    Parson parson;
    if (!parson.load(infp))
      return false;
    if (!parson.nom[0])
      return false;

    Parson *p = urb->import(parson);
    if (!p)
      return false;

    fprintf(outfp, "imported %s to tier %u\n", p->nom, urb->tier(p));
  }

  return true;
}

NEW_CMD(grid) {
  Urb *urb = server->urb;

  if (args.size() != 2)
    return false;
  unsigned int w = (unsigned int)atoi(args[0].c_str());
  unsigned int h = (unsigned int)atoi(args[1].c_str());
  if (w > 32 || h > 32)
    return false;
  unsigned int wh = w * h;

  Org org;
  org.pick(urb->zones[0], wh);

  urb->pipex->load();
  urb->pipex->generate(&org);

  org.sort_centerv();

  PPM ppm(64 * w, 64 * h);
  for (unsigned int x = 0; x < w; ++x)
    for (unsigned int y = 0; y < h; ++y)
      org.member[y * w + x]->paste_partrait(&ppm, x * 64, y * 64);

  std::string jpeg;
  ppm.write_jpeg(&jpeg);
  size_t ret = fwrite(jpeg.data(), 1, jpeg.length(), outfp);
  if (ret != jpeg.length())
    return false;
  return true;
}

NEW_CMD(burn) {
  Urb *urb = server->urb;

  if (args.size() < 1)
    return false;
  unsigned int n = (unsigned int)atoi(args[0].c_str());
  if (n > 65535)
    n = 65535;

  double nu = 0.0001;
  double pi = 0.0001;
  if (args.size() >= 2)
    nu = strtod(args[1].c_str(), NULL);
  if (args.size() >= 3)
    pi = strtod(args[2].c_str(), NULL);

  std::vector<Parson*> parsons;
  parsons.resize(n);
  Zone *zone = urb->zones[0];
  for (unsigned int i = 0; i < n; ++i)
    parsons[i] = zone->pick();

  urb->pipex->burn(parsons.data(), parsons.size(), nu, pi);
  urb->pipex->save();
  urb->pipex->report("burn", outfp);
  return true;
}

}
