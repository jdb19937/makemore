#include <system.hh>
#include <string.h>
#include <server.hh>
#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>
#include <ppm.hh>

using namespace makemore;

extern "C" void mainmore(Process *);

void mainmore(
  Process *process
) {
  if (process->args.size() != 2 && process->args.size() != 3)
    return;

  std::string nom = process->args[0];
  std::string png = process->args[1];

  bool ap = true;
  if (process->args.size() == 3 && process->args[2] == "0") {
    ap = false;
  }

  Urb *urb = process->system->server->urb;
  Parson *prs = urb->make(nom, 0);
  assert(prs);

  Partrait prt;
  if (png.length() > 4 && !memcmp(png.data(), "\x89PNG", 4)) {
    prt.from_png(png);
  } else {
    PPM ppm;
    ppm.read_jpeg(png);

//ppm.rotl();

    prt.w = ppm.w;
    prt.h = ppm.h;
    prt.rgb = ppm.data;
    ppm.data = NULL;
    ppm.w = 0;
    ppm.h = 0;
  }
  
  // prt.reflect();

fprintf(stderr, "got png %u %u\n", prt.w, prt.h);
  if (ap) {
    Pose curpose = Pose::STANDARD;
    // curpose.scale = 64; //prt.h / 4.0;
curpose.scale = prt.w / 8.0; 
    curpose.center.x = prt.w / 2.0;
    curpose.center.y = prt.h / 2.0;
    prt.set_pose(curpose);

    urb->ruffposer->autopose(&prt);
    urb->fineposer->autopose(&prt);
    urb->fineposer->autopose(&prt);
  } else {
    prt.set_pose(Pose::STANDARD);
  }

  Urbite who(nom, urb, prs);
  who.make_home_dir();
  std::string srcfn = who.home_dir() + "/source.png";
  assert(srcfn.length() < 255);
  prt.save(srcfn);
  strcpy(prs->srcfn, srcfn.c_str());

  Triangle curmark = prt.get_mark();
  if (curmark.p.x < 0) curmark.p.x = 0;
  if (curmark.p.y < 0) curmark.p.y = 0;
  if (curmark.q.x < 0) curmark.q.x = 0;
  if (curmark.q.y < 0) curmark.q.y = 0;
  if (curmark.r.x < 0) curmark.r.x = 0;
  if (curmark.r.y < 0) curmark.r.y = 0;
  if (curmark.p.x >= prt.w) curmark.p.x = prt.w - 1;
  if (curmark.p.y >= prt.h) curmark.p.y = prt.h - 1;
  if (curmark.q.x >= prt.w) curmark.q.x = prt.w - 1;
  if (curmark.q.y >= prt.h) curmark.q.y = prt.h - 1;
  if (curmark.r.x >= prt.w) curmark.r.x = prt.w - 1;
  if (curmark.r.y >= prt.h) curmark.r.y = prt.h - 1;
  prt.set_mark(curmark);

  Partrait stdprt(256, 256);
  stdprt.set_pose(Pose::STANDARD);
  prt.warp(&stdprt);

  Styler *sty = urb->get_sty(prs->sty);
  assert(sty);
  urb->enc->encode(stdprt, prs->controls);
  sty->encode(prs->controls, prs);

  for (unsigned int j = 0; j < Parson::ncontrols; ++j)
    prs->variations[j] = 1.0;

  // Pose pp = prt.get_pose();
  // prs->angle = pp.angle * 0.5;
  // prs->skew = pp.skew * 0.5;
  // prs->stretch = 1.0 + (pp.stretch - 1.0) * 0.5;

  strvec outvec;
  outvec.resize(1);
  outvec[0] = "ok";
  (void) process->write(outvec);
}
