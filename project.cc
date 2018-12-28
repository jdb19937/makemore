#define __MAKEMORE_PROJECT_CC__ 1

#include <string.h>
#include <assert.h>

#include "project.hh"
#include "multitron.hh"

Project::Project(const char *_dir, unsigned int _mbn) {
  assert(strlen(_dir) < 4000);
  mbn = _mbn;
  dir = _dir;

  char sampleslayfn[4096];
  sprintf(sampleslayfn, "%s/samples.lay", _dir);
  sampleslay = new Layout;
  sampleslay->load_file(sampleslayfn);

  char samplesdatfn[4096];
  sprintf(samplesdatfn, "%s/samples.dat", _dir);
  samples = new Dataset(samplesdatfn, sampleslay->n);

  char contextlayfn[4096];
  sprintf(contextlayfn, "%s/context.lay", _dir);
  contextlay = new Layout;
  contextlay->load_file(contextlayfn);

  char contextdatfn[4096];
  sprintf(contextdatfn, "%s/context.dat", _dir);
  context = new Dataset(contextdatfn, contextlay->n);

  char controlslayfn[4096];
  sprintf(controlslayfn, "%s/controls.lay", _dir);
  controlslay = new Layout;
  controlslay->load_file(controlslayfn);

  char encmapfn[4096], enctopfn[4096];
  sprintf(enctopfn, "%s/enc.top", _dir);
  sprintf(encmapfn, "%s/enc.map", _dir);
  enctop = new Topology;
  enctop->load_file(enctopfn);
  enctron = new Multitron(*enctop, mbn, encmapfn);

  char genmapfn[4096], gentopfn[4096];
  sprintf(gentopfn, "%s/gen.top", _dir);
  sprintf(genmapfn, "%s/gen.map", _dir);
  gentop = new Topology;
  gentop->load_file(gentopfn);
  gentron = new Multitron(*gentop, mbn, genmapfn);

  char dismapfn[4096], distopfn[4096];
  sprintf(distopfn, "%s/dis.top", _dir);
  sprintf(dismapfn, "%s/dis.map", _dir);
  distop = new Topology;
  distop->load_file(distopfn);
  distron = new Multitron(*distop, mbn, dismapfn);

}

Project::~Project() {
  delete enctron;
  delete enctop;
  delete gentron;
  delete gentop;
  delete distron;
  delete distop;

  delete contextlay;
  delete context;
  delete sampleslay;
  delete samples;
  delete controlslay;
}

#if 0
  cuzero(fpassbuf, mbn * outrn); 
  cucutpaste(in, out, mbn, inrn, outrn - npass, outrn, passbuf);

  return passbuf;
}

void Multitron::train(double nu) {
  if (npass > 0) {
    assert(fpassbuf);
    cucutadd(fpassbuf, mbn, outrn, outrn - npass, fout);
  }

  for (auto mi = megatrons.rbegin(); mi != megatrons.rend(); ++mi) {
    (*mi)->train(nu);
  }
}

void Multitron::sync(double t) {
  for (auto mi = megatrons.rbegin(); mi != megatrons.rend(); ++mi) {
    (*mi)->sync(t);
  }
}
#endif
