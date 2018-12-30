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


  char contextlayfn[4096];
  sprintf(contextlayfn, "%s/context.lay", _dir);
  contextlay = new Layout;
  contextlay->load_file(contextlayfn);


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

enctron->mt1->kappa = 4.0;

  encpasstron = passthrutron(contextlay->n, mbn, enctron);

  char genmapfn[4096], gentopfn[4096];
  sprintf(gentopfn, "%s/gen.top", _dir);
  sprintf(genmapfn, "%s/gen.map", _dir);
  gentop = new Topology;
  gentop->load_file(gentopfn);
  gentron = new Multitron(*gentop, mbn, genmapfn);

gentron->mt1->kappa = 4.0;

  encgentron = compositron(encpasstron, gentron);

#if 0
  char dismapfn[4096], distopfn[4096];
  sprintf(distopfn, "%s/dis.top", _dir);
  sprintf(dismapfn, "%s/dis.map", _dir);
  distop = new Topology;
  distop->load_file(distopfn);
  distron = new Multitron(*distop, mbn, dismapfn);
#endif

}

Project::~Project() {
  delete enctron;
  delete enctop;
  delete gentron;
  delete gentop;
#if 0
  delete distron;
  delete distop;
#endif
  delete encpasstron; 
  delete encgentron;

  delete contextlay;
  delete sampleslay;
  delete controlslay;
}


SimpleProject::SimpleProject(const char *_dir, unsigned int _mbn) : Project(_dir, _mbn) {
  char samplesdatfn[4096];
  sprintf(samplesdatfn, "%s/samples.dat", _dir);
  samples = new Dataset(samplesdatfn, sampleslay->n);

  char contextdatfn[4096];
  sprintf(contextdatfn, "%s/context.dat", _dir);
  context = new Dataset(contextdatfn, contextlay->n);
}

SimpleProject::~SimpleProject() {
  delete samples; 
  delete context;
}


ZoomProject::ZoomProject(const char *_dir, unsigned int _mbn) : Project(_dir, _mbn) {
  char hifreqdatfn[4096];
  sprintf(hifreqdatfn, "%s/hifreq.dat", _dir);
  hifreq = new Dataset(hifreqdatfn, sampleslay->n);

  char lofreqlayfn[4096];
  sprintf(lofreqlayfn, "%s/lofreq.lay", _dir);
  lofreqlay = new Layout;
  lofreqlay->load_file(lofreqlayfn);

  char lofreqdatfn[4096];
  sprintf(lofreqdatfn, "%s/lofreq.dat", _dir);
  lofreq = new Dataset(lofreqdatfn, lofreqlay->n);

  char attrslayfn[4096];
  sprintf(attrslayfn, "%s/attrs.lay", _dir);
  attrslay = new Layout;
  attrslay->load_file(attrslayfn);

  char attrsdatfn[4096];
  sprintf(attrsdatfn, "%s/attrs.dat", _dir);
  attrs = new Dataset(attrsdatfn, attrslay->n);
}

ZoomProject::~ZoomProject() {
  delete lofreq;
  delete lofreqlay;
  delete attrs;
  delete attrslay;
}
