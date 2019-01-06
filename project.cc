#define __MAKEMORE_PROJECT_CC__ 1

#include <string.h>
#include <assert.h>

#include <math.h>

#include "project.hh"
#include "multitron.hh"
#include "twiddle.hh"
#include "cudamem.hh"
#include "ppm.hh"

using namespace std;

static string read_word(FILE *fp, char sep) {
  int c = getc(fp);
  if (c == EOF)
    return "";

  char buf[2];
  buf[0] = (char)c;
  buf[1] = 0;
  string word(buf);

  while (1) {
    c = getc(fp);
    if (c == EOF)
      return "";
    if (c == sep)
      break;
    buf[0] = (char)c;
    word += buf;
  }

  return word;
}

Project *open_project(const char *dir, unsigned int mbn) {
  assert(mbn > 0);
  assert(strlen(dir) < 4000);

  char fn[4096];
  sprintf(fn, "%s/config.tsv", dir);
  FILE *fp = fopen(fn, "r");
  assert(fp);

  map<string, string> *config = new map<string,string>;

  while (1) {
    string k = read_word(fp, '\t');
    if (!k.length())
      break;
    string v = read_word(fp, '\n');

    assert(config->find(k) == config->end());
    assert(v.length());
    (*config)[k] = v;
  }
  fclose(fp);

  assert(config->find("type") != config->end());
  string type = (*config)["type"];
  assert(type.length());

  Project *proj = NULL;
  if (type == "image") {
    proj = new ImageProject(dir, mbn, config);
  } else if (type == "zoom") {
    proj = new ZoomProject(dir, mbn, config);
  } else if (type == "pipeline") {
    proj = new PipelineProject(dir, mbn, config);
  } else {
    fprintf(stderr, "unknown project type %s\n", type.c_str());
    assert(0);
  }

  return proj;
}

Project::Project(const char *_dir, unsigned int _mbn, map<string,string> *_config) {
  config = _config;

  assert(_mbn > 0);
  mbn = _mbn;

  assert(strlen(_dir) < 4000);
  dir = _dir;

  char contextlayfn[4096];
  sprintf(contextlayfn, "%s/context.lay", _dir);
  contextlay = new Layout;
  contextlay->load_file(contextlayfn);

  char controlslayfn[4096];
  sprintf(controlslayfn, "%s/controls.lay", _dir);
  controlslay = new Layout;
  controlslay->load_file(controlslayfn);

  char outputlayfn[4096];
  sprintf(outputlayfn, "%s/output.lay", _dir);
  outputlay = new Layout;
  outputlay->load_file(outputlayfn);

  char adjustlayfn[4096];
  sprintf(adjustlayfn, "%s/adjust.lay", _dir);
  adjustlay = new Layout;
  adjustlay->load_file(adjustlayfn);

  contextbuf = new double[contextlay->n * mbn]();
  controlbuf = new double[controlslay->n * mbn]();
  outputbuf = new double[outputlay->n * mbn]();
  adjustbuf = new double[adjustlay->n * mbn]();

  bcontextbuf = new uint8_t[contextlay->n * mbn]();
  bcontrolbuf = new uint8_t[controlslay->n * mbn]();
  boutputbuf = new uint8_t[outputlay->n * mbn]();
  badjustbuf = new uint8_t[adjustlay->n * mbn]();

}

Project::~Project() {
  delete contextlay;
  delete controlslay;
  delete outputlay;
  delete adjustlay;

  if (config)
    delete config;

  delete[] contextbuf;
  delete[] controlbuf;
  delete[] outputbuf;
  delete[] adjustbuf;

  delete[] bcontextbuf;
  delete[] bcontrolbuf;
  delete[] boutputbuf;
  delete[] badjustbuf;
}



ImageProject::ImageProject(const char *_dir, unsigned int _mbn, map<string,string> *_config) : Project(_dir, _mbn, _config) {
  genwoke = false;

  char sampleslayfn[4096];
  sprintf(sampleslayfn, "%s/samples.lay", _dir);
  sampleslay = new Layout;
  sampleslay->load_file(sampleslayfn);

  samplesbuf = new double[sampleslay->n * mbn];

  char encmapfn[4096], enctopfn[4096];
  sprintf(enctopfn, "%s/enc.top", _dir);
  sprintf(encmapfn, "%s/enc.map", _dir);
  enctop = new Topology;
  enctop->load_file(enctopfn);
  enctron = new Multitron(*enctop, mbn, encmapfn);
  enctron->mt1->activated = false;

  encpasstron = passthrutron(contextlay->n, mbn, enctron);

  char genmapfn[4096], gentopfn[4096];
  sprintf(gentopfn, "%s/gen.top", _dir);
  sprintf(genmapfn, "%s/gen.map", _dir);
  gentop = new Topology;
  gentop->load_file(gentopfn);
  gentron = new Multitron(*gentop, mbn, genmapfn);
  gentron->mt1->activated = false;
  genpasstron = passthrutron(contextlay->n, mbn, gentron);
  encgentron = compositron(encpasstron, gentron);

  char dismapfn[4096], distopfn[4096];
  sprintf(distopfn, "%s/dis.top", _dir);
  sprintf(dismapfn, "%s/dis.map", _dir);
  distop = new Topology;
  distop->load_file(distopfn);
  distron = new Multitron(*distop, mbn, dismapfn);
  // distron->err2 = 0.5;
  gendistron = compositron(genpasstron, distron);

  assert(outputlay->n == contextlay->n + sampleslay->n);
  assert(gentron->inn == mbn * (contextlay->n + controlslay->n));
  assert(gentron->outn == mbn * sampleslay->n);
  assert(enctron->inn == mbn * (contextlay->n + sampleslay->n));
  assert(enctron->outn == mbn * controlslay->n);
  assert(distron->inn == outputlay->n * mbn);
  assert(distron->outn == mbn);



  cumake(&encin, mbn * (contextlay->n + sampleslay->n));
  cumake(&genin, mbn * (contextlay->n + controlslay->n));
  cumake(&gentgt, mbn * sampleslay->n);

  distgtbuf = new double[mbn];
  cumake(&disin, mbn * outputlay->n);
  cumake(&distgt, mbn);
  cumake(&enctgt, mbn * controlslay->n);
}

ImageProject::~ImageProject() {
  delete enctron;
  delete enctop;
  delete gentron;
  delete gentop;
  delete encpasstron; 
  delete encgentron;

  delete sampleslay;
  delete[] samplesbuf;

  delete[] distgtbuf;
  cufree(disin);
  cufree(distgt);
  cufree(encin);
  cufree(genin);
  cufree(gentgt);
  cufree(enctgt);
}

void ImageProject::learn(FILE *infp, double nu, double dpres, double fpres, double cpres, double zpres, double fcut, double dcut, unsigned int i) {
  size_t ret;

  double genwake = 0.1;
  double gensleep = 0.4;

  double ferr2 = encgentron->err2;
  double derr2 = distron->err2;

  if (genwoke && derr2 > gensleep) {
    fprintf(stderr, "putting generator to bed\n");
    distron->randomize(8, 8);
    genwoke = false;
  }
  if (!genwoke && derr2 < genwake) {
    fprintf(stderr, "waking generator\n");
    genwoke = true;
  }

  if (zpres > 0) {
    loadbatch(infp);
    separate();

    for (unsigned int j = 0, jn = controlslay->n * mbn; j < jn; ++j)
      controlbuf[j] = randgauss();

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      encude(
        outputbuf + mbi * outputlay->n + contextlay->n,
        sampleslay->n,
        gentgt + mbi * sampleslay->n
      );
      encude(
        outputbuf + mbi * outputlay->n,
        contextlay->n,
        genin + mbi * (contextlay->n + controlslay->n)
      );
      encude(
        controlbuf + mbi * controlslay->n,
        controlslay->n,
        genin + mbi * (contextlay->n + controlslay->n) + contextlay->n
      );
    }

    gentron->feed(genin, NULL);
    gentron->target(gentgt);
    gentron->train(nu * zpres);
  }

  if (fpres > 0 && genwoke) {
    loadbatch(infp);
    separate();

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      encude(
        outputbuf + mbi * outputlay->n + contextlay->n,
        sampleslay->n,
        gentgt + mbi * sampleslay->n
      );
    }

    encude(outputbuf, mbn * outputlay->n, encin);
    encgentron->feed(encin, NULL);
    encgentron->target(gentgt);
 
    if (ferr2 > fcut)
      encgentron->train(nu * fpres); // * (ferr2 - fcut));
  }

  if (dpres > 0) {
    assert(mbn % 2 == 0);
    loadbatch(infp);
    separate();

    encude(outputbuf, mbn * outputlay->n, encin);
    const double *encout = enctron->feed(encin, NULL);
    decude(encout, mbn * controlslay->n, controlbuf);


    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      encude(
        outputbuf + mbi * outputlay->n,
        contextlay->n,
        genin + mbi * (contextlay->n + controlslay->n)
      );

      if (mbi % 2 == 0) {
        for (unsigned int j = 0, jn = controlslay->n; j < jn; ++j) {
          controlbuf[mbi * jn + j] = randgauss();
        }
      }

      encude(
        controlbuf + mbi * controlslay->n,
        controlslay->n,
        genin + mbi * (contextlay->n + controlslay->n) + contextlay->n
      );
    }

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      if (mbi % 2) {
        distgtbuf[mbi] = 1.0;
      } else {
        distgtbuf[mbi] = 0.0;
      }
    }
    encude(distgtbuf, mbn, distgt);

    gendistron->feed(genin, NULL);
    distron->target(distgt);
    distron->train(nu * dpres);
  }



  if (cpres > 0 && genwoke) {
    loadbatch(infp);
    separate();

    for (unsigned int j = 0, jn = controlslay->n * mbn; j < jn; ++j)
      controlbuf[j] = randgauss();

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      encude(
        outputbuf + mbi * outputlay->n,
        contextlay->n,
        genin + mbi * (contextlay->n + controlslay->n)
      );

      encude(
        controlbuf + mbi * controlslay->n,
        controlslay->n,
        genin + mbi * (contextlay->n + controlslay->n) + contextlay->n
      );
    }

    for (unsigned int mbi = 0; mbi < mbn; ++mbi)
      distgtbuf[mbi] = 1.0;
    encude(distgtbuf, mbn, distgt);


    gendistron->feed(genin, NULL);
    gendistron->target(distgt, false);
    distron->train(0);
//fprintf(stderr, "cpres=%lf\n", cpres);
    genpasstron->train(nu * cpres);
  }

}

void ImageProject::report(const char *prog, unsigned int i) {
  fprintf(
    stderr,
    "%s ImageProject %s i=%u genwoke=%u\n"
    "gen_err2=%lf    gen_errm=%lf\n"
    "encgen_err2=%lf encgen_errm=%lf\n"
    "dis_err2=%lf    dis_errm=%lf\n"
    "\n",
    prog, dir.c_str(), i, (unsigned)genwoke,
    gentron->err2, gentron->errm, 
    encgentron->err2, encgentron->errm,
    distron->err2, distron->errm
  );
}

void ImageProject::regenerate(
  FILE *infp
) {
  size_t ret;
  const double *encout;

  if (infp)
    loadbatch(infp);
  separate();

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    memcpy(
      contextbuf + mbi * contextlay->n,
      outputbuf + mbi * outputlay->n,
      contextlay->n * sizeof(double)
    );
  }

  encude(outputbuf, mbn * outputlay->n, encin);
  encout = enctron->feed(encin, NULL);
  decude(encout, controlslay->n * mbn, controlbuf);

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(
      contextbuf + mbi * contextlay->n,
      contextlay->n,
      genin + mbi * (contextlay->n + controlslay->n)
    );

    encude(
      controlbuf + mbi * controlslay->n,
      controlslay->n,
      genin + mbi * (contextlay->n + controlslay->n) + contextlay->n
    );
  }

  const double *genpassout = genpasstron->feed(genin, NULL);
  decude(genpassout, outputlay->n * mbn, outputbuf);
  reconstruct();


  encodeout();
}





void ImageProject::generate(
  FILE *infp,
  double dev
) {
  size_t ret;

  if (infp)
    loadcontext(infp);
  for (unsigned int j = 0, jn = mbn * controlslay->n; j < jn; ++j)
    controlbuf[j] = randgauss() * dev;

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(
      contextbuf + mbi * contextlay->n,
      contextlay->n,
      genin + mbi * (contextlay->n + controlslay->n)
    );

    encude(
      controlbuf + mbi * controlslay->n,
      controlslay->n,
      genin + mbi * (contextlay->n + controlslay->n) + contextlay->n
    );
  }

  const double *genpassout = genpasstron->feed(genin, NULL);
  decude(genpassout, outputlay->n * mbn, outputbuf);
  reconstruct();

  encodeout();
}


void ImageProject::write_ppm(FILE *fp) {
  assert(mbn > 0);

  unsigned int labn = adjustlay->n;
  assert(labn % 3 == 0);
  unsigned int dim = round(sqrt(labn / 3));

  assert(adjustlay->n <= outputlay->n);
  unsigned int laboff = outputlay->n - adjustlay->n;

bool wide = 0;

  if (wide) {
    unsigned int wdim = round(sqrt(mbn * 2));
    unsigned int hdim = (int)(wdim/2);
    if (wdim * hdim < mbn)
      ++wdim;
    assert(wdim * hdim >= mbn);

  PPM ppm(wdim * dim, hdim * dim, 0);
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    unsigned int xpos = mbi % wdim;
    unsigned int ypos = mbi / wdim;
    ppm.pastelab(outputbuf + mbi * outputlay->n + laboff, dim, dim, xpos * dim, ypos * dim);
  }
  ppm.write(fp);

  } else {
  unsigned int ldim = round(sqrt(mbn));
  if (ldim * ldim < mbn)
    ++ldim;

  PPM ppm(ldim * dim, ldim * dim, 0);
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    unsigned int xpos = mbi % ldim;
    unsigned int ypos = mbi / ldim;
    ppm.pastelab(outputbuf + mbi * outputlay->n + laboff, dim, dim, xpos * dim, ypos * dim);
  }
  ppm.write(fp);
  }
}


void ImageProject::save() {
  enctron->sync(1);
  gentron->sync(1);
  distron->sync(1);
}

void ImageProject::load() {
  enctron->sync(0);
  gentron->sync(0);
  distron->sync(0);
}


ZoomProject::ZoomProject(const char *_dir, unsigned int _mbn, map<string,string> *_config) : ImageProject(_dir, _mbn, _config) {
  assert(mbn > 0);

  char lofreqlayfn[4096];
  sprintf(lofreqlayfn, "%s/lofreq.lay", _dir);
  lofreqlay = new Layout;
  lofreqlay->load_file(lofreqlayfn);

  char attrslayfn[4096];
  sprintf(attrslayfn, "%s/attrs.lay", _dir);
  attrslay = new Layout;
  attrslay->load_file(attrslayfn);

  assert(contextlay->n == lofreqlay->n + attrslay->n);
  assert(outputlay->n == lofreqlay->n + sampleslay->n + attrslay->n);
  assert(sampleslay->n == 3 * lofreqlay->n);

  lofreqbuf = new double[lofreqlay->n * mbn];
}

ZoomProject::~ZoomProject() {
  delete attrslay;
  delete lofreqlay;

  delete[] lofreqbuf;
}

void ZoomProject::separate() {
  unsigned int labn = sampleslay->n + lofreqlay->n;
  assert(labn % 3 == 0);
  unsigned int dim = round(sqrt(labn / 3));

  assert(dim * dim * 3 == labn);
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    twiddle3(outputbuf + mbi * outputlay->n + attrslay->n, dim, dim, lofreqbuf, samplesbuf);
    memcpy(outputbuf + mbi * outputlay->n + attrslay->n, lofreqbuf, lofreqlay->n * sizeof(double));
    memcpy(outputbuf + mbi * outputlay->n + attrslay->n + lofreqlay->n, samplesbuf, sampleslay->n * sizeof(double));
  }
}

void ImageProject::separate() {

}

void ZoomProject::reconstruct() {
  unsigned int labn = sampleslay->n + lofreqlay->n;
  assert(labn % 3 == 0);
  unsigned int dim = round(sqrt(labn / 3));

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    memcpy(
      samplesbuf + mbi * sampleslay->n,
      outputbuf + mbi * outputlay->n + contextlay->n, 
      sampleslay->n * sizeof(double)
    );

    untwiddle3(
      contextbuf + mbi * contextlay->n + attrslay->n,
      samplesbuf + mbi * sampleslay->n,
      dim, dim,
      outputbuf + mbi * outputlay->n + attrslay->n
    );
  }
}

void ImageProject::reconstruct() {

}

void ImageProject::loadbatch(FILE *infp) {
  int ret;
  ret = fread(boutputbuf, 1, outputlay->n * mbn, infp);
  assert(ret == outputlay->n * mbn);
  for (unsigned int j = 0, jn = mbn * outputlay->n; j < jn; ++j)
    outputbuf[j] = (0.5 + (double)boutputbuf[j]) / 256.0;
}

void ImageProject::loadcontext(FILE *infp) {
  int ret;
  ret = fread(bcontextbuf, 1, contextlay->n * mbn, infp);
  assert(ret == contextlay->n * mbn);
  for (unsigned int j = 0, jn = mbn * contextlay->n; j < jn; ++j)
    contextbuf[j] = (0.5 + (double)bcontextbuf[j]) / 256.0;
}

void ImageProject::encodeout() {
  for (unsigned int j = 0, jn = mbn * outputlay->n; j < jn; ++j) {
    int v = (int)(outputbuf[j] * 256.0);
    boutputbuf[j] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
}



void ZoomProject::report(const char *prog, unsigned int i) {
  fprintf(stderr, "[ZoomProject] ");
  ImageProject::report(prog, i);
}



PipelineProject::PipelineProject(const char *_dir, unsigned int _mbn, map<string,string> *_config) : Project(_dir, _mbn, _config) {

  assert(config);
  assert(mbn > 0);

  Project *proj = NULL;
  unsigned int stage = 0;
  char stagekey[32], stagedir[8192];
  string stageval;

  while (1) {
    ++stage;
    sprintf(stagekey, "stage%u", stage);
    stageval = (*config)[stagekey];
    if (stageval == "")
      break;
    assert(stageval.length() < 4000);
    assert(dir.length() < 4000);
    sprintf(stagedir, "%s/%s", dir.c_str(), stageval.c_str());

    Project *lastproj = proj;
    proj = open_project(stagedir, mbn);

    if (lastproj) {
      assert(lastproj->outputlay->n == proj->contextlay->n);
    }

    stages.push_back(proj);
  }

  assert(stages.size());

  unsigned int n_controls = 0;
  for (unsigned int i = 0; i < stages.size(); ++i) {
    n_controls += stages[i]->controlslay->n;
  }
  assert(n_controls == controlslay->n);
}

PipelineProject::~PipelineProject() {
  for (auto pi = stages.begin(); pi != stages.end(); ++pi)
    delete *pi;
  stages.clear();
}
  

void PipelineProject::generate(
  FILE *infp,
  double dev
) {
  assert(stages.size());
  stages[0]->generate(infp, dev);

  for (unsigned int i = 1; i < stages.size(); ++i) {
    Project *lastproj = stages[i - 1];
    Project *proj = stages[i];

    unsigned int n = proj->contextlay->n;
    assert(n == lastproj->outputlay->n);
    assert(proj->mbn == lastproj->mbn);
    memcpy(proj->contextbuf, lastproj->outputbuf, n * mbn * sizeof(double));

    proj->generate(NULL, dev);
  }

}
 
const uint8_t *PipelineProject::output() const {
  auto pi = stages.rbegin();
  assert(pi != stages.rend());
  Project *proj = *pi;
  return proj->output();
}

void PipelineProject::write_ppm(FILE *fp) {
  auto pi = stages.rbegin();
  assert(pi != stages.rend());
  Project *proj = *pi;
  proj->write_ppm(fp);
}

void PipelineProject::load() {
  for (auto pi = stages.begin(); pi != stages.end(); ++pi) {
    Project *proj = *pi;
    proj->load();
  }
}






