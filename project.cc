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

  char targetlayfn[4096];
  sprintf(targetlayfn, "%s/target.lay", _dir);
  targetlay = new Layout;
  targetlay->load_file(targetlayfn);

  char adjustlayfn[4096];
  sprintf(adjustlayfn, "%s/adjust.lay", _dir);
  adjustlay = new Layout;
  adjustlay->load_file(adjustlayfn);

  contextbuf = new double[contextlay->n * mbn]();
  controlbuf = new double[controlslay->n * mbn]();
  dcontrolbuf = new double[controlslay->n * mbn]();
  outputbuf = new double[outputlay->n * mbn]();
  targetbuf = new double[targetlay->n * mbn]();
  adjustbuf = new double[adjustlay->n * mbn]();

  bcontextbuf = new uint8_t[contextlay->n * mbn]();
  bcontrolbuf = new uint8_t[controlslay->n * mbn]();
  boutputbuf = new uint8_t[outputlay->n * mbn]();
  btargetbuf = new uint8_t[targetlay->n * mbn]();
  badjustbuf = new uint8_t[adjustlay->n * mbn]();

}

Project::~Project() {
  delete contextlay;
  delete controlslay;
  delete outputlay;
  delete targetlay;

  if (config)
    delete config;

  delete[] contextbuf;
  delete[] controlbuf;
  delete[] dcontrolbuf;
  delete[] outputbuf;
  delete[] targetbuf;
  delete[] adjustbuf;

  delete[] bcontextbuf;
  delete[] bcontrolbuf;
  delete[] boutputbuf;
  delete[] badjustbuf;
  delete[] btargetbuf;
}

bool Project::loadcontext(FILE *infp) {
  int ret;
  ret = fread(bcontextbuf, 1, contextlay->n * mbn, infp);
  if (ret != contextlay->n * mbn) {
    return false;
  }
  for (unsigned int j = 0, jn = mbn * contextlay->n; j < jn; ++j)
    contextbuf[j] = (0.0 + (double)bcontextbuf[j]) / 256.0;
  return true;
}


bool Project::loadcontrols(FILE *infp) {
  int ret;
  ret = fread(bcontrolbuf, 1, controlslay->n * mbn, infp);
  if (ret != controlslay->n * mbn) {
    return false;
  }
  for (unsigned int j = 0, jn = mbn * controlslay->n; j < jn; ++j) {
    double z = ((double)bcontrolbuf[j] + 0.0) / 256.0;
    z = unsigmoid(z);
    controlbuf[j] = z;
  }
  return true;
}

void Project::nulladjust() {
  for (unsigned int j = 0, jn = mbn * adjustlay->n; j < jn; ++j) {
    adjustbuf[j] = 0;
  }
}


void Project::randcontrols(double dev) {
  int ret;
  for (unsigned int j = 0, jn = controlslay->n * mbn; j < jn; ++j)
    controlbuf[j] = randgauss() * dev;
}

void Project::encodectx() {
  for (unsigned int j = 0, jn = mbn * outputlay->n; j < jn; ++j) {
    int v = lround(contextbuf[j] * 256.0);
    bcontextbuf[j] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
}

void Project::encodeout() {
  for (unsigned int j = 0, jn = mbn * outputlay->n; j < jn; ++j) {
    int v = lround(outputbuf[j] * 256.0);
    boutputbuf[j] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
}

void Project::encodetgt() {
  for (unsigned int j = 0, jn = mbn * targetlay->n; j < jn; ++j) {
    int v = lround(targetbuf[j] * 256.0);
    btargetbuf[j] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
}

void Project::encodeadj() {
  for (unsigned int j = 0, jn = mbn * adjustlay->n; j < jn; ++j) {
    double v = adjustbuf[j];
    v /= 2.0;
    v += 0.5;
    v *= 256.0;
    v = lround(v);
    badjustbuf[j] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
}

void Project::encodectrl() {
  for (unsigned int j = 0, jn = mbn * controlslay->n; j < jn; ++j) {
    double v = sigmoid( controlbuf[j] );
    v *= 256.0;
    v = round(v);
    bcontrolbuf[j] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
}

bool Project::loadadjust(FILE *infp) {
  int ret;
  ret = fread(badjustbuf, 1, adjustlay->n * mbn, infp);
  if (ret != adjustlay->n * mbn) {
    return false;
  }
  for (unsigned int j = 0, jn = mbn * adjustlay->n; j < jn; ++j) {
    double z = ((double)badjustbuf[j] + 0.0) / 256.0;
    z -= 0.5;
    z *= 2.0;
    adjustbuf[j] = z;
  }
  return true;
}

bool Project::loadtarget(FILE *infp) {
  int ret;
  ret = fread(btargetbuf, 1, targetlay->n * mbn, infp);
  if (ret != targetlay->n * mbn) {
    return false;
  }
  for (unsigned int j = 0, jn = mbn * targetlay->n; j < jn; ++j) {
    double z = ((double)btargetbuf[j] + 0.0) / 256.0;
    targetbuf[j] = z;
  }
  return true;
}


bool Project::loadbatch(FILE *infp) {
  int ret;
  ret = fread(boutputbuf, 1, outputlay->n * mbn, infp);
  if (ret != outputlay->n * mbn) {
    return false;
  }
  for (unsigned int j = 0, jn = mbn * outputlay->n; j < jn; ++j)
    outputbuf[j] = (0.0 + (double)boutputbuf[j]) / 256.0;
  return true;
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
  cumake(&genfin, mbn * (contextlay->n + controlslay->n));
  cumake(&gentgt, mbn * sampleslay->n);

  distgtbuf = new double[mbn];
  cumake(&disin, mbn * outputlay->n);
  cumake(&distgt, mbn);
  cumake(&enctgt, mbn * controlslay->n);

  assert(adjustlay->n == targetlay->n);
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
  cufree(genfin);
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
) {
  size_t ret;
  const double *encout;

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
  uint8_t *hyper
) {
  size_t ret;
  double nu = hyper ? (0.01 * (double)hyper[0]) : 0;
  double mu = hyper ? (0.001 * (double)hyper[1]) : 0;

  if (nu > 0)
    cuzero(genfin, (contextlay->n + controlslay->n) * mbn);

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

  const double *genpassout = genpasstron->feed(genin, nu > 0 ? genfin : NULL);
  decude(genpassout, outputlay->n * mbn, outputbuf);
  assert(genpassout);
  assert(genpasstron->outn == outputlay->n * mbn);

  reconstruct();

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    unsigned int boff = mbi * adjustlay->n;
    unsigned int ooff = mbi * outputlay->n + (outputlay->n - adjustlay->n);
    for (unsigned int j = 0, jn = targetlay->n; j < jn; ++j, ++boff, ++ooff)
      outputbuf[ooff] += adjustbuf[boff];
  }

  if (nu > 0 || mu > 0) {
    separate();

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      encude(
        outputbuf + mbi * outputlay->n + contextlay->n,
        sampleslay->n,
        gentgt + mbi * sampleslay->n
      );
    }

    gentron->target(gentgt);
    gentron->train(mu);

    if (nu > 0) {
      for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
        decude(
          genfin + mbi * (contextlay->n + controlslay->n) + contextlay->n, 
          controlslay->n,
          dcontrolbuf + mbi * controlslay->n
        );
      }

      for (unsigned int j = 0, jn = mbn * controlslay->n; j < jn; ++j) {
//fprintf(stderr, "%lf,", dcontrolbuf[j]);
        dcontrolbuf[j] *= nu;
        controlbuf[j] += dcontrolbuf[j];
      }
//fprintf(stderr,"\n");
      encodectrl();
    }

    reconstruct();
  }

  encodeout();
}


void ImageProject::write_ppm(FILE *fp) {
  assert(mbn > 0);

  unsigned int labn = targetlay->n;
  assert(labn % 3 == 0);
  unsigned int dim = round(sqrt(labn / 3));

  assert(targetlay->n <= outputlay->n);
  unsigned int laboff = outputlay->n - targetlay->n;

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

  unsigned int n_adjusts = 0;
  for (unsigned int i = 0; i < stages.size(); ++i) {
    n_adjusts += stages[i]->adjustlay->n;
  }
  assert(adjustlay->n == n_adjusts);

  assert(stages[stages.size() - 1]->targetlay->n == targetlay->n);
}

PipelineProject::~PipelineProject() {
  for (auto pi = stages.begin(); pi != stages.end(); ++pi)
    delete *pi;
  stages.clear();
}
  

void PipelineProject::generate(
  uint8_t *hyper
) {
  assert(stages.size());

  unsigned int coff = 0;
  for (unsigned int i = 0; i < stages.size(); ++i) {
    assert(coff < controlslay->n);
    memcpy(stages[i]->controlbuf, controlbuf + coff * mbn, mbn * stages[i]->controlslay->n * sizeof(double));
    coff += stages[i]->controlslay->n;
  }
  assert(coff == controlslay->n);


  unsigned int aoff = 0;
  for (unsigned int i = 0; i < stages.size(); ++i) {
    assert(aoff < adjustlay->n);
    memcpy(stages[i]->adjustbuf, adjustbuf + aoff * mbn, mbn * stages[i]->adjustlay->n * sizeof(double));
    aoff += stages[i]->adjustlay->n;
  }
  assert(aoff == adjustlay->n);



  Project *proj;

  proj = stages[0];
  assert(contextlay->n == proj->contextlay->n);
  memcpy(proj->contextbuf, contextbuf, sizeof(double) * mbn * contextlay->n);
  proj->generate(hyper ? hyper + 0 : NULL);

  for (unsigned int i = 1; i < stages.size(); ++i) {
    Project *lastproj = stages[i - 1];
    Project *proj = stages[i];

    assert(proj->contextlay->n == lastproj->outputlay->n);
    assert(proj->mbn == lastproj->mbn);
    memcpy(proj->contextbuf, lastproj->outputbuf, mbn * sizeof(double) * proj->contextlay->n);

    proj->generate(hyper ? (hyper + (i * 2)) : NULL);
  }

  proj = stages[stages.size() - 1];
  assert(outputlay->n == proj->outputlay->n);
  memcpy(outputbuf, proj->outputbuf, sizeof(double) * mbn * outputlay->n);
  encodeout();


  if (hyper ? (hyper[0] || hyper[2] || hyper[4] || hyper[6]) : false) {
    coff = 0;
    for (unsigned int i = 0; i < stages.size(); ++i) {
      assert(coff < controlslay->n);
      memcpy(
        controlbuf + coff * mbn,
        stages[i]->controlbuf,
        mbn * stages[i]->controlslay->n * sizeof(double)
      );
      coff += stages[i]->controlslay->n;
    }
    assert(coff == controlslay->n);
    encodectrl();
  }
}




void PipelineProject::dotarget1(
) {
  assert(stages.size());

  unsigned int coff = 0;
  for (unsigned int i = 0; i < stages.size(); ++i) {
    assert(coff < controlslay->n);
    memcpy(stages[i]->controlbuf, controlbuf + coff * mbn, mbn * stages[i]->controlslay->n * sizeof(double));
    coff += stages[i]->controlslay->n;
  }
  assert(coff == controlslay->n);


  unsigned int aoff = 0;
  for (unsigned int i = 0; i < stages.size(); ++i) {
    assert(aoff < adjustlay->n);
    memcpy(stages[i]->adjustbuf, adjustbuf + aoff * mbn, mbn * stages[i]->adjustlay->n * sizeof(double));
    aoff += stages[i]->adjustlay->n;
  }
  assert(aoff == adjustlay->n);



  Project *proj = stages[0];
  assert(proj->contextlay->n == contextlay->n);
  memcpy(proj->contextbuf, contextbuf, mbn * contextlay->n * sizeof(double));

  proj->generate();
  for (unsigned int i = 1; i < stages.size(); ++i) {
    Project *lastproj = stages[i - 1];
    Project *proj = stages[i];

    assert(proj->contextlay->n == lastproj->outputlay->n);
    assert(proj->mbn == lastproj->mbn);
    memcpy(proj->contextbuf, lastproj->outputbuf, proj->contextlay->n * mbn * sizeof(double));

    proj->generate();
  }

  proj = stages[stages.size() - 1];
  assert(proj->targetlay->n == targetlay->n);
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    memcpy(
      targetbuf + mbi * targetlay->n,
      proj->outputbuf + mbi * proj->outputlay->n + proj->outputlay->n - proj->targetlay->n,
      targetlay->n * sizeof(double)
    );
  }

  encodetgt();
}

void PipelineProject::dotarget2() {
  Project *proj = stages[stages.size() - 1];
  assert(proj->targetlay->n == targetlay->n);
  memcpy(proj->targetbuf, targetbuf, mbn * sizeof(double) * targetlay->n);

  for (int i = stages.size() - 2; i >= 0; --i) {
    Project *lastproj = proj;
    proj = stages[i];
    unsigned int dim = lround(sqrt(lastproj->targetlay->n / 3));
    assert(dim * dim * 3 == lastproj->targetlay->n);
    assert(dim * dim * 3 == proj->targetlay->n * 4);

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      twiddle3(
        lastproj->targetbuf + mbi * proj->targetlay->n,
        dim, dim,
        proj->targetbuf + mbi * proj->targetlay->n, NULL
      );
    }
  }
}

void PipelineProject::readjust() {
  unsigned int aoff;
  nulladjust();




  Project *proj = stages[0];
  assert(proj->contextlay->n == contextlay->n);
  memcpy(proj->contextbuf, contextbuf, mbn * contextlay->n * sizeof(double));

  unsigned int coff = 0;
  for (unsigned int i = 0; i < stages.size(); ++i) {
    assert(coff < controlslay->n);
    memcpy(stages[i]->controlbuf, controlbuf + coff * mbn, mbn * stages[i]->controlslay->n * sizeof(double));
    coff += stages[i]->controlslay->n;
  }
  assert(coff == controlslay->n);




  for (unsigned int i = 0; i < stages.size(); ++i) {
    Project *lastproj = i > 0 ? stages[i-1] : NULL;
    Project *proj = stages[i];

    if (lastproj) {
      assert(proj->contextlay->n == lastproj->outputlay->n);
      assert(proj->mbn == lastproj->mbn);
      memcpy(proj->contextbuf, lastproj->outputbuf, proj->contextlay->n * mbn * sizeof(double));
    }

//fprintf(stderr, "adj=");
    proj->generate();
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      unsigned int boff = mbi * proj->targetlay->n;
      unsigned int ooff = mbi * proj->outputlay->n + (proj->outputlay->n - proj->targetlay->n);
      for (unsigned int j = 0, jn = proj->targetlay->n; j < jn; ++j, ++boff, ++ooff) {
        proj->adjustbuf[boff] = proj->targetbuf[boff] - proj->outputbuf[ooff];
//fprintf(stderr, "%lf,", proj->adjustbuf[boff]);
        proj->outputbuf[ooff] = proj->targetbuf[boff];
      }
    }
//fprintf(stderr, "\n");
  }






  aoff = 0;
  for (unsigned int i = 0; i < stages.size(); ++i) {
    assert(aoff < adjustlay->n);
    memcpy(
      adjustbuf + aoff * mbn,
      stages[i]->adjustbuf,
      mbn * stages[i]->adjustlay->n * sizeof(double)
    );
    aoff += stages[i]->adjustlay->n;
#if 0
    stages[i]->nulladjust();
#endif
  }

  assert(aoff == adjustlay->n);
  encodeadj();
}

#if 0
void PipelineProject::burnin(
) {
  generate();


  for (unsigned int i = 0; i < stages.size(); ++i) {
    Project *lastproj = i > 0 ? stages[i-1] : NULL;
    Project *proj = stages[i];

    if (lastproj) {
      unsigned int n = proj->contextlay->n;
      assert(n == lastproj->outputlay->n);
      assert(proj->mbn == lastproj->mbn);
      memcpy(proj->contextbuf, lastproj->outputbuf, n * mbn * sizeof(double));
    }
    memset(proj->adjustbuf, 0, targetlay->n * mbn * sizeof(double));

    proj->generate();

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      unsigned int boff = mbi * proj->targetlay->n;
      unsigned int ooff = mbi * proj->outputlay->n + (proj->outputlay->n - proj->targetlay->n);
      for (unsigned int j = 0, jn = proj->targetlay->n; j < jn; ++j, ++boff, ++ooff) {
        proj->adjustbuf[boff] = proj->targetbuf[boff] - proj->outputbuf[ooff];
        proj->outputbuf[ooff] = proj->targetbuf[boff];
      }
    }
  }




  generate();
}
#endif
 
const uint8_t *PipelineProject::output() const {
  assert(stages.size() > 0);
  const Project *proj = stages[stages.size() - 1];
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

void PipelineProject::save() {
  for (auto pi = stages.begin(); pi != stages.end(); ++pi) {
    Project *proj = *pi;
    proj->save();
  }
}



void PipelineProject::nulladjust() {
  for (unsigned int j = 0, jn = mbn * adjustlay->n; j < jn; ++j) {
    adjustbuf[j] = 0;
  }
  for (auto pi = stages.begin(); pi != stages.end(); ++pi) {
    Project *proj = *pi;
    proj->nulladjust();
  }
}



