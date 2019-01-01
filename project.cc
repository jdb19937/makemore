#define __MAKEMORE_PROJECT_CC__ 1

#include <string.h>
#include <assert.h>

#include <math.h>

#include "project.hh"
#include "multitron.hh"
#include "twiddle.hh"
#include "cudamem.hh"
#include "ppm.hh"

static std::string read_word(FILE *fp, char sep) {
  int c = getc(fp);
  if (c == EOF)
    return "";

  char buf[2];
  buf[0] = (char)c;
  buf[1] = 0;
  std::string word(buf);

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

  std::string type;

  while (1) {
    std::string k = read_word(fp, '\t');
    if (!k.length())
      break;
    std::string v = read_word(fp, '\n');
    assert(v.length());

    if (k == "type") {
      assert(type == "");
      type = v;
    }
  }
  fclose(fp);

  assert(type.length());

  Project *proj = NULL;
  if (type == "simple") {
    proj = new SimpleProject(dir, mbn);
  } else if (type == "zoom") {
    proj = new ZoomProject(dir, mbn);
  } else {
    fprintf(stderr, "unknown project type %s\n", type.c_str());
    assert(0);
  }

  return proj;
}

Project::Project(const char *_dir, unsigned int _mbn) {
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
}

Project::~Project() {
  delete contextlay;
  delete controlslay;
  delete outputlay;
}



SimpleProject::SimpleProject(const char *_dir, unsigned int _mbn) : Project(_dir, _mbn) {
  char sampleslayfn[4096];
  sprintf(sampleslayfn, "%s/samples.lay", _dir);
  sampleslay = new Layout;
  sampleslay->load_file(sampleslayfn);



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
  encgentron = compositron(encpasstron, gentron);

  char dismapfn[4096], distopfn[4096];
  sprintf(distopfn, "%s/dis.top", _dir);
  sprintf(dismapfn, "%s/dis.map", _dir);
  distop = new Topology;
  distop->load_file(distopfn);
  distron = new Multitron(*distop, mbn, dismapfn);
  encdistron = compositron(encpasstron, distron);


  assert(outputlay->n == contextlay->n + sampleslay->n);
  assert(gentron->inn == mbn * (contextlay->n + controlslay->n));
  assert(gentron->outn == mbn * sampleslay->n);
  assert(enctron->inn == mbn * (contextlay->n + sampleslay->n));
  assert(enctron->outn == mbn * controlslay->n);

  labn = sampleslay->n;
  assert(contextlay->n + labn == outputlay->n);
  assert(labn % 3 == 0);

  dim = round(sqrt(labn / 3));
  assert(dim * dim * 3 == labn);


  cumake(&encin, mbn * (contextlay->n + sampleslay->n));
  cumake(&genin, mbn * (contextlay->n + controlslay->n));
  cumake(&gentgt, mbn * sampleslay->n);
  bcontextbuf = new uint8_t[contextlay->n * mbn];
  contextbuf = new double[contextlay->n * mbn];
  controlbuf = new double[controlslay->n * mbn];
  samplesbuf = new double[sampleslay->n * mbn];
  outputbuf = new double[outputlay->n * mbn];
  boutputbuf = new uint8_t[outputlay->n * mbn];
}

SimpleProject::~SimpleProject() {
  delete enctron;
  delete enctop;
  delete gentron;
  delete gentop;
  delete encpasstron; 
  delete encgentron;

  delete sampleslay;

  delete[] mbbuf;
  delete[] outputbuf;
  delete[] boutputbuf;
  delete[] samplesbuf;
  delete[] controlbuf;
  delete[] contextbuf;
  delete[] bcontextbuf;
  cufree(encin);
  cufree(genin);
  cufree(gentgt);
}

void SimpleProject::learn(FILE *infp, ControlSource control_source, double nu, unsigned int i) {
  size_t ret;


  ret = fread(boutputbuf, 1, outputlay->n * mbn, infp);
  assert(ret == outputlay->n * mbn);
  for (unsigned int j = 0, jn = mbn * outputlay->n; j < jn; ++j)
    outputbuf[j] = (0.5 + (double)boutputbuf[j]) / 256.0;

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(
      outputbuf + mbi * outputlay->n + contextlay->n,
      sampleslay->n,
      gentgt + mbi * sampleslay->n
    );
  }

  switch (control_source) {
  case CONTROL_SOURCE_TRAINING:
    encude(outputbuf, mbn * outputlay->n, encin);
    encgentron->feed(encin, NULL);
    encgentron->target(gentgt);
    encgentron->train(nu);
    return; // !

  case CONTROL_SOURCE_CENTER:
    for (unsigned int j = 0; j < controlslay->n * mbn; ++j)
      controlbuf[j] = 0;
    break;

  case CONTROL_SOURCE_RANDOM:
    for (unsigned int j = 0; j < controlslay->n * mbn; ++j)
      controlbuf[j] = randgauss();
    break;

  default:
    assert(0);
    break;
  }

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

  gentron->feed(genin, NULL);
  gentron->target(gentgt);
  gentron->train(nu);
}

void SimpleProject::report(const char *prog, unsigned int i) {
  fprintf(
    stderr,
    "%s %s i=%u gen_err2=%lf gen_errm=%lf encgen_err2=%lf encgen_errm=%lf\n",
    prog, dir.c_str(), i,
    gentron->err2, gentron->errm, encgentron->err2, encgentron->errm
  );
}


void SimpleProject::generate(
  FILE *infp,
  ControlSource control_source
) {
  size_t ret;
  const double *encout;

  switch (control_source) {
  case CONTROL_SOURCE_TRAINING:
    ret = fread(boutputbuf, 1, outputlay->n * mbn, infp);
    assert(ret == outputlay->n * mbn);
    for (unsigned int j = 0, jn = mbn * outputlay->n; j < jn; ++j)
      outputbuf[j] = (0.5 + (double)boutputbuf[j]) / 256.0;
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
    break;

  case CONTROL_SOURCE_CENTER:
    ret = fread(bcontextbuf, 1, contextlay->n * mbn, infp);
    assert(ret == contextlay->n * mbn);
    for (unsigned int j = 0, jn = mbn * contextlay->n; j < jn; ++j)
      contextbuf[j] = (0.5 + (double)bcontextbuf[j]) / 256.0;

    for (unsigned int j = 0; j < controlslay->n * mbn; ++j)
      controlbuf[j] = 0;
    break;

  case CONTROL_SOURCE_RANDOM:
    ret = fread(bcontextbuf, 1, contextlay->n * mbn, infp);
    assert(ret == contextlay->n * mbn);
    for (unsigned int j = 0, jn = mbn * contextlay->n; j < jn; ++j)
      contextbuf[j] = (0.5 + (double)bcontextbuf[j]) / 256.0;

    for (unsigned int j = 0; j < controlslay->n * mbn; ++j)
      controlbuf[j] = randgauss();
    break;

  default:
    assert(0);
    break;
  }

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

  const double *genout = gentron->feed(genin, NULL);
  decude(genout, sampleslay->n * mbn, samplesbuf);

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    memcpy(
      outputbuf + (mbi * outputlay->n) + 0,
      contextbuf + (mbi * contextlay->n), 
      contextlay->n * sizeof(double)
    );

    memcpy(
      outputbuf + (mbi * outputlay->n) + contextlay->n,
      samplesbuf + (mbi * sampleslay->n),
      sampleslay->n * sizeof(double)
    );
  }
  for (unsigned int j = 0, jn = mbn * outputlay->n; j < jn; ++j) {
    int v = (int)(outputbuf[j] * 256.0);
    boutputbuf[j] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
}

void SimpleProject::write_ppm(FILE *fp) {
  assert(mbn > 0);

  unsigned int ldim = round(sqrt(mbn));
  if (ldim * ldim < mbn)
    ++ldim;

  PPM ppm(ldim * dim, ldim * dim, 0);
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    unsigned int xpos = mbi % ldim;
    unsigned int ypos = mbi / ldim;
    ppm.pastelab(outputbuf + mbi * outputlay->n + contextlay->n, dim, dim, xpos * dim, ypos * dim);
  }

  ppm.write(fp);
}


void SimpleProject::save() {
  enctron->sync(1);
  gentron->sync(1);
}

void SimpleProject::load() {
  enctron->sync(0);
  gentron->sync(0);
}


ZoomProject::ZoomProject(const char *_dir, unsigned int _mbn) : Project(_dir, _mbn) {
  assert(mbn > 0);

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
  encgentron = compositron(encpasstron, gentron);

  char dismapfn[4096], distopfn[4096];
  sprintf(distopfn, "%s/dis.top", _dir);
  sprintf(dismapfn, "%s/dis.map", _dir);
  distop = new Topology;
  distop->load_file(distopfn);
  distron = new Multitron(*distop, mbn, dismapfn);
  encdistron = compositron(encpasstron, distron);

  char hifreqlayfn[4096];
  sprintf(hifreqlayfn, "%s/hifreq.lay", _dir);
  hifreqlay = new Layout;
  hifreqlay->load_file(hifreqlayfn);

  char lofreqlayfn[4096];
  sprintf(lofreqlayfn, "%s/lofreq.lay", _dir);
  lofreqlay = new Layout;
  lofreqlay->load_file(lofreqlayfn);

  char attrslayfn[4096];
  sprintf(attrslayfn, "%s/attrs.lay", _dir);
  attrslay = new Layout;
  attrslay->load_file(attrslayfn);

  assert(contextlay->n == lofreqlay->n + attrslay->n);
  assert(outputlay->n == lofreqlay->n + hifreqlay->n + attrslay->n);
  assert(hifreqlay->n == 3 * lofreqlay->n);
  assert(gentron->inn == mbn * (contextlay->n + controlslay->n));
  assert(gentron->outn == mbn * hifreqlay->n);
  assert(enctron->inn == mbn * (contextlay->n + hifreqlay->n));
  assert(enctron->inn == mbn * outputlay->n);
  assert(enctron->outn == mbn * controlslay->n);

  labn = lofreqlay->n + hifreqlay->n;
  assert(attrslay->n + labn == outputlay->n);
  assert(labn % 3 == 0);

  dim = round(sqrt(labn / 3));
  assert(dim * dim * 3 == labn);
  assert(dim * dim * 9 == hifreqlay->n * 4);


  cumake(&encin, mbn * (contextlay->n + hifreqlay->n));
  cumake(&genin, mbn * (contextlay->n + controlslay->n));
  cumake(&gentgt, mbn * (hifreqlay->n));
  controlbuf = new double[controlslay->n * mbn];
  outputbuf = new double[outputlay->n * mbn];
  contextbuf = new double[contextlay->n * mbn];
  boutputbuf = new uint8_t[outputlay->n * mbn];
  bcontextbuf = new uint8_t[contextlay->n * mbn];

  lofreqbuf = new double[lofreqlay->n * mbn];
  hifreqbuf = new double[hifreqlay->n * mbn];
}

ZoomProject::~ZoomProject() {
  delete enctron;
  delete enctop;
  delete gentron;
  delete gentop;
  delete encpasstron; 
  delete encgentron;

  delete lofreqlay;
  delete attrslay;
  delete hifreqlay;



  cufree(encin);
  cufree(genin);
  cufree(gentgt);
  delete[] contextbuf;
  delete[] bcontextbuf;
  delete[] lofreqbuf;
  delete[] controlbuf;
  delete[] hifreqbuf;
  delete[] outputbuf;
  delete[] boutputbuf;
}


void ZoomProject::learn(FILE *infp, ControlSource control_source, double nu, unsigned int i) {
  size_t ret;

  ret = fread(boutputbuf, 1, outputlay->n * mbn, infp);
  assert(ret == outputlay->n * mbn);
  for (unsigned int j = 0, jn = mbn * outputlay->n; j < jn; ++j)
    outputbuf[j] = (0.5 + (double)boutputbuf[j]) / 256.0;

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    twiddle3(outputbuf + mbi * outputlay->n + attrslay->n, dim, dim, lofreqbuf, hifreqbuf);
    memcpy(outputbuf + mbi * outputlay->n + attrslay->n, lofreqbuf, lofreqlay->n * sizeof(double));
    memcpy(outputbuf + mbi * outputlay->n + contextlay->n, hifreqbuf, hifreqlay->n * sizeof(double));

    encude(outputbuf + mbi * outputlay->n + contextlay->n, hifreqlay->n, gentgt + mbi * hifreqlay->n);
  }

  switch (control_source) {
  case CONTROL_SOURCE_TRAINING:
    encude(outputbuf, mbn * outputlay->n, encin);
    encgentron->feed(encin, NULL);
    encgentron->target(gentgt);
    encgentron->train(nu);
    return; // !

  case CONTROL_SOURCE_CENTER:
    for (unsigned int j = 0; j < controlslay->n * mbn; ++j)
      controlbuf[j] = 0;
    break;

  case CONTROL_SOURCE_RANDOM:
    for (unsigned int j = 0; j < controlslay->n * mbn; ++j)
      controlbuf[j] = randgauss();
    break;

  default:
    assert(0);
    break;
  }

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

  gentron->feed(genin, NULL);
  gentron->target(gentgt);
  gentron->train(nu);
}



void ZoomProject::generate(
  FILE *infp,
  ControlSource control_source
) {
  size_t ret;
  const double *encout;


  switch (control_source) {
  case CONTROL_SOURCE_TRAINING:
    ret = fread(boutputbuf, 1, outputlay->n * mbn, infp);
    assert(ret == outputlay->n * mbn);
    for (unsigned int j = 0, jn = mbn * outputlay->n; j < jn; ++j)
      outputbuf[j] = (0.5 + (double)boutputbuf[j]) / 256.0;

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      twiddle3(outputbuf + mbi * outputlay->n + attrslay->n, dim, dim, lofreqbuf, hifreqbuf);
      memcpy(outputbuf + mbi * outputlay->n + attrslay->n, lofreqbuf, lofreqlay->n * sizeof(double));
      memcpy(outputbuf + mbi * outputlay->n + attrslay->n + lofreqlay->n, hifreqbuf, hifreqlay->n * sizeof(double));
    }

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
    break;

  case CONTROL_SOURCE_CENTER:
    ret = fread(bcontextbuf, 1, contextlay->n * mbn, infp);
    assert(ret == contextlay->n * mbn);
    for (unsigned int j = 0, jn = mbn * contextlay->n; j < jn; ++j)
      contextbuf[j] = (0.5 + (double)bcontextbuf[j]) / 256.0;

    for (unsigned int j = 0; j < controlslay->n * mbn; ++j)
      controlbuf[j] = 0;
    break;

  case CONTROL_SOURCE_RANDOM:
    ret = fread(bcontextbuf, 1, contextlay->n * mbn, infp);
    assert(ret == contextlay->n * mbn);
    for (unsigned int j = 0, jn = mbn * contextlay->n; j < jn; ++j)
      contextbuf[j] = (0.5 + (double)bcontextbuf[j]) / 256.0;

    for (unsigned int j = 0; j < controlslay->n * mbn; ++j)
      controlbuf[j] = randgauss();
    break;

  default:
    assert(0);
    break;
  }

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






  const double *genout = gentron->feed(genin, NULL);
  decude(genout, hifreqlay->n * mbn, hifreqbuf);

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    memcpy(
      outputbuf + mbi * outputlay->n,
      contextbuf + mbi * contextlay->n, 
      attrslay->n * sizeof(double)
    );

    untwiddle3(
      contextbuf + mbi * contextlay->n + attrslay->n,
      hifreqbuf + mbi * hifreqlay->n,
      dim, dim,
      outputbuf + mbi * outputlay->n + attrslay->n
    );
  }

  for (unsigned int j = 0, jn = mbn * outputlay->n; j < jn; ++j) {
    int v = (int)(outputbuf[j] * 256.0);
    boutputbuf[j] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
}



void ZoomProject::write_ppm(FILE *fp) {
  assert(mbn > 0);

  unsigned int ldim = round(sqrt(mbn));
  if (ldim * ldim < mbn)
    ++ldim;

  PPM ppm(ldim * dim, ldim * dim, 0);
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    unsigned int xpos = mbi % ldim;
    unsigned int ypos = mbi / ldim;
    ppm.pastelab(outputbuf + mbi * outputlay->n + attrslay->n, dim, dim, xpos * dim, ypos * dim);
  }

  ppm.write(fp);
}


void ZoomProject::save() {
  enctron->sync(1);
  gentron->sync(1);
}

void ZoomProject::load() {
  enctron->sync(0);
  gentron->sync(0);
}

void ZoomProject::report(const char *prog, unsigned int i) {
  fprintf(
    stderr,
    "%s %s i=%u gen_err2=%lf gen_errm=%lf encgen_err2=%lf encgen_errm=%lf\n",
    prog, dir.c_str(), i,
    gentron->err2, gentron->errm, encgentron->err2, encgentron->errm
  );
}






PipelineProject::PipelineProject(const char *_dir, unsigned int _mbn) : Project(_dir, _mbn) {
  // read tsv, build projects

  p0 = projects[0];
  p1 = projects[projects.size() - 1];
}

PipelineProject::~PipelineProject() {
  for (auto pi = projects.begin(); pi != projects.end(); ++pi)
    delete *pi;
}

void PipelineProject::generate(
  FILE *infp,
  ControlSource control_source
) {
  assert(0);
}

void PipelineProject::write_ppm(FILE *fp) {
  p1->write_ppm(fp);
}

const uint8_t *PipelineProject::output() const {
  return p1->output();
}

void PipelineProject::load() {
  for (auto pi = projects.begin(); pi != projects.end(); ++pi)
    (*pi)->load();
}


