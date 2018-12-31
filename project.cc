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

  char samplesdatfn[4096];
  sprintf(samplesdatfn, "%s/samples.dat", _dir);
  samples = new Dataset(samplesdatfn, sampleslay->n);

  char contextdatfn[4096];
  sprintf(contextdatfn, "%s/context.dat", _dir);
  context = new Dataset(contextdatfn, contextlay->n);

  assert(context->n == samples->n);
  assert(context->k == contextlay->n);
  assert(samples->k == sampleslay->n);
  assert(outputlay->n == contextlay->n + sampleslay->n);
  assert(gentron->inn == mbn * (contextlay->n + controlslay->n));
  assert(gentron->outn == mbn * sampleslay->n);
  assert(enctron->inn == mbn * (contextlay->n + sampleslay->n));
  assert(enctron->outn == mbn * controlslay->n);

  cumake(&encin, mbn * (contextlay->n + sampleslay->n));
  cumake(&genin, mbn * (contextlay->n + controlslay->n));
  cumake(&gentgt, mbn * sampleslay->n);
  contextbuf = new double[contextlay->n * mbn];
  controlbuf = new double[controlslay->n * mbn];
  samplesbuf = new double[sampleslay->n * mbn];
  outputbuf = new double[outputlay->n * mbn];
  mbbuf = new unsigned int[mbn];
}

SimpleProject::~SimpleProject() {
  delete enctron;
  delete enctop;
  delete gentron;
  delete gentop;
  delete encpasstron; 
  delete encgentron;

  delete samples; 
  delete context;

  delete sampleslay;

  delete[] mbbuf;
  delete[] outputbuf;
  delete[] samplesbuf;
  delete[] controlbuf;
  delete[] contextbuf;
  cufree(encin);
  cufree(genin);
  cufree(gentgt);
}

void SimpleProject::learn(ControlSource control_source, double nu, unsigned int i) {
  size_t ret;

  context->pick_minibatch(mbn, mbbuf);
  samples->encude_minibatch(mbbuf, mbn, gentgt);

  switch (control_source) {
  case CONTROL_SOURCE_TRAINING:
    context->encude_minibatch(mbbuf, mbn, encin, 0, outputlay->n);
    samples->encude_minibatch(mbbuf, mbn, encin, contextlay->n, outputlay->n);
    encgentron->feed(encin, NULL);
    encgentron->target(gentgt);
    encgentron->train(nu);
    return; // !

  case CONTROL_SOURCE_STDIN:
    ret = fread(controlbuf, sizeof(double), controlslay->n * mbn, stdin);
    assert(ret == controlslay->n * mbn);
    break;

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
      controlbuf + mbi * controlslay->n,
      controlslay->n,
      genin + mbi * (contextlay->n + controlslay->n) + contextlay->n
    );
  }

  context->encude_minibatch(mbbuf, mbn, genin, 0, contextlay->n + controlslay->n);

  gentron->feed(genin, NULL);
  gentron->target(gentgt);
  gentron->train(nu);
}

void SimpleProject::report(unsigned int i) {
  fprintf(
    stderr,
    "i=%u gen_err2=%lf gen_errm=%lf encgen_err2=%lf encgen_errm=%lf\n",
    i, gentron->err2, gentron->errm, encgentron->err2, encgentron->errm
  );
}


void SimpleProject::generate(
  ContextSource context_source,
  ControlSource control_source
) {
  size_t ret;
  const double *encout;

  unsigned int labn = sampleslay->n;
  assert(contextlay->n + labn == outputlay->n);
  assert(labn % 3 == 0);

  unsigned int dim = round(sqrt(labn / 3));
  assert(dim * dim * 3 == labn);

  bool picked = 0;

  switch (context_source) {
  case CONTEXT_SOURCE_TRAINING:
    if (!picked) {
      context->pick_minibatch(mbn, mbbuf);
      picked = true;
    }
    context->copy_minibatch(mbbuf, mbn, contextbuf);
    break;
    
  case CONTEXT_SOURCE_STDIN:
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      ret = fread(contextbuf + mbi * contextlay->n, sizeof(double), contextlay->n, stdin);
      assert(ret == contextlay->n);

      if (control_source == CONTROL_SOURCE_STDIN) {
        ret = fread(controlbuf + mbi * controlslay->n, sizeof(double), controlslay->n, stdin);
        assert(ret == controlslay->n * mbn);
      }
    }
    break;

  default:
    assert(0);
  }

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(
      contextbuf + mbi * contextlay->n,
      contextlay->n,
      genin + mbi * (contextlay->n + controlslay->n)
    );
  }


  switch (control_source) {
  case CONTROL_SOURCE_STDIN:
    break;

  case CONTROL_SOURCE_TRAINING:
    if (!picked) {
      context->pick_minibatch(mbn, mbbuf);
      picked = true;
    }
    context->encude_minibatch(mbbuf, mbn, encin, 0, contextlay->n + sampleslay->n);
    samples->encude_minibatch(mbbuf, mbn, encin, contextlay->n, contextlay->n + sampleslay->n);
    encout = enctron->feed(encin, NULL);
    decude(encout, controlslay->n * mbn, controlbuf);
    break;

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
}

void SimpleProject::write_ppm(FILE *fp) {
  assert(mbn > 0);


  unsigned int labn = sampleslay->n;
  assert(contextlay->n + sampleslay->n == outputlay->n);
  assert(labn % 3 == 0);

  unsigned int dim = round(sqrt(labn / 3));
  assert(dim * dim * 3 == labn);

  PPM p;
  p.unvectorize(outputbuf + contextlay->n, dim, dim);
  p.write(fp);
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

  char hifreqdatfn[4096];
  sprintf(hifreqdatfn, "%s/hifreq.dat", _dir);
  hifreq = new Dataset(hifreqdatfn, hifreqlay->n);

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


  assert(attrs->n == lofreq->n);
  assert(attrs->n == hifreq->n);

  assert(attrs->k == attrslay->n);
  assert(lofreq->k == lofreqlay->n);
  assert(hifreq->k == hifreqlay->n);
  assert(contextlay->n == lofreqlay->n + attrslay->n);
  assert(outputlay->n == lofreqlay->n + hifreqlay->n + attrslay->n);
  assert(hifreqlay->n == 3 * lofreqlay->n);
  assert(gentron->inn == mbn * (contextlay->n + controlslay->n));
  assert(gentron->outn == mbn * hifreqlay->n);
  assert(enctron->inn == mbn * (contextlay->n + hifreqlay->n));
  assert(enctron->inn == mbn * outputlay->n);
  assert(enctron->outn == mbn * controlslay->n);


  cumake(&encin, mbn * (contextlay->n + hifreqlay->n));
  cumake(&genin, mbn * (contextlay->n + controlslay->n));
  cumake(&gentgt, mbn * (hifreqlay->n));
  attrsbuf = new double[attrslay->n * mbn];
  lofreqbuf = new double[lofreqlay->n * mbn];
  controlbuf = new double[controlslay->n * mbn];
  hifreqbuf = new double[hifreqlay->n * mbn];
  outputbuf = new double[outputlay->n * mbn];
  mbbuf = new unsigned int[mbn];
}

ZoomProject::~ZoomProject() {
  delete enctron;
  delete enctop;
  delete gentron;
  delete gentop;
  delete encpasstron; 
  delete encgentron;

  delete lofreq;
  delete lofreqlay;

  delete attrs;
  delete attrslay;

  delete hifreq;
  delete hifreqlay;



  cufree(encin);
  cufree(genin);
  cufree(gentgt);
  delete[] attrsbuf;
  delete[] lofreqbuf;
  delete[] controlbuf;
  delete[] hifreqbuf;
  delete[] mbbuf;
  delete[] outputbuf;
}


void ZoomProject::learn(ControlSource control_source, double nu, unsigned int i) {
  size_t ret;

  attrs->pick_minibatch(mbn, mbbuf);
  hifreq->encude_minibatch(mbbuf, mbn, gentgt);

  switch (control_source) {
  case CONTROL_SOURCE_TRAINING:
    attrs->encude_minibatch(mbbuf, mbn, encin, 0, outputlay->n);
    lofreq->encude_minibatch(mbbuf, mbn, encin, attrslay->n, outputlay->n);
    hifreq->encude_minibatch(mbbuf, mbn, encin, attrslay->n + lofreqlay->n, outputlay->n);

    encgentron->feed(encin, NULL);
    encgentron->target(gentgt);
    encgentron->train(nu);
    return; // !

  case CONTROL_SOURCE_STDIN:
    ret = fread(controlbuf, sizeof(double), controlslay->n * mbn, stdin);
    assert(ret == controlslay->n * mbn);
    break;

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
      controlbuf + mbi * controlslay->n,
      controlslay->n,
      genin + mbi * (contextlay->n + controlslay->n) + contextlay->n
    );
  }

  attrs->encude_minibatch(mbbuf, mbn, genin, 0, contextlay->n + controlslay->n);
  lofreq->encude_minibatch(mbbuf, mbn, genin, attrslay->n, contextlay->n + controlslay->n);

  gentron->feed(genin, NULL);
  gentron->target(gentgt);
  gentron->train(nu);
}



void ZoomProject::generate(
  ContextSource context_source,
  ControlSource control_source
) {
  size_t ret;
  const double *encout;

  unsigned int labn = lofreqlay->n + hifreqlay->n;
  assert(attrslay->n + labn == outputlay->n);
  assert(labn % 3 == 0);

  unsigned int dim = round(sqrt(labn / 3));
  assert(dim * dim * 3 == labn);
  assert(dim * dim * 9 == hifreqlay->n * 4);

  bool picked = false;


  switch (context_source) {
  case CONTEXT_SOURCE_TRAINING:
    if (!picked) {
      attrs->pick_minibatch(mbn, mbbuf);
      picked = true;
    }
    attrs->copy_minibatch(mbbuf, mbn, attrsbuf);
    lofreq->copy_minibatch(mbbuf, mbn, lofreqbuf);
    break;
    
  case CONTEXT_SOURCE_STDIN:
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      ret = fread(attrsbuf + mbi * attrslay->n, sizeof(double), attrslay->n, stdin);
      assert(ret == attrslay->n);
      ret = fread(lofreqbuf + mbi * lofreqlay->n, sizeof(double), lofreqlay->n, stdin);
      assert(ret == lofreqlay->n);

      if (control_source == CONTROL_SOURCE_STDIN) {
        ret = fread(controlbuf + mbi * controlslay->n, sizeof(double), controlslay->n, stdin);
        assert(ret == controlslay->n);
      }
    }
    break;

  default:
    assert(0);
  }

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(
      attrsbuf + mbi * attrslay->n,
      attrslay->n,
      genin + mbi * (contextlay->n + controlslay->n)
    );
    encude(
      lofreqbuf + mbi * lofreqlay->n,
      lofreqlay->n,
      genin + mbi * (contextlay->n + controlslay->n) + attrslay->n
    );
  }


  switch (control_source) {
  case CONTROL_SOURCE_STDIN:
    break;

  case CONTROL_SOURCE_TRAINING:
    if (!picked) {
      attrs->pick_minibatch(mbn, mbbuf);
      picked = true;
    }
    attrs->encude_minibatch(mbbuf, mbn, encin, 0, outputlay->n);
    lofreq->encude_minibatch(mbbuf, mbn, encin, attrslay->n, outputlay->n);
    hifreq->encude_minibatch(mbbuf, mbn, encin, attrslay->n + lofreqlay->n, outputlay->n);
    encout = enctron->feed(encin, NULL);
    decude(encout, controlslay->n * mbn, controlbuf);
    break;

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
      attrsbuf + mbi * attrslay->n, 
      attrslay->n * sizeof(double)
    );

    untwiddle3(
      lofreqbuf + mbi * lofreqlay->n,
      hifreqbuf + mbi * hifreqlay->n,
      dim, dim,
      outputbuf + mbi * outputlay->n + attrslay->n
    );
  }
}



void ZoomProject::write_ppm(FILE *fp) {
  assert(mbn > 0);

  unsigned int labn = lofreqlay->n + hifreqlay->n;
  assert(attrslay->n + labn == outputlay->n);
  assert(labn % 3 == 0);

  unsigned int dim = round(sqrt(labn / 3));
  assert(dim * dim * 3 == labn);
  assert(dim * dim * 9 == hifreqlay->n * 4);

  PPM p;
  p.unvectorize(outputbuf + attrslay->n, dim, dim);
  p.write(fp);
}


void ZoomProject::save() {
  enctron->sync(1);
  gentron->sync(1);
}

void ZoomProject::load() {
  enctron->sync(0);
  gentron->sync(0);
}

void ZoomProject::report(unsigned int i) {
  fprintf(
    stderr,
    "i=%u gen_err2=%lf gen_errm=%lf encgen_err2=%lf encgen_errm=%lf\n",
    i, gentron->err2, gentron->errm, encgentron->err2, encgentron->errm
  );
}








