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
  genwoke = false;


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

  distgtbuf = new double[mbn];
  cumake(&disin, mbn * outputlay->n);
  cumake(&distgt, mbn);
  cumake(&enctgt, mbn * controlslay->n);
}

SimpleProject::~SimpleProject() {
  delete enctron;
  delete enctop;
  delete gentron;
  delete gentop;
  delete encpasstron; 
  delete encgentron;

  delete sampleslay;

  delete[] outputbuf;
  delete[] boutputbuf;
  delete[] samplesbuf;
  delete[] controlbuf;
  delete[] contextbuf;
  delete[] bcontextbuf;
  delete[] distgtbuf;
  cufree(disin);
  cufree(distgt);
  cufree(encin);
  cufree(genin);
  cufree(gentgt);
  cufree(enctgt);
}

void SimpleProject::learn(FILE *infp, double nu, double dpres, double fpres, double cpres, double zpres, double fcut, double dcut, unsigned int i) {
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
    ret = fread(boutputbuf, 1, outputlay->n * mbn, infp);
    assert(ret == outputlay->n * mbn);
    for (unsigned int j = 0, jn = mbn * outputlay->n; j < jn; ++j)
      outputbuf[j] = (0.5 + (double)boutputbuf[j]) / 256.0;
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

    encude(outputbuf, mbn * outputlay->n, encin);
    encgentron->feed(encin, NULL);
    encgentron->target(gentgt);
 
    if (ferr2 > fcut)
      encgentron->train(nu * fpres); // * (ferr2 - fcut));
  }

  if (dpres > 0) {
    assert(mbn % 2 == 0);
  
    ret = fread(boutputbuf, 1, outputlay->n * mbn, infp);
    assert(ret == outputlay->n * mbn);
    for (unsigned int j = 0, jn = mbn * outputlay->n; j < jn; ++j)
      outputbuf[j] = (0.5 + (double)boutputbuf[j]) / 256.0;

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
    ret = fread(boutputbuf, 1, outputlay->n * mbn, infp);
    assert(ret == outputlay->n * mbn);
    for (unsigned int j = 0, jn = mbn * outputlay->n; j < jn; ++j)
      outputbuf[j] = (0.5 + (double)boutputbuf[j]) / 256.0;
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




#if 0
    ret = fread(boutputbuf, 1, outputlay->n * mbn, infp);
    assert(ret == outputlay->n * mbn);
    for (unsigned int j = 0, jn = mbn * outputlay->n; j < jn; ++j)
      outputbuf[j] = (0.5 + (double)boutputbuf[j]) / 256.0;
    encude(outputbuf, mbn * outputlay->n, encin);

    for (unsigned int mbi = 0; mbi < mbn; ++mbi)
      distgtbuf[mbi] = 0.0;
    encude(distgtbuf, mbn, distgt);

    encpasstron->feed(encin, NULL);
    gendistron->feed(encpasstron->output(), encpasstron->foutput());
    gendistron->train(0);
    encpasstron->train(nu * (fcut - ferr2) * (dcut - derr2) * cpres);
#endif
  }
}

void SimpleProject::report(const char *prog, unsigned int i) {
  fprintf(
    stderr,
    "%s SimpleProject %s i=%u genwoke=%u\n"
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


void SimpleProject::generate(
  FILE *infp,
  double dev,
  int fidelity
) {
  size_t ret;
  const double *encout;

  if (fidelity) {
    ret = fread(boutputbuf, 1, outputlay->n * mbn, infp);
    assert(ret == outputlay->n * mbn);
    for (unsigned int j = 0, jn = mbn * outputlay->n; j < jn; ++j)
      outputbuf[j] = (0.5 + (double)boutputbuf[j]) / 256.0;
    if (fidelity == 2)
      return;

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
  } else {

    ret = fread(bcontextbuf, 1, contextlay->n * mbn, infp);
    assert(ret == contextlay->n * mbn);
    for (unsigned int j = 0, jn = mbn * contextlay->n; j < jn; ++j)
      contextbuf[j] = (0.5 + (double)bcontextbuf[j]) / 256.0;
    for (unsigned int j = 0, jn = mbn * controlslay->n; j < jn; ++j)
      controlbuf[j] = randgauss() * dev;
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
    ppm.pastelab(outputbuf + mbi * outputlay->n + contextlay->n, dim, dim, xpos * dim, ypos * dim);
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
    ppm.pastelab(outputbuf + mbi * outputlay->n + contextlay->n, dim, dim, xpos * dim, ypos * dim);
  }
  ppm.write(fp);
  }
}


void SimpleProject::save() {
  enctron->sync(1);
  gentron->sync(1);
  distron->sync(1);
}

void SimpleProject::load() {
  enctron->sync(0);
  gentron->sync(0);
  distron->sync(0);
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
  genpasstron = passthrutron(contextlay->n, mbn, gentron);

  char dismapfn[4096], distopfn[4096];
  sprintf(distopfn, "%s/dis.top", _dir);
  sprintf(dismapfn, "%s/dis.map", _dir);
  distop = new Topology;
  distop->load_file(distopfn);
  distron = new Multitron(*distop, mbn, dismapfn);
  // distron->err2 = 0.5;
  gendistron = compositron(genpasstron, distron);

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

  distgtbuf = new double[mbn];
  cumake(&disin, mbn * (contextlay->n + controlslay->n));
  cumake(&distgt, mbn);
  cumake(&enctgt, mbn * controlslay->n);
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



  delete[] distgtbuf;
  cufree(disin);
  cufree(distgt);
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
  delete[] enctgt;
}


void ZoomProject::learn(FILE *infp, double nu, double dpres, double fpres, double cpres, double zpres, double fcut, double dcut, unsigned int i) {
  size_t ret;

  double ferr2 = encgentron->err2;
  double derr2 = distron->err2;

  if (fpres > 0) {
    ret = fread(boutputbuf, 1, outputlay->n * mbn, infp);
    assert(ret == outputlay->n * mbn);
    for (unsigned int j = 0, jn = mbn * outputlay->n; j < jn; ++j)
      outputbuf[j] = (0.5 + (double)boutputbuf[j]) / 256.0;
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      twiddle3(outputbuf + mbi * outputlay->n + attrslay->n, dim, dim, lofreqbuf, hifreqbuf);
      memcpy(outputbuf + mbi * outputlay->n + attrslay->n, lofreqbuf, lofreqlay->n * sizeof(double));
      memcpy(outputbuf + mbi * outputlay->n + contextlay->n, hifreqbuf, hifreqlay->n * sizeof(double));
    }

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      encude(
        outputbuf + mbi * outputlay->n + contextlay->n,
        hifreqlay->n,
        gentgt + mbi * hifreqlay->n
      );
    }

    encude(outputbuf, mbn * outputlay->n, encin);
    encgentron->feed(encin, NULL);
    encgentron->target(gentgt);

    if (ferr2 > fcut)
      encgentron->train(nu * fpres);
  }






  if (dpres > 0) {
    assert(mbn % 2 == 0);
  
    ret = fread(boutputbuf, 1, outputlay->n * mbn, infp);
    assert(ret == outputlay->n * mbn);
    for (unsigned int j = 0, jn = mbn * outputlay->n; j < jn; ++j)
      outputbuf[j] = (0.5 + (double)boutputbuf[j]) / 256.0;
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      twiddle3(outputbuf + mbi * outputlay->n + attrslay->n, dim, dim, lofreqbuf, hifreqbuf);
      memcpy(outputbuf + mbi * outputlay->n + attrslay->n, lofreqbuf, lofreqlay->n * sizeof(double));
      memcpy(outputbuf + mbi * outputlay->n + contextlay->n, hifreqbuf, hifreqlay->n * sizeof(double));
    }

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
    if (derr2 > dcut)
      distron->train(nu * dpres);
  }





  if (ferr2 < fcut && derr2 < dcut && cpres > 0) {
    ret = fread(boutputbuf, 1, outputlay->n * mbn, infp);
    assert(ret == outputlay->n * mbn);
    for (unsigned int j = 0, jn = mbn * outputlay->n; j < jn; ++j)
      outputbuf[j] = (0.5 + (double)boutputbuf[j]) / 256.0;
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      twiddle3(outputbuf + mbi * outputlay->n + attrslay->n, dim, dim, lofreqbuf, hifreqbuf);
      memcpy(outputbuf + mbi * outputlay->n + attrslay->n, lofreqbuf, lofreqlay->n * sizeof(double));
      memcpy(outputbuf + mbi * outputlay->n + contextlay->n, hifreqbuf, hifreqlay->n * sizeof(double));
    }
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
    gendistron->target(distgt);
    distron->train(0);
//fprintf(stderr, "cpres=%lf\n", cpres);
    genpasstron->train(nu * (dcut - derr2) * (fcut - ferr2) * cpres);



    ret = fread(boutputbuf, 1, outputlay->n * mbn, infp);
    assert(ret == outputlay->n * mbn);
    for (unsigned int j = 0, jn = mbn * outputlay->n; j < jn; ++j)
      outputbuf[j] = (0.5 + (double)boutputbuf[j]) / 256.0;
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      twiddle3(outputbuf + mbi * outputlay->n + attrslay->n, dim, dim, lofreqbuf, hifreqbuf);
      memcpy(outputbuf + mbi * outputlay->n + attrslay->n, lofreqbuf, lofreqlay->n * sizeof(double));
      memcpy(outputbuf + mbi * outputlay->n + contextlay->n, hifreqbuf, hifreqlay->n * sizeof(double));
    }
    encude(outputbuf, mbn * outputlay->n, encin);

    for (unsigned int mbi = 0; mbi < mbn; ++mbi)
      distgtbuf[mbi] = 0.0;
    encude(distgtbuf, mbn, distgt);

    encpasstron->feed(encin, NULL);
    gendistron->feed(encpasstron->output(), encpasstron->foutput());
    gendistron->train(0);
    encpasstron->train(nu * (dcut - derr2) * (fcut - ferr2) * cpres);
  }
}


void ZoomProject::generate(
  FILE *infp,
  double dev,
  int fidelity
) {
  size_t ret;
  const double *encout;


  if (fidelity) {
    ret = fread(boutputbuf, 1, outputlay->n * mbn, infp);
    assert(ret == outputlay->n * mbn);
    for (unsigned int j = 0, jn = mbn * outputlay->n; j < jn; ++j)
      outputbuf[j] = (0.5 + (double)boutputbuf[j]) / 256.0;

    if (fidelity == 2)
      return;

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
  } else {
    ret = fread(bcontextbuf, 1, contextlay->n * mbn, infp);
    assert(ret == contextlay->n * mbn);
    for (unsigned int j = 0, jn = mbn * contextlay->n; j < jn; ++j)
      contextbuf[j] = (0.5 + (double)bcontextbuf[j]) / 256.0;
    for (unsigned int j = 0; j < controlslay->n * mbn; ++j)
      controlbuf[j] = randgauss() * dev;
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
    ppm.pastelab(outputbuf + mbi * outputlay->n + attrslay->n, dim, dim, xpos * dim, ypos * dim);
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
    ppm.pastelab(outputbuf + mbi * outputlay->n + attrslay->n, dim, dim, xpos * dim, ypos * dim);
  }

  ppm.write(fp);
}
}


void ZoomProject::save() {
  enctron->sync(1);
  gentron->sync(1);
  distron->sync(1);
}

void ZoomProject::load() {
  enctron->sync(0);
  gentron->sync(0);
  distron->sync(0);
}

void ZoomProject::report(const char *prog, unsigned int i) {
  fprintf(
    stderr,
    "%s ZoomProject %s i=%u\n"
    "gen_err2=%lf    gen_errm=%lf\n"
    "encgen_err2=%lf encgen_errm=%lf\n"
    "dis_err2=%lf    dis_errm=%lf\n"
    "\n",
    prog, dir.c_str(), i,
    gentron->err2, gentron->errm, 
    encgentron->err2, encgentron->errm,
    distron->err2, distron->errm
  );
}








