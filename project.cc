#define __MAKEMORE_PROJECT_CC__ 1

#include <string>

#include "project.hh"
#include "cudamem.hh"
#include "tron.hh"
#include "multitron.hh"
#include "twiddle.hh"
#include "closest.hh"

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

Project::Project(const char *_dir, unsigned int _mbn) {
  mbn = _mbn;
  assert(mbn > 0);

  dir = _dir;
  assert(strlen(_dir) < 4000);

  char configfn[4096];
  sprintf(configfn, "%s/config.tsv", _dir);
  FILE *configfp = fopen(configfn, "r");
  assert(configfp);

  while (1) {
    string k = read_word(configfp, '\t');
    if (!k.length())
      break;
    string v = read_word(configfp, '\n');

    assert(config.find(k) == config.end());
    assert(v.length());
    config[k] = v;
  }
  fclose(configfp);

  char ctxlayfn[4096];
  sprintf(ctxlayfn, "%s/context.lay", _dir);
  ctxlay = new Layout;
  ctxlay->load_file(ctxlayfn);

  char ctrlayfn[4096];
  sprintf(ctrlayfn, "%s/control.lay", _dir);
  ctrlay = new Layout;
  ctrlay->load_file(ctrlayfn);

  char outlayfn[4096];
  sprintf(outlayfn, "%s/output.lay", _dir);
  outlay = new Layout;
  outlay->load_file(outlayfn);

  char tgtlayfn[4096];
  sprintf(tgtlayfn, "%s/target.lay", _dir);
  tgtlay = new Layout;
  tgtlay->load_file(tgtlayfn);

  do_zoom = (config["type"] == "zoom");
  lowoff = (unsigned int)atoi(config["lowoff"].c_str());
  if (do_zoom) {
    assert((ctxlay->n - lowoff) % 3 == 0);
    assert(lowoff < ctxlay->n);
    assert((ctxlay->n - lowoff) * 3 == tgtlay->n);
  } else {
    assert(tgtlay->n == outlay->n);
  }


  char encmapfn[4096], enctopfn[4096];
  sprintf(enctopfn, "%s/enc.top", _dir);
  sprintf(encmapfn, "%s/enc.map", _dir);
  enctop = new Topology;
  enctop->load_file(enctopfn);
  enc = new Multitron(*enctop, mbn, encmapfn);

  char genmapfn[4096], gentopfn[4096];
  sprintf(gentopfn, "%s/gen.top", _dir);
  sprintf(genmapfn, "%s/gen.map", _dir);
  gentop = new Topology;
  gentop->load_file(gentopfn);
  gen = new Multitron(*gentop, mbn, genmapfn);

  if (config["activated"] == "1") {
    gen->mt1->activated = true;
  } else {
    assert(config["activated"] == "0" || config["activated"] == "");
    gen->mt1->activated = false;
  }

  encinlay = new Layout(*ctxlay);
  *encinlay += *tgtlay;
  assert(enc->inn == mbn * encinlay->n);
  assert(enc->outn == mbn * ctrlay->n);

  geninlay = new Layout(*ctxlay);
  *geninlay += *ctrlay;
  assert(gen->inn == mbn * geninlay->n);
  assert(gen->outn == mbn * tgtlay->n);


  encpass = new Passthrutron(ctxlay->n, mbn, enc);
  genpass = new Passthrutron(ctxlay->n, mbn, gen);

  encgen = new Compositron(encpass, gen);
  genenc = new Compositron(genpass, enc);

  realctr = new double[mbn * ctrlay->n];
  fakectr = new double[mbn * ctrlay->n];
  morectr = new double[mbn * ctrlay->n];
  cumake(&cuenctgt, enc->outn);
  cumake(&cuencin, enc->inn);
  cumake(&cugentgt, gen->outn);
  cumake(&cugenin, gen->inn);

  bctxbuf = new uint8_t[mbn * ctxlay->n]();
  bctrbuf = new uint8_t[mbn * ctrlay->n]();
  btgtbuf = new uint8_t[mbn * tgtlay->n]();
  boutbuf = new uint8_t[mbn * outlay->n]();
  badjbuf = new uint8_t[mbn * outlay->n]();
  bsepbuf = new uint8_t[mbn * (ctxlay->n + tgtlay->n)];

  ctxbuf = new double[mbn * ctxlay->n]();
  ctrbuf = new double[mbn * ctrlay->n]();
  tgtbuf = new double[mbn * tgtlay->n]();
  outbuf = new double[mbn * outlay->n]();
  adjbuf = new double[mbn * outlay->n]();
  sepbuf = new double[mbn * (ctxlay->n + tgtlay->n)];

  rounds = 0;
}

Project::~Project() {
  delete encpass; 
  delete genpass;
  delete encgen;
  delete genenc;
  delete encinlay;
  delete geninlay;

  delete outlay;
  delete ctxlay;
  delete tgtlay;
  delete ctrlay;

  cufree(cugenin);
  cufree(cuencin);
  cufree(cugentgt);
  cufree(cuenctgt);

  delete[] realctr;
  delete[] fakectr;
  delete[] morectr;
  delete[] ctxbuf;
  delete[] tgtbuf;
  delete[] sepbuf;
  delete[] outbuf;
  delete[] adjbuf;

  delete[] bctxbuf;
  delete[] btgtbuf;
  delete[] boutbuf;
  delete[] bsepbuf;
  delete[] badjbuf;
}


void Project::load_ctxtgt(FILE *infp) {
  int ret;

  if (do_zoom) {
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      ret = fread(bsepbuf + mbi * (ctxlay->n + tgtlay->n), 1, ctxlay->n + tgtlay->n, infp);
      assert(ret == ctxlay->n + tgtlay->n);
    }
    for (unsigned int j = 0, jn = mbn * (ctxlay->n + tgtlay->n); j < jn; ++j)
      sepbuf[j] = (double)(bsepbuf[j] + 0.5) / 256.0;

    unsigned int labn = ctxlay->n + tgtlay->n - lowoff;
    unsigned int dim = lround(sqrt(labn / 3));
    assert(dim * dim * 3 == labn);

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      memcpy(
        ctxbuf + mbi * ctxlay->n,
        sepbuf + mbi * (ctxlay->n + tgtlay->n),
        sizeof(double) * lowoff
      );
      twiddle3(
        sepbuf + mbi * (ctxlay->n + tgtlay->n) + lowoff,
        dim, dim,
        ctxbuf + mbi * ctxlay->n + lowoff,
        tgtbuf + mbi * tgtlay->n
      );
    }
  } else {
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      ret = fread(bctxbuf + mbi * ctxlay->n, 1, ctxlay->n, infp);
      assert(ret == ctxlay->n);
      ret = fread(btgtbuf + mbi * tgtlay->n, 1, tgtlay->n, infp);
      assert(ret == tgtlay->n);
    }
    for (unsigned int j = 0, jn = mbn * ctxlay->n; j < jn; ++j)
      ctxbuf[j] = (double)(bctxbuf[j] + 0.5) / 256.0;
    for (unsigned int j = 0, jn = mbn * tgtlay->n; j < jn; ++j)
      tgtbuf[j] = (double)(btgtbuf[j] + 0.5) / 256.0;
  }
}

void Project::reconstruct() {
  if (!do_zoom)
     return;

  unsigned int dim = lround(sqrt(outlay->n / 3));
  assert(dim * dim * 3 == outlay->n);
  assert(outlay->n % 12 == 0);

  memcpy(sepbuf, outbuf, sizeof(double) * mbn * outlay->n);

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    untwiddle3(
      sepbuf + mbi * outlay->n + 0,
      sepbuf + mbi * outlay->n + (outlay->n / 4),
      dim, dim,
      outbuf + mbi * outlay->n
    );
  }
}

void Project::present(double nu, double mu, double xi) {
  if (nu > 0 || mu > 0) {
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      encude(ctxbuf + mbi * ctxlay->n, ctxlay->n, cuencin + mbi * encinlay->n + 0);
      encude(tgtbuf + mbi * tgtlay->n, tgtlay->n, cuencin + mbi * encinlay->n + ctxlay->n);
    }

    encgen->feed(cuencin, NULL);
    if (mu > 0)
      decude(enc->output(), enc->outn, realctr);

    encude(tgtbuf, gen->outn, cugentgt);
    encgen->target(cugentgt);
    encgen->train(nu);
  }
  

  if (mu > 0) {

    for (unsigned int j = 0, jn = mbn * ctrlay->n; j < jn; ++j)
      fakectr[j] = sigmoid(randgauss());
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      encude( ctxbuf + mbi * ctxlay->n, ctxlay->n, cugenin + mbi * geninlay->n + 0);
      encude(fakectr + mbi * ctrlay->n, ctrlay->n, cugenin + mbi * geninlay->n + ctxlay->n);
    }

    for (unsigned int j = 0, jn = mbn * ctrlay->n; j < jn; ++j)
      fakectr[j] = 0;
    encude(fakectr, enc->outn, cuenctgt);

    genenc->feed(cugenin, NULL);
    enc->target(cuenctgt);
    enc->train(mu);



    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      encude(realctr + mbi * ctrlay->n, ctrlay->n, cugenin + mbi * geninlay->n + ctxlay->n);
    }
    for (unsigned int j = 0, jn = mbn * ctrlay->n; j < jn; ++j)
      realctr[j] = 1;
    encude(realctr, enc->outn, cuenctgt);

    genenc->feed(cugenin, NULL);
    enc->target(cuenctgt);
    enc->train(mu);
  }


  if (xi > 0) {
    for (unsigned int j = 0, jn = mbn * ctrlay->n; j < jn; ++j)
      fakectr[j] = sigmoid(randgauss());

    if (mu > 0) {
      for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
        encude(fakectr + mbi * ctrlay->n, ctrlay->n, cugenin + mbi * geninlay->n + ctxlay->n);
      }
    } else {
      for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
        encude( ctxbuf + mbi * ctxlay->n, ctxlay->n, cugenin + mbi * geninlay->n + 0);
        encude(fakectr + mbi * ctrlay->n, ctrlay->n, cugenin + mbi * geninlay->n + ctxlay->n);
      }
      for (unsigned int j = 0, jn = mbn * ctrlay->n; j < jn; ++j)
        realctr[j] = 1;
      encude(realctr, enc->outn, cuenctgt);
    }

    genenc->feed(cugenin, NULL);
    enc->target(cuenctgt, false);
    enc->train(0);
    genpass->update_stats();
    genpass->train(xi);
  }


#if 0
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      unsigned int mbj = closest(fakectr + mbi * ctrlay->n, realctr, ctrlay->n, mbn);
      memcpy(morectr + mbi * ctrlay->n, realctr + mbj * ctrlay->n, ctrlay->n * sizeof(double));
      //need to copy context too
    }
#endif
#if 0
    assert(mbn % 2 == 0);
    for (unsigned int mbi = 0; mbi < mbn; ++mbi)
      for (unsigned int j = mbi * ctrlay->n, jn = j + ctrlay->n; j < jn; ++j)
        morectr[j] = mbi % 2 ? realctr[j] : fakectr[j];
#endif
#if 0
    memcpy(morectr, realctr, mbn * sizeof(double) * ctrlay->n);
#endif
#if 0
    memcpy(morectr, fakectr, mbn * sizeof(double) * ctrlay->n);

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      encude( ctxbuf + mbi * ctxlay->n, ctxlay->n, cugenin + mbi * geninlay->n + 0);
      encude(fakectr + mbi * ctrlay->n, ctrlay->n, cugenin + mbi * geninlay->n + ctxlay->n);
    }
    genenc->feed(cugenin, NULL);
    encude(morectr, enc->outn, cuenctgt);
#endif


#if 0
    enc->target(cuenctgt);
    enc->train(mu);
    gen->target(cugentgt); //???
    gen->train(mu);
#endif

#if 0
    genenc->target(cuenctgt);
    genenc->train(mu);
#endif

#if 0
    genenc->target(cuenctgt);
    enc->train(0);
    genpass->train(mu);
#endif


  ++rounds;
}

void Project::load_ctx(FILE *infp) {
  int ret;

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    ret = fread(bctxbuf + mbi * ctxlay->n, 1, ctxlay->n, infp);
    assert(ret == ctxlay->n);
  }
  for (unsigned int j = 0, jn = mbn * ctxlay->n; j < jn; ++j)
    ctxbuf[j] = (double)(bctxbuf[j] + 0.5) / 256.0;
}



void Project::generate(unsigned int reps) {
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude( ctxbuf + mbi * ctxlay->n, ctxlay->n, cugenin + mbi * geninlay->n + 0);
    encude( ctrbuf + mbi * ctrlay->n, ctrlay->n, cugenin + mbi * geninlay->n + ctxlay->n);
  }

  for (unsigned int i = 1; i < reps; ++i) {
    assert(enc->outn == mbn * ctrlay->n);
    const double *cuencout = genenc->feed(cugenin, NULL);
    decude(cuencout, enc->outn, fakectr);

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      encude(fakectr + mbi * ctrlay->n, ctrlay->n, cugenin + mbi * geninlay->n + ctxlay->n);
    }
  }

  const double *cugenout = gen->feed(cugenin, NULL);
  decude(cugenout, gen->outn, tgtbuf);

  passgenerate();
}


void Project::passgenerate() {
  if (!do_zoom) {
    assert(outlay->n == tgtlay->n);
    memcpy(outbuf, tgtbuf, sizeof(double) * mbn * tgtlay->n);
    return;
  }

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    memcpy(
      outbuf + mbi * outlay->n,
      ctxbuf + mbi * ctxlay->n + lowoff,
      (ctxlay->n - lowoff) * sizeof(double)
    );
    memcpy(
      outbuf + mbi * outlay->n + ctxlay->n - lowoff,
      tgtbuf + mbi * tgtlay->n,
      tgtlay->n * sizeof(double)
    );
  }
  reconstruct();
}

void Project::regenerate() {
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(ctxbuf + mbi * ctxlay->n, ctxlay->n, cuencin + mbi * encinlay->n + 0);
    encude(tgtbuf + mbi * tgtlay->n, tgtlay->n, cuencin + mbi * encinlay->n + ctxlay->n);
  }

  const double *cugenout = encgen->feed(cuencin, NULL);
  decude(cugenout, gen->outn, tgtbuf);

  passgenerate();
}

void Project::reencode() {
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(ctxbuf + mbi * ctxlay->n, ctxlay->n, cuencin + mbi * encinlay->n + 0);
    encude(tgtbuf + mbi * tgtlay->n, tgtlay->n, cuencin + mbi * encinlay->n + ctxlay->n);
  }

  const double *cuencout = enc->feed(cuencin, NULL);
  assert(enc->outn == mbn * ctrlay->n);
  decude(cuencout, enc->outn, ctrbuf);
}

void Project::burnmask(double nu) {
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(ctxbuf + mbi * ctxlay->n, ctxlay->n, cugenin + mbi * geninlay->n + 0);
    encude(ctrbuf + mbi * ctrlay->n, ctrlay->n, cugenin + mbi * geninlay->n + ctxlay->n);
  }
  assert(gen->outn == mbn * tgtlay->n);
  encude(tgtbuf, gen->outn, cugentgt);

  gen->feed(cugenin, NULL);
  gen->target(cugentgt);
  gen->train(nu);
}


void Project::report(const char *prog) {
  fprintf(
    stderr,
    "%s Project %s rounds=%u\n"
    "encgen_err2=%lf encgen_errm=%lf\n"
    "genenc_err2=%lf genenc_errm=%lf\n"
    "enc_err2=%lf enc_errm=%lf\n"
    "gen_err2=%lf gen_errm=%lf\n"
    "genpass_err2=%lf genpass_errm=%lf\n"
    "\n",
    prog, dir.c_str(), rounds,
    encgen->err2, encgen->errm,
    genenc->err2, genenc->errm,
    enc->err2, enc->errm,
    gen->err2, gen->errm,
    genpass->err2, genpass->errm
  );
}

void Project::save() {
  enc->sync(1);
  gen->sync(1);
}

void Project::load() {
  enc->sync(0);
  gen->sync(0);
}

void Project::scramble(double mean, double dev) {
  for (unsigned int j = 0, jn = mbn * ctrlay->n; j < jn; ++j) {
    ctrbuf[j] = sigmoid( mean + randgauss() * dev );
  }
}

void Project::encode_ctx() {
  for (unsigned int j = 0, jn = mbn * ctxlay->n; j < jn; ++j) {
    int v = lround(ctxbuf[j] * 256.0);
    bctxbuf[j] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
}

void Project::encode_out() {
  for (unsigned int j = 0, jn = mbn * outlay->n; j < jn; ++j) {
    int v = lround(outbuf[j] * 256.0);
    boutbuf[j] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
}

void Project::encode_tgt() {
  for (unsigned int j = 0, jn = mbn * tgtlay->n; j < jn; ++j) {
    int v = lround(tgtbuf[j] * 256.0);
    btgtbuf[j] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
}

void Project::encode_adj() {
  for (unsigned int j = 0, jn = mbn * outlay->n; j < jn; ++j) {
    double v = adjbuf[j];
    v /= 2.0;
    v += 0.5;
    v *= 256.0;
    v = lround(v);
    badjbuf[j] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
}

void Project::encode_ctr() {
  for (unsigned int j = 0, jn = mbn * ctrlay->n; j < jn; ++j) {
    double v = ctrbuf[j];
    v *= 256.0;
    v = round(v);
    bctrbuf[j] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
}
