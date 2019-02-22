#define __MAKEMORE_PROJECT_CC__ 1

#include <string>
#include <netinet/in.h>

#include "stage.hh"
#include "project.hh"
#include "cudamem.hh"
#include "tron.hh"
#include "multitron.hh"
#include "twiddle.hh"
#include "closest.hh"

namespace makemore {

using namespace std;

Stage::Stage(const char *_dir, unsigned int _mbn) : Project(_dir, _mbn) {
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

  assert(config["type"] == "stage");
  assert(config["zoom"] == "0" || config["zoom"] == "1");
  assert(config["lowoff"] != "");

  lowoff = (unsigned int)atoi(config["lowoff"].c_str());
  do_zoom = (config["zoom"] == "1");

  if (do_zoom) {
    assert((ctxlay->n - lowoff) % 3 == 0);
    assert(lowoff < ctxlay->n);
    assert((ctxlay->n - lowoff) * 3 == tgtlay->n);
  } else {
    assert(tgtlay->n == outlay->n);
  }


  char dismapfn[4096], distopfn[4096];
  sprintf(distopfn, "%s/dis.top", _dir);
  sprintf(dismapfn, "%s/dis.map", _dir);
  distop = new Topology;
  distop->load_file(distopfn);
  dis = new Multitron(*distop, mbn, dismapfn);

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
  gendis = new Compositron(genpass, dis);

  realctr = new double[mbn * ctrlay->n];
  fakectr = new double[mbn * ctrlay->n];
  fakectx = new double[mbn * ctxlay->n];
  morectr = new double[mbn * ctrlay->n];
  distgt = new double[mbn];

  cumake(&cuenctgt, enc->outn);
  cumake(&cudistgt, dis->outn);
  cumake(&cuencin, enc->inn);
  cumake(&cudisin, dis->inn);
  cumake(&cugentgt, gen->outn);
  cumake(&cugenin, gen->inn);

  bctxbuf = new uint8_t[mbn * ctxlay->n]();
  bctrbuf = new uint8_t[mbn * ctrlay->n]();
  btgtbuf = new uint8_t[mbn * tgtlay->n]();
  boutbuf = new uint8_t[mbn * outlay->n]();
  sadjbuf = new uint16_t[mbn * outlay->n]();
  bsepbuf = new uint8_t[mbn * (ctxlay->n + tgtlay->n)];

  ctxbuf = new double[mbn * ctxlay->n]();
  ctrbuf = new double[mbn * ctrlay->n]();
  tgtbuf = new double[mbn * tgtlay->n]();
  outbuf = new double[mbn * outlay->n]();
  adjbuf = new double[mbn * outlay->n]();
  sepbuf = new double[mbn * (ctxlay->n + tgtlay->n)];

  assert(dis->inn == enc->inn);
  assert(mbn == dis->outn);

  cumake(&cutgtlayx, tgtlay->n);
  encude(tgtlay->x, tgtlay->n, cutgtlayx);

  cumake(&cutgtlayy, tgtlay->n);
  encude(tgtlay->y, tgtlay->n, cutgtlayy);

  rounds = 0;
}

Stage::~Stage() {
  delete encpass; 
  delete genpass;
  delete encgen;
  delete genenc;
  delete gendis;
  delete encinlay;
  delete geninlay;

  delete outlay;
  delete ctxlay;
  delete tgtlay;
  delete ctrlay;

  cufree(cugenin);
  cufree(cuencin);
  cufree(cudisin);
  cufree(cugentgt);
  cufree(cuenctgt);
  cufree(cudistgt);

  cufree(cutgtlayx);
  cufree(cutgtlayy);

  delete[] distgt;
  delete[] realctr;
  delete[] fakectr;
  delete[] fakectx;
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
  delete[] sadjbuf;
}


void Stage::load_ctxtgt(FILE *infp) {
  int ret;

  if (do_zoom) {
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      ret = fread(bsepbuf + mbi * (ctxlay->n + tgtlay->n), 1, ctxlay->n + tgtlay->n, infp);
      assert(ret == ctxlay->n + tgtlay->n);
    }
    for (unsigned int j = 0, jn = mbn * (ctxlay->n + tgtlay->n); j < jn; ++j)
      sepbuf[j] = ((double)bsepbuf[j] + 0.5) / 256.0;

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
      ctxbuf[j] = ((double)bctxbuf[j] + 0.5) / 256.0;
    for (unsigned int j = 0, jn = mbn * tgtlay->n; j < jn; ++j)
      tgtbuf[j] = ((double)btgtbuf[j] + 0.5) / 256.0;
  }
}

void Stage::reconstruct() {
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

static void recenter(double *x, unsigned int n) {
  for (unsigned int i = 0; i < n; ++i) {
    x[i] = unsigmoid(x[i]);
  }

  double mean = 0, stddev = 0;
  for (unsigned int i = 0; i < n; ++i) {
    mean += x[i];
  }
  mean /= (double)n;

  for (unsigned int i = 0; i < n; ++i) {
    x[i] -= mean;
  }

  for (unsigned int i = 0; i < n; ++i) {
    stddev += x[i] * x[i];
  }
  stddev /= (double)n;
  stddev = sqrt(stddev);

  if (stddev < 1e-9)
    stddev = 1e-9;

  for (unsigned int i = 0; i < n; ++i) {
    x[i] /= stddev;
  }

  for (unsigned int i = 0; i < n; ++i) {
    x[i] = sigmoid(x[i]);
  }
}

void Stage::train_recombine(double yo, double wu, unsigned int js) {
  assert(mbn % 2 == 0);
  assert(ctrlay->n % js == 0);

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(ctxbuf + mbi * ctxlay->n, ctxlay->n, cuencin + mbi * encinlay->n + 0);
    encude(tgtbuf + mbi * tgtlay->n, tgtlay->n, cuencin + mbi * encinlay->n + ctxlay->n);
  }

  const double *cuencout = enc->feed(cuencin, NULL);
  decude(cuencout, enc->outn, realctr);
  for (unsigned int mbi = 0; mbi < mbn; mbi += 2) {
    for (unsigned int j = mbi * ctrlay->n, jn = j + ctrlay->n; j < jn; j += js) {
      double jprob = 0.5;

      if (randrange(0, 1) < jprob) {
        for (unsigned int s = j, sn = j + js, t = s + ctrlay->n; s < sn; ++s, ++t) {
          fakectr[s] = realctr[s];
          fakectr[t] = realctr[t];
        }
      } else {
        for (unsigned int s = j, sn = j + js, t = s + ctrlay->n; s < sn; ++s, ++t) {
          fakectr[s] = realctr[t];
          fakectr[t] = realctr[s];
        }
      }
    }
  }

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude( ctxbuf + mbi * ctxlay->n, ctxlay->n, cugenin + mbi * geninlay->n + 0);
    encude(fakectr + mbi * ctrlay->n, ctrlay->n, cugenin + mbi * geninlay->n + ctxlay->n);
  }

  encude(realctr, enc->outn, cuenctgt);
  genenc->feed(cugenin, NULL);
  genenc->target(cuenctgt);
  enc->train(0);
  genpass->update_stats();
  genpass->train(wu);


  encude(fakectr, enc->outn, cuenctgt);
  enc->feed(genpass->output(), NULL);
  enc->target(cuenctgt);
  enc->train(yo);
}

void Stage::train_scramble(double yo, double wu) {
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(ctxbuf + mbi * ctxlay->n, ctxlay->n, cuencin + mbi * encinlay->n + 0);
    encude(tgtbuf + mbi * tgtlay->n, tgtlay->n, cuencin + mbi * encinlay->n + ctxlay->n);
  }

  const double *cuencout = enc->feed(cuencin, NULL);
  decude(cuencout, enc->outn, realctr);

#if 0
fprintf(stderr, "realctr: ");
for (unsigned int j = 0; j < 10; ++j)
fprintf(stderr, "%lf, ", realctr[j]);
fprintf(stderr, "\n");
#endif

  for (unsigned int mbi = 0; mbi < mbn; ++mbi)
    for (unsigned int j = mbi * ctrlay->n, jn = j + ctrlay->n; j < jn; ++j)
      fakectr[j] = sigmoid(randgauss());

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude( ctxbuf + mbi * ctxlay->n, ctxlay->n, cugenin + mbi * geninlay->n + 0);
    encude(fakectr + mbi * ctrlay->n, ctrlay->n, cugenin + mbi * geninlay->n + ctxlay->n);
  }

  encude(realctr, enc->outn, cuenctgt);
  genenc->feed(cugenin, NULL);
  genenc->target(cuenctgt);
  enc->train(0);
  genpass->update_stats();
  genpass->train(wu);


  encude(fakectr, enc->outn, cuenctgt);
  enc->feed(genpass->output(), NULL);
  enc->target(cuenctgt);
  enc->train(yo);
}

double Stage::encgenerr() {
  assert(mbn == 1);

  encude(ctxbuf, ctxlay->n, cuencin + 0);
  encude(tgtbuf, tgtlay->n, cuencin + ctxlay->n);
  encude(tgtbuf, gen->outn, cugentgt);

  encgen->feed(cuencin, NULL);

  gen->reset_stats();
  gen->target(cugentgt);
  return gen->err2;
}

void Stage::train_fidelity(double nu, double pi, double dcut) {
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(ctxbuf + mbi * ctxlay->n, ctxlay->n, cuencin + mbi * encinlay->n + 0);
    encude(tgtbuf + mbi * tgtlay->n, tgtlay->n, cuencin + mbi * encinlay->n + ctxlay->n);
  }

  encude(tgtbuf, gen->outn, cugentgt);

  encgen->feed(cuencin, NULL);
  gen->target(cugentgt);
  // gen->train(pi * (dis->err2 < dcut ? (1.0 - dis->err2 / dcut) : 0.0));
  gen->train(pi);
  encpass->update_stats();
  encpass->train(nu);
}
  

void Stage::train_judgement(double mu, double dcut) {
  assert(mbn % 2 == 0);

  for (unsigned int mbi = 0; mbi < mbn; ++mbi)
    for (unsigned int j = mbi * ctrlay->n, jn = j + ctrlay->n; j < jn; ++j)
      fakectr[j] = sigmoid(randgauss());

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude( ctxbuf + mbi * ctxlay->n, ctxlay->n, cugenin + mbi * geninlay->n + 0);
    encude(fakectr + mbi * ctrlay->n, ctrlay->n, cugenin + mbi * geninlay->n + ctxlay->n);
  }

  const double *cugenout = gen->feed(cugenin, NULL);

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(ctxbuf + mbi * ctxlay->n, ctxlay->n, cudisin + mbi * encinlay->n + 0);

    if (mbi % 2) {
      encude(tgtbuf + mbi * tgtlay->n, tgtlay->n, cudisin + mbi * encinlay->n + ctxlay->n);
    } else {
      cucopy(cugenout + mbi * tgtlay->n, tgtlay->n, cudisin + mbi * encinlay->n + ctxlay->n);
    }
  }


  for (unsigned int mbi = 0; mbi < mbn; ++mbi)
    distgt[mbi] = (mbi % 2) ? 1.0 : 0.0;
  encude(distgt, dis->outn, cudistgt);

  dis->feed(cudisin, NULL);
  dis->target(cudistgt);
  dis->train(mu * (dis->err2 < dcut ? dis->err2 / dcut : 1.0));
}

void Stage::train_creativity(double xi, double dcut) {
  // random -> 1 (gen)
  for (unsigned int mbi = 0; mbi < mbn; ++mbi)
    for (unsigned int j = mbi * ctrlay->n, jn = j + ctrlay->n; j < jn; ++j)
      fakectr[j] = sigmoid(randgauss());

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude( ctxbuf + mbi * ctxlay->n, ctxlay->n, cugenin + mbi * geninlay->n + 0);
    encude(fakectr + mbi * ctrlay->n, ctrlay->n, cugenin + mbi * geninlay->n + ctxlay->n);
  }

  gendis->feed(cugenin, NULL);

  for (unsigned int mbi = 0; mbi < mbn; ++mbi)
    distgt[mbi] = 1.0;
  encude(distgt, dis->outn, cudistgt);

  gendis->target(cudistgt);
  dis->train(0);
  genpass->update_stats();
  genpass->train(xi * (dis->err2 < dcut ? (1.0 - dis->err2 / dcut) : 0.0));
}

void Stage::load_ctx(FILE *infp) {
  int ret;

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    ret = fread(bctxbuf + mbi * ctxlay->n, 1, ctxlay->n, infp);
    assert(ret == ctxlay->n);
  }
  for (unsigned int j = 0, jn = mbn * ctxlay->n; j < jn; ++j)
    ctxbuf[j] = ((double)bctxbuf[j] + 0.5) / 256.0;
}



void Stage::generate(unsigned int reps) {
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
  for (unsigned int j = 0, jn = mbn * tgtlay->n; j < jn; ++j)
    tgtbuf[j] = tgtbuf[j] > 1.0 ? 1.0 : tgtbuf[j] < 0.0 ? 0.0 : tgtbuf[j];

  passgenerate();
}


void Stage::passgenerate() {
  if (!do_zoom) {
    assert(outlay->n == tgtlay->n);
    memcpy(outbuf, tgtbuf, sizeof(double) * mbn * tgtlay->n);
    return;
  }

  assert(outlay->n == ctxlay->n - lowoff + tgtlay->n);

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

void Stage::regenerate() {
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(ctxbuf + mbi * ctxlay->n, ctxlay->n, cuencin + mbi * encinlay->n + 0);
    encude(tgtbuf + mbi * tgtlay->n, tgtlay->n, cuencin + mbi * encinlay->n + ctxlay->n);
  }

  const double *cugenout = encgen->feed(cuencin, NULL);
  decude(cugenout, gen->outn, tgtbuf);
  for (unsigned int j = 0, jn = mbn * tgtlay->n; j < jn; ++j)
    tgtbuf[j] = tgtbuf[j] > 1.0 ? 1.0 : tgtbuf[j] < 0.0 ? 0.0 : tgtbuf[j];

  passgenerate();
}

void Stage::reencode(bool force) {
  const double *cuencout;

  if (force) {
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      encude(ctxbuf + mbi * ctxlay->n, ctxlay->n, cuencin + mbi * encinlay->n + 0);
      encude(tgtbuf + mbi * tgtlay->n, tgtlay->n, cuencin + mbi * encinlay->n + ctxlay->n);
    }

    cuencout = enc->feed(cuencin, NULL);
    assert(enc->outn == mbn * ctrlay->n);
    decude(cuencout, enc->outn, ctrbuf);
    return;
  }

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(ctxbuf + mbi * ctxlay->n, ctxlay->n, cugenin + mbi * geninlay->n + 0);
    encude(ctrbuf + mbi * ctrlay->n, ctrlay->n, cugenin + mbi * geninlay->n + ctxlay->n);
  }
  const double *cugenout0 = gen->feed(cugenin, NULL);
  double *genout0 = new double[mbn * tgtlay->n];
  decude(cugenout0, mbn * tgtlay->n, genout0);
  for (unsigned int j = 0, jn = mbn * tgtlay->n; j < jn; ++j)
    genout0[j] = genout0[j] > 1.0 ? 1.0 : genout0[j] < 0.0 ? 0.0 : genout0[j];

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(ctxbuf + mbi * ctxlay->n, ctxlay->n, cuencin + mbi * encinlay->n + 0);
    encude(tgtbuf + mbi * tgtlay->n, tgtlay->n, cuencin + mbi * encinlay->n + ctxlay->n);
  }

  cuencout = enc->feed(cuencin, NULL);
  assert(enc->outn == mbn * ctrlay->n);
  decude(cuencout, enc->outn, fakectr);

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude( ctxbuf + mbi * ctxlay->n, ctxlay->n, cugenin + mbi * geninlay->n + 0);
    encude(fakectr + mbi * ctrlay->n, ctrlay->n, cugenin + mbi * geninlay->n + ctxlay->n);
  }
  const double *cugenout1 = gen->feed(cugenin, NULL);
  double *genout1 = new double[mbn * tgtlay->n];
  decude(cugenout1, mbn * tgtlay->n, genout1);
  for (unsigned int j = 0, jn = mbn * tgtlay->n; j < jn; ++j)
    genout1[j] = genout1[j] > 1.0 ? 1.0 : genout1[j] < 0.0 ? 0.0 : genout1[j];


  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    double err0 = 0;
    for (unsigned int j = tgtlay->n * mbi, jn = j + tgtlay->n; j < jn; ++j)
      err0 += (genout0[j] - tgtbuf[j]) * (genout0[j] - tgtbuf[j]);
    double err1 = 0;
    for (unsigned int j = tgtlay->n * mbi, jn = j + tgtlay->n; j < jn; ++j)
      err1 += (genout1[j] - tgtbuf[j]) * (genout1[j] - tgtbuf[j]);
fprintf(stderr, "err0=%lf err1=%lf\n", err0, err1);

    if (err1 < err0) {
      memcpy(ctrbuf + mbi * ctrlay->n, fakectr + mbi * ctrlay->n, sizeof(double) * ctrlay->n);
    }
  }

  delete[] genout0;
  delete[] genout1;
  encode_ctr();
}

void Stage::burn(double nu, double pi) {
#if 0
  if (pi > 0) {
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      encude(ctxbuf + mbi * ctxlay->n, ctxlay->n, cugenin + mbi * geninlay->n + 0);
      encude(ctrbuf + mbi * ctrlay->n, ctrlay->n, cugenin + mbi * geninlay->n + ctxlay->n);
    }
    assert(gen->outn == mbn * tgtlay->n);
    encude(tgtbuf, gen->outn, cugentgt);

    gen->feed(cugenin, NULL);
    gen->target(cugentgt);
    gen->train(pi);
  }


  if (nu > 0) {
    assert(encinlay->n == ctxlay->n + tgtlay->n);

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      encude(ctxbuf + mbi * ctxlay->n, ctxlay->n, cuencin + mbi * encinlay->n + 0);
      encude(tgtbuf + mbi * tgtlay->n, tgtlay->n, cuencin + mbi * encinlay->n + ctxlay->n);
    }
    assert(enc->outn == mbn * ctrlay->n);
    encude(ctrbuf, enc->outn, cuenctgt);

    enc->feed(cuencin, NULL);
    enc->target(cuenctgt);
    enc->train(nu);
  }
#endif



#if 1
  if (nu > 0 || pi > 0) {
    assert(encinlay->n == ctxlay->n + tgtlay->n);
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      encude(ctxbuf + mbi * ctxlay->n, ctxlay->n, cuencin + mbi * encinlay->n + 0);
      encude(tgtbuf + mbi * tgtlay->n, tgtlay->n, cuencin + mbi * encinlay->n + ctxlay->n);
    }
    assert(gen->outn == mbn * tgtlay->n);
    encude(tgtbuf, gen->outn, cugentgt);

    encgen->feed(cuencin, NULL);
    gen->target(cugentgt);

#if 1
    double *cugenfout = gen->foutput();
    for (unsigned int mbi = 0; mbi < mbn; ++mbi)
      cufocus(cugenfout + mbi * tgtlay->n, cutgtlayx, cutgtlayy, tgtlay->n);
#endif

    gen->train(pi);

    encpass->update_stats();
    encpass->train(nu);
  }
#endif
}

void Stage::condition(double yo, double wu) {
  assert(ctxlay->n >= 72);
  memcpy(fakectx, ctxbuf, mbn * ctxlay->n * sizeof(double));
  for (unsigned int mbi = 0; mbi < mbn; ++mbi)
    for (unsigned int j = mbi * ctxlay->n + 69, jn = j + 3; j < jn; ++j)
      fakectx[j] = 1.0;

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(fakectx + mbi * ctxlay->n, ctxlay->n, cuencin + mbi * encinlay->n + 0);
    encude( tgtbuf + mbi * tgtlay->n, tgtlay->n, cuencin + mbi * encinlay->n + ctxlay->n);
  }

  assert(gen->outn == mbn * tgtlay->n);
  encude(tgtbuf, gen->outn, cugentgt);

  encgen->feed(cuencin, NULL);
  encgen->target(cugentgt);
  gen->train(wu);

  encpass->update_stats();
  encpass->train(yo);
}


void Stage::report(const char *prog) {
  fprintf(
    stderr,
    "%s %s rounds=%u\n"
    "%s %s encgen_err2=%g encgen_errm=%g\n"
    "%s %s encpass_err2=%g encpass_errm=%g\n"
    "%s %s enc_err2=%g enc_errm=%g\n"
    "%s %s gen_err2=%g gen_errm=%g\n"
    "%s %s genpass_err2=%g genpass_errm=%g\n"
    "%s %s genenc_err2=%g genenc_errm=%g\n"
    "\n",
    prog, dir.c_str(), rounds,
    prog, dir.c_str(), encgen->err2, encgen->errm,
    prog, dir.c_str(), encpass->err2, encpass->errm,
    prog, dir.c_str(), enc->err2, enc->errm,
    prog, dir.c_str(), gen->err2, gen->errm,
    prog, dir.c_str(), genpass->err2, genpass->errm,
    prog, dir.c_str(), genenc->err2, genenc->errm
  );
}

void Stage::save() {
  enc->sync(1);
  gen->sync(1);
  dis->sync(1);
}

void Stage::load() {
  enc->sync(0);
  gen->sync(0);
  dis->sync(0);
}

void Stage::scramble(double mean, double dev) {
  for (unsigned int j = 0, jn = mbn * ctrlay->n; j < jn; ++j) {
    ctrbuf[j] = sigmoid( mean + randgauss() * dev );
  }
}

void Stage::encode_ctx() {
  for (unsigned int j = 0, jn = mbn * ctxlay->n; j < jn; ++j) {
    int v = (int)(ctxbuf[j] * 256.0);
    bctxbuf[j] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
}

void Stage::encode_out() {
  for (unsigned int j = 0, jn = mbn * outlay->n; j < jn; ++j) {
    int v = (int)(outbuf[j] * 256.0);
    boutbuf[j] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
}

void Stage::encode_tgt() {
  for (unsigned int j = 0, jn = mbn * tgtlay->n; j < jn; ++j) {
    int v = (int)(tgtbuf[j] * 256.0);
    btgtbuf[j] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
}

void Stage::encode_adj() {
  for (unsigned int j = 0, jn = mbn * outlay->n; j < jn; ++j) {
    double v = adjbuf[j];
    v /= 2.0;
    v += 0.5;
    v *= 256.0 * 128.0;
    v = lround(v);
    sadjbuf[j] = htons(v < 0 ? 0 : v > 65535 ? 65535 : v);
  }
}

void Stage::encode_ctr() {
  for (unsigned int j = 0, jn = mbn * ctrlay->n; j < jn; ++j) {
    int v = (int)(ctrbuf[j] * 256.0);
    bctrbuf[j] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
}

}
