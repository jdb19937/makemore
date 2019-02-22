#define __MAKEMORE_CONFAB_CC__ 1

#include <string>
#include <netinet/in.h>

#include "cudamem.hh"
#include "tron.hh"
#include "multitron.hh"
#include "twiddle.hh"
#include "closest.hh"
#include "confab.hh"
#include "shibboleth.hh"

namespace makemore {

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

Confab::Confab(const char *_dir, unsigned int _mbn) {
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

  assert(!strcmp(config["type"].c_str(), "confab"));

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

  assert(tgtlay->n == outlay->n);

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

  gen->mt1->activated = false;
  enc->mt1->activated = true;

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

//  char vocabfn[4096];
//  sprintf(vocabfn, "%s/vocab.txt", _dir);
//  vocab.load(vocabfn);

  rounds = 0;
}

Confab::~Confab() {
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


void Confab::load_ctxtgt(FILE *infp) {
  int ret;

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


void Confab::load_ctx(FILE *infp) {
  int ret;

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    ret = fread(bctxbuf + mbi * ctxlay->n, 1, ctxlay->n, infp);
    assert(ret == ctxlay->n);
  }
  for (unsigned int j = 0, jn = mbn * ctxlay->n; j < jn; ++j)
    ctxbuf[j] = ((double)bctxbuf[j] + 0.5) / 256.0;
}



void Confab::generate(unsigned int reps) {
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude( ctxbuf + mbi * ctxlay->n, ctxlay->n, cugenin + mbi * geninlay->n + 0);
    encude( ctrbuf + mbi * ctrlay->n, ctrlay->n, cugenin + mbi * geninlay->n + ctxlay->n);
  }

  assert(geninlay->n == ctxlay->n + ctrlay->n);

  const double *cugenout = gen->feed(cugenin, NULL);
  decude(cugenout, gen->outn, tgtbuf);

  memcpy(outbuf, tgtbuf, sizeof(double) * mbn * tgtlay->n);
}


void Confab::passgenerate() {
  assert(outlay->n == tgtlay->n);
  memcpy(outbuf, tgtbuf, sizeof(double) * mbn * tgtlay->n);
}

void Confab::regenerate() {
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

void Confab::reencode() {
  const double *cuencout;

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(ctxbuf + mbi * ctxlay->n, ctxlay->n, cuencin + mbi * encinlay->n + 0);
    encude(tgtbuf + mbi * tgtlay->n, tgtlay->n, cuencin + mbi * encinlay->n + ctxlay->n);
  }

  cuencout = enc->feed(cuencin, NULL);
  assert(enc->outn == mbn * ctrlay->n);
  decude(cuencout, enc->outn, ctrbuf);
}

#if 0
void Confab::condition(double yo, double wu) {
  assert(mbn % 2 == 0);

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    for (unsigned int j = mbi * ctrlay->n, jn = j + ctrlay->n; j < jn; ++j)
      fakectr[j] = sigmoid(randgauss());
  }

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude( ctxbuf + mbi * ctxlay->n, ctxlay->n, cugenin + mbi * geninlay->n + 0);
    encude(fakectr + mbi * ctrlay->n, ctrlay->n, cugenin + mbi * geninlay->n + ctxlay->n);
  }
  const double *cugenout = gen->feed(cugenin, NULL);

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    for (unsigned int j = mbi * ctrlay->n, jn = j + ctrlay->n; j < jn; ++j)
      fakectr[j] = mbi % 2 ? 1.0 : 0.0;
  }

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(ctxbuf + mbi * ctxlay->n, ctxlay->n, cuencin + mbi * encinlay->n + 0);
    if (mbi % 2) {
      encude(tgtbuf + mbi * tgtlay->n, tgtlay->n, cuencin + mbi * encinlay->n + ctxlay->n);
    } else {
      cucopy(cugenout + mbi * tgtlay->n, tgtlay->n, cuencin + mbi * encinlay->n + ctxlay->n);
    }
  }

  encude(fakectr, enc->outn, cuenctgt);
  enc->feed(cuencin, NULL);
  enc->target(cuenctgt);
  enc->train(yo);



  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    for (unsigned int j = mbi * ctrlay->n, jn = j + ctrlay->n; j < jn; ++j)
      fakectr[j] = 1.0;
  }
  encude(fakectr, enc->outn, cuenctgt);
  genenc->feed(cugenin, NULL);
  enc->target(cuenctgt, false);
  enc->train(0);

  genpass->update_stats();
  genpass->train(wu);
}
#endif
void Confab::condition(double yo, double wu) {
  assert(mbn % 2 == 0);

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(ctxbuf + mbi * ctxlay->n, ctxlay->n, cuencin + mbi * encinlay->n + 0);
    encude(tgtbuf + mbi * tgtlay->n, tgtlay->n, cuencin + mbi * encinlay->n + ctxlay->n);
  }

  const double *cuencout = enc->feed(cuencin, NULL);
  decude(cuencout, enc->outn, fakectr);
  for (unsigned int mbi = 0; mbi < mbn; mbi += 2) {
    for (unsigned int j = mbi * ctrlay->n, jn = j + ctrlay->n; j < jn; ++j) {
      fakectr[j] = sigmoid(randgauss());
    } 
  }

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude( ctxbuf + mbi * ctxlay->n, ctxlay->n, cugenin + mbi * geninlay->n + 0);
    encude(fakectr + mbi * ctrlay->n, ctrlay->n, cugenin + mbi * geninlay->n + ctxlay->n);
  }

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    for (unsigned int j = mbi * ctrlay->n, jn = j + ctrlay->n; j < jn; ++j) {
      fakectr[j] = mbi % 2 ? 1.0 : 0.0;
    }
  }
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(fakectr + mbi * ctrlay->n, ctrlay->n, cuenctgt + mbi * ctrlay->n);
  }

  genenc->feed(cugenin, NULL);
  enc->target(cuenctgt);
  enc->train(yo);


  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    for (unsigned int j = mbi * ctrlay->n, jn = j + ctrlay->n; j < jn; ++j) {
      fakectr[j] = mbi % 2 ? 0.0 : 1.0;
    }
  }
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(fakectr + mbi * ctrlay->n, ctrlay->n, cuenctgt + mbi * ctrlay->n);
  }

  genenc->feed(cugenin, NULL);
  genenc->target(cuenctgt);
  enc->train(0);

  genpass->train(wu);
  
}

void Confab::burn(double nu, double pi) {
#if 1
  assert(encinlay->n == ctxlay->n + tgtlay->n);
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(ctxbuf + mbi * ctxlay->n, ctxlay->n, cuencin + mbi * encinlay->n + 0);
    encude(tgtbuf + mbi * tgtlay->n, tgtlay->n, cuencin + mbi * encinlay->n + ctxlay->n);
  }
  assert(gen->outn == mbn * tgtlay->n);
  encude(tgtbuf, gen->outn, cugentgt);

  encgen->feed(cuencin, NULL);
  encgen->target(cugentgt);
  gen->train(pi);

  encpass->update_stats();
  encpass->train(nu);
#endif

#if 0

  scramble(0,0);
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(ctxbuf + mbi * ctxlay->n, ctxlay->n, cugenin + mbi * geninlay->n + 0);
    encude(ctrbuf + mbi * ctrlay->n, ctrlay->n, cugenin + mbi * geninlay->n + ctxlay->n);
  }
  assert(gen->outn == mbn * tgtlay->n);
  encude(tgtbuf, gen->outn, cugentgt);

  gen->feed(cugenin, NULL);
  gen->target(cugentgt);
  gen->train(pi);
#endif
}


void Confab::report(const char *prog) {
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

void Confab::save() {
  enc->sync(1);
  gen->sync(1);
  dis->sync(1);
}

void Confab::load() {
  enc->sync(0);
  gen->sync(0);
  dis->sync(0);
}

void Confab::scramble(double mean, double dev) {
  for (unsigned int j = 0, jn = mbn * ctrlay->n; j < jn; ++j) {
    ctrbuf[j] = sigmoid( mean + randgauss() * dev );
  }
}

void Confab::encode_ctx() {
  for (unsigned int j = 0, jn = mbn * ctxlay->n; j < jn; ++j) {
    int v = (int)(ctxbuf[j] * 256.0);
    bctxbuf[j] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
}

void Confab::encode_out() {
  for (unsigned int j = 0, jn = mbn * outlay->n; j < jn; ++j) {
    int v = (int)(outbuf[j] * 256.0);
    boutbuf[j] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
}

void Confab::encode_tgt() {
  for (unsigned int j = 0, jn = mbn * tgtlay->n; j < jn; ++j) {
    int v = (int)(tgtbuf[j] * 256.0);
    btgtbuf[j] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
}

void Confab::encode_adj() {
  for (unsigned int j = 0, jn = mbn * outlay->n; j < jn; ++j) {
    double v = adjbuf[j];
    v /= 2.0;
    v += 0.5;
    v *= 256.0 * 128.0;
    v = lround(v);
    sadjbuf[j] = htons(v < 0 ? 0 : v > 65535 ? 65535 : v);
  }
}

void Confab::encode_ctr() {
  for (unsigned int j = 0, jn = mbn * ctrlay->n; j < jn; ++j) {
    int v = (int)(ctrbuf[j] * 256.0);
    bctrbuf[j] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
}

}
