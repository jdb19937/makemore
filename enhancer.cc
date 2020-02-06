#define __MAKEMORE_ENHANCER_CC__ 1

#include <string>
#include <algorithm>

#include "cudamem.hh"
#include "tron.hh"
#include "multitron.hh"
#include "encoder.hh"
#include "zoomgen.hh"
#include "parson.hh"
#include "strutils.hh"
#include "imgutils.hh"
#include "cholo.hh"
#include "numutils.hh"
#include "zoomdis.hh"
#include "enhancer.hh"
#include "pic.hh"

namespace makemore {

using namespace std;

Enhancer::Enhancer(const std::string &_dir) : Project(_dir, 1) {
  assert(mbn > 0);

  assert(config["type"] == "enhancer");

  char genmapfn[4096];
  sprintf(genmapfn, "%s/gen.map", dir.c_str());
  genmap = new Mapfile(genmapfn);
  gen = new Supertron(genmap);

  char dismapfn[4096];
  sprintf(dismapfn, "%s/dis.map", dir.c_str());
  dismap = new Mapfile(dismapfn);
  dis = new Supertron(dismap);

  cumake(&cugenin, gen->inn);
  cumake(&cudistgt, dis->outn);
  cumake(&cudisin, dis->inn);
  cumake(&cudisfin, dis->inn);
  cumake(&cugentgt, gen->outn);

  rounds = 0;
}

Enhancer::~Enhancer() {
  cufree(cugenin);
  cufree(cudistgt);
  cufree(cudisin);
  cufree(cudisfin);
  cufree(cugentgt);
}


void Enhancer::report(const char *prog) {
  fprintf(
    stderr,
    "%s %s rounds=%u\n"
    "gen_err2=%g gen_errm=%g\n"
    "dis_err2=%g dis_errm=%g\n",
    prog, dir.c_str(), rounds,
    gen->err2, gen->errm,
    dis->err2, dis->errm
  );
}

void Enhancer::save() {
  genmap->save();
  dismap->save();
}

void Enhancer::load() {
  genmap->load();
  dismap->load();
}

void Enhancer::observe(const double *cudsamp, double nu) {
  const double nh = 0.5;
  double *scbuf = new double[dis->outn];

#if 0
assert(dis->layers[0]->wn >= 10);
double weight[10];
decude(dis->layers[0]->weight, 10, weight);
for (unsigned int j = 0;j < 10; ++j)
fprintf(stderr, "%lf ", weight[j]);
fprintf(stderr, "\n");
#endif



  double genmul = 1.0;
  double dismul = 1.0;

//   double genmul = dis->err2 > 0.5 ? 0.0 : 2.0 * (0.5 - dis->err2);
//   double dismul = dis->err2 > 0.5 ? 1.0 : 2.0 * dis->err2;

  // double dismul = 1.0;
  // double genmul = 0.0;

#if 0
  assert(dis->outn == 32 * 32 * 3);
  cusplice(cudsamp, 32 * 32, 9, 0, 3, cudistgt, 3, 0);
#endif
#if 1
  cucopy(cudsamp, dis->inn, cudisin);
  #if 1
    cusplice(cudsamp, 32 * 32, 9, 3, 3, cugentgt, 3, 0);
    cuaddnoise(cugentgt, 32 * 32 * 3, nh);
    cusplice(cugentgt, 32 * 32, 3, 0, 3, cudisin, 9, 3);
  #endif
  dis->feed(cudisin, NULL);
  cucopy(dis->output(), dis->outn, cudistgt);
#endif
#if 0
  for (unsigned int j = 0; j < dis->outn; ++j)
    scbuf[j] = 0.5;
  encude(scbuf, dis->outn, cudistgt);
#endif
#if 0
  for (unsigned int j = 0; j < dis->outn; ++j)
    scbuf[j] = 0;
  encude(scbuf, dis->outn, cudistgt);
#endif



  assert(gen->inn == 32 * 32 * 6);
  cusplice(cudsamp, 32 * 32, 9, 3, 6, cugenin, 6, 0);
  gen->feed(cugenin, NULL);

#if 0
  cuspliceadd(cugenin, 32 * 32, 6, 0, 3, gen->output(), 3, 0);
#endif

  cucopy(cudsamp, dis->inn, cudisin);
#if 1
  cusplice(cudsamp, 32 * 32, 9, 3, 3, cugentgt, 3, 0);
  cuaddnoise(cugentgt, 32 * 32 * 3, nh);
  cusplice(cugentgt, 32 * 32, 3, 0, 3, cudisin, 9, 3);
#endif

  cusplice(gen->output(), 32 * 32, 3, 0, 3, cudisin, 9, 0);
  cuzero(cudisfin, dis->inn);
  dis->feed(cudisin, cudisfin);


#if 0
  assert(dis->outn == 32 * 32);
  double *tmpd = new double[dis->outn];
  decude(dis->output(), dis->outn, tmpd);
  for (unsigned int j = 0; j < dis->outn; ++j)
    tmpd[j] += 0.5;
  uint8_t *tmpb = new uint8_t[dis->outn];
  dtobv(tmpd, tmpb, dis->outn);
  FILE *fp = fopen("tmp.pgm", "w");
  fprintf(fp, "P5\n%d %d\n255\n", 32, 32);
  fwrite(tmpb, 1, 32 * 32, fp);
  fclose(fp);
  delete[] tmpb;
  delete[] tmpd;
#endif

  dis->target(cudistgt);
  dis->train(0);

  cusplice(cudisfin, 32 * 32, 9, 0, 3, gen->foutput(), 3, 0);
  gen->update_stats();
  gen->train(nu * genmul);



#if 1
  cusplice(cudsamp, 32 * 32, 9, 3, 3, cugentgt, 3, 0);
  cuaddnoise(cugentgt, 32 * 32 * 3, nh);
  cusplice(cugentgt, 32 * 32, 3, 0, 3, cudisin, 9, 3);
#endif



  dis->feed(cudisin, NULL);
#if 1
  for (unsigned int j = 0; j < dis->outn; ++j)
    scbuf[j] = 1.0;
  encude(scbuf, dis->outn, cudistgt);
#endif
#if 0
  for (unsigned int j = 0; j < dis->outn; ++j)
    scbuf[j] = -0.5;
  encude(scbuf, dis->outn, cudistgt);
#endif
#if 0
  decude(dis->output(), dis->outn, scbuf);
  for (unsigned int j = 0; j < dis->outn; ++j) {
    if (scbuf[j] > 0) {
      scbuf[j] += sigmoid(-scbuf[j]);
    } else if (scbuf[j] < 0) {
      scbuf[j] += -sigmoid(scbuf[j]);
    }
  }
  encude(scbuf, dis->outn, cudistgt);
#endif
#if 0
  decude(dis->output(), dis->outn, scbuf);
  for (unsigned int j = 0; j < dis->outn; ++j) {
    scbuf[j] += randgauss() / (scbuf[j] * scbuf[j] + 1e-2);
  }
  encude(scbuf, dis->outn, cudistgt);
#endif
#if 0
  assert(dis->outn == 32 * 32 * 3);
  cusplice(cudsamp, 32 * 32, 9, 0, 3, cudistgt, 3, 0);
  cusubvec(cudistgt, gen->output(), 32 * 32 * 3, cudistgt);
  // cumulvec(cudistgt, cudistgt, 32 * 32 * 3, cudistgt);
  cuabsvec(cudistgt, 32 * 32 * 3);
  // culog1vec(cudistgt, 32 * 32 * 3);
  // cuclampvec(cudistgt, 32 * 32 * 3, 4.0);
#endif

  dis->target(cudistgt);
  dis->update_stats();
  dis->train(nu * dismul);



  cucopy(cudsamp, dis->inn, cudisin);
#if 1
  cusplice(cudsamp, 32 * 32, 9, 3, 3, cugentgt, 3, 0);
  cuaddnoise(cugentgt, 32 * 32 * 3, nh);
  cusplice(cugentgt, 32 * 32, 3, 0, 3, cudisin, 9, 3);
#endif


  dis->feed(cudisin, NULL);
#if 0
  for (unsigned int j = 0; j < dis->outn; ++j)
    scbuf[j] = 0.5;
  encude(scbuf, dis->outn, cudistgt);
#endif
#if 1
  for (unsigned int j = 0; j < dis->outn; ++j)
    scbuf[j] = 0;
  encude(scbuf, dis->outn, cudistgt);
#endif
  dis->target(cudistgt);
  dis->update_stats();
  dis->train(nu * dismul);

  delete[] scbuf;
}

void Enhancer::generate(const Partrait *spic, Partrait *tpicp) {
  assert(gen->inn == spic->h * spic->w * 6);

  const uint8_t *srgb = spic->rgb;
  double *rgbwxy = new double[gen->inn], *rgbwxyp = rgbwxy;

  for (unsigned int y = 0; y < spic->h; ++y) {
  for (unsigned int x = 0; x < spic->w; ++x) {

    *rgbwxyp++ = (double)*srgb++ / 255.0;
    *rgbwxyp++ = (double)*srgb++ / 255.0;
    *rgbwxyp++ = (double)*srgb++ / 255.0;

    double qx = (double)x / (double)spic->w;
    double qw = qx < 0.5 ? (2.0 * qx) : 2.0 * (1.0 - qx);
    double qy = (double)y / (double)spic->h;

    *rgbwxyp++ = qw;
    *rgbwxyp++ = qx;
    *rgbwxyp++ = qy;

  }}

  encude(rgbwxy, gen->inn, cugenin);
  delete[] rgbwxy;

  gen->feed(cugenin, NULL);

#if 0
  cuspliceadd(cugenin, spic->w * spic->h, 6, 0, 3, gen->output(), 3, 0);
#endif


  if (tpicp) {
    double *rgb = new double[gen->outn];
    decude(gen->output(), gen->outn, rgb);
    tpicp->create(spic->w, spic->h);
    dtobv(rgb, tpicp->rgb, gen->outn);
    delete[] rgb;
  }
}

}
