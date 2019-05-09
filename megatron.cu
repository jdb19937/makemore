#define __MAKEMORE_MEGATRON_CU__ 1

#include <stdio.h>

#include <math.h>

#include <vector>
#include <map>

#include "cudamem.hh"
#include "megatron.hh"
#include "mapfile.hh"

namespace makemore {

double adam_b1 = 0.9;
double adam_b2 = 0.999;
double adam_b3 = 0.5;
double adam_eps = 1e-8;


__global__ void gpu_megatron_feed(
  const double *in,
  double *fin, double *out, double *fout,
  unsigned int inn, unsigned int outn,
  unsigned int wn,
  unsigned int **iwmap, unsigned int **owmap,
  unsigned int **iomap, unsigned int **oimap,
  unsigned int *wimap, unsigned int *womap,
  double *weight,
  double eta, double nu, bool activated,

  unsigned int inrn, unsigned int outrn, unsigned int mbn
) {
  unsigned int outi = blockIdx.x * blockDim.x + threadIdx.x;
  if (outi >= outn)
    return;
  unsigned int outri = outi % outrn;
  unsigned int mbi = outi / outrn;

  unsigned int *inrip = oimap[outri];
  unsigned int *wip = owmap[outri];

  double sum = 0;
  while (*inrip) {
    unsigned int ini = mbi * inrn + *inrip - 1;
    unsigned int wi = *wip;

    sum += weight[wi] * in[ini];

    ++inrip;
    ++wip;
  }

  unsigned int wi = *wip;
  sum += weight[wi] * 1.0;

  if (activated) {
    double q = 1.0 / (1.0 + exp(-sum));
    out[outi] = q;
  } else {
    out[outi] = sum;
  }

  fout[outi] = 0.0;
} 

__global__ void gpu_megatron_train0(
  const double *in,
  double *fin, double *out, double *fout,
  unsigned int inn, unsigned int outn,
  unsigned int wn,
  unsigned int **iwmap, unsigned int **owmap,
  unsigned int **iomap, unsigned int **oimap,
  unsigned int *wimap, unsigned int *womap,
  double *weight,
  double eta, double nu, bool activated,

  unsigned int inrn, unsigned int outrn, unsigned int mbn
) {
  unsigned int outi = blockIdx.x * blockDim.x + threadIdx.x;
  if (outi >= outn)
    return;

  double o = out[outi];
  double fo = fout[outi];

  if (o > 1.0)
    o = 1.0;
  else if (o < 0.0)
    o = 0.0;

  fout[outi] = fo * o * (1.0 - o);
}


__global__ void gpu_megatron_train1(
  const double *in,
  double *fin, double *out, double *fout,
  unsigned int inn, unsigned int outn,
  unsigned int wn,
  unsigned int **iwmap, unsigned int **owmap,
  unsigned int **iomap, unsigned int **oimap,
  unsigned int *wimap, unsigned int *womap,
  double *weight,
  double eta, double nu, bool activated,

  unsigned int inrn, unsigned int outrn, unsigned int mbn
) {
  unsigned int ini = blockIdx.x * blockDim.x + threadIdx.x;
  if (ini >= inn)
    return;

  unsigned int inri = ini % inrn;
  unsigned int mbi = ini / inrn;

  unsigned int *outrip = iomap[inri];
  unsigned int *wip = iwmap[inri];

  double sum = 0;
  while (*outrip) {
    unsigned int outi = mbi * outrn + *outrip - 1;
    unsigned int wi = *wip;

    sum += weight[wi] * fout[outi];

    ++outrip;
    ++wip;
  }

  fin[ini] += sum;
}


__global__ void gpu_megatron_train2(
  const double *in,
  double *fin, double *out, double *fout,
  unsigned int inn, unsigned int outn,
  unsigned int wn,
  unsigned int **iwmap, unsigned int **owmap,
  unsigned int **iomap, unsigned int **oimap,
  unsigned int *wimap, unsigned int *womap,
  double *weight,

  double *m, double *v,  double a, double b1, double b2, double b3, double eps,

  bool activated,

  unsigned int inrn, unsigned int outrn, unsigned int mbn
) {
  unsigned int wi = blockIdx.x * blockDim.x + threadIdx.x;
  if (wi >= wn)
    return;

if (!(a > 0))
  return;


  unsigned int outri = womap[wi];
  --outri;
  unsigned int inri = wimap[wi];

  if (inri == 0) {
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      unsigned int outi = mbi * outrn + outri;
      double dw = fout[outi];
  m[wi] = b1 * m[wi] + (1 - b1) * dw;
  v[wi] = b2 * v[wi] + (1 - b2) * dw * dw;
  weight[wi] += a * m[wi] / (pow(v[wi], b3) + eps);
//weight[wi] += a * dw;
    }
  } else {
    --inri;

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      unsigned int outi = mbi * outrn + outri;
      unsigned int ini = mbi * inrn + inri;
      double dw = fout[outi] * in[ini];
  m[wi] = b1 * m[wi] + (1 - b1) * dw;
  v[wi] = b2 * v[wi] + (1 - b2) * dw * dw;
  weight[wi] += a * m[wi] / (pow(v[wi], b3) + eps);
//weight[wi] += a * dw;
    }
  }

}

const double *Megatron::feed(const double *_in, double *_fin) {
  in = _in;
  fin = _fin;

  int bs = 128;
  int gs = (outn + bs - 1) / bs;

  gpu_megatron_feed<<<gs, bs>>>(
    in, fin, out, fout, inn, outn,
    wn, iwmap, owmap, iomap, oimap, wimap, womap,
    weight, eta, 1.0, activated,
    inrn, outrn, mbn
  );

  return out;
}


void Megatron::train(double nu) {
  if (activated) {
    int bs0 = 128;
    int gs0 = (outn + bs0 - 1) / bs0;
    gpu_megatron_train0<<<gs0, bs0>>>(
      in, fin, out, fout, inn, outn,
      wn, iwmap, owmap, iomap, oimap, wimap, womap,
      weight, eta, nu, activated,
      inrn, outrn, mbn
    );
  }

  if (fin) {
    int bs1 = 128;
    int gs1 = (inn + bs1 - 1) / bs1;
    gpu_megatron_train1<<<gs1, bs1>>>(
      in, fin, out, fout, inn, outn,
      wn, iwmap, owmap, iomap, oimap, wimap, womap,
      weight, eta, nu, activated,
      inrn, outrn, mbn
    );
  }

  int bs2 = 128;
  int gs2 = (wn + bs2 - 1) / bs2;
  gpu_megatron_train2<<<gs2, bs2>>>(
    in, fin, out, fout, inn, outn,
    wn, iwmap, owmap, iomap, oimap, wimap, womap,
    weight,
m,v, eta*nu, adam_b1, adam_b2, adam_b3, adam_eps,
    activated,
    inrn, outrn, mbn
  );
}

Megatron::Megatron(const Wiring *_wire, Mapfile *_mapfile, unsigned int _mbn, double _eta, bool _activated)
  : Tron(_wire->inn * _mbn, _wire->outn * _mbn)
{
  mbn = _mbn;
  assert(mbn > 0);
  assert(inn % mbn == 0);
  inrn = inn / mbn;
  assert(outn % mbn == 0);
  outrn = outn / mbn;

  wire = _wire;
  mapfile = _mapfile;

  cumake(&out, outn);
  cumake(&fout, outn);

  cumake(&owmap, outrn);
  cumake(&oimap, outrn);
  cumake(&iomap, inrn);
  cumake(&iwmap, inrn);

  eta = _eta;
  activated = _activated;

  _makemaps();

  cumake(&weight, wn);
  mapfile->map(weight, wn);
  mapfile->load(weight);

  cumake(&m, wn);
  mapfile->map(m, wn);
  mapfile->load(m);

  cumake(&v, wn);
  mapfile->map(v, wn);
  mapfile->load(v);
}

Megatron::~Megatron() {
  cufree(out);
  cufree(fout);

  cufree(owmap);
  cufree(oimap);
  cufree(iwmap);
  cufree(iomap);
  cufree(womap);
  cufree(wimap);

  cufree(weight);
}

void Megatron::_makemaps() {
  using namespace std;

  vector< vector<unsigned int> > moi, mow, mio, miw;
  assert(wire->outn == outrn);
  assert(wire->inn == inrn);
  moi.resize(wire->outn);
  mow.resize(wire->outn);
  mio.resize(wire->inn);
  miw.resize(wire->inn);

  unsigned int wi = 0;
  wire->_makemaps(mio, miw, moi, mow);
  wi += wire->wn;

  for (unsigned int outri = 0; outri < wire->outn; ++outri) {
    moi[outri].push_back(0);
    mow[outri].push_back(wi);

    ++wi;
  }

  wn = wi;
  std::vector<unsigned int> mwi, mwo;
  mwi.resize(wn);
  mwo.resize(wn);

  for (unsigned int outri = 0; outri < wire->outn; ++outri) {
    for (auto wip = mow[outri].begin(); wip != mow[outri].end(); ++wip) {
      mwo[*wip] = outri + 1;
      mwi[*wip] = 0;
    }
  }
  for (unsigned int inri = 0; inri < wire->inn; ++inri) {
    for (auto wip = miw[inri].begin(); wip != miw[inri].end(); ++wip) {
      mwi[*wip] = inri + 1;
    }
  }

  for (unsigned int inri = 0; inri < wire->inn; ++inri) {
    mio[inri].push_back(0);
    miw[inri].push_back((unsigned int)-1);
  }

    
  unsigned int mapbufn = 0;
  for (unsigned int outri = 0; outri < outrn; ++outri) {
    const vector<unsigned int>& v = moi[outri];
    const vector<unsigned int>& w = mow[outri];
    assert(v.size());
    mapbufn += v.size();
    assert(w.size());
    mapbufn += w.size();
  }

  for (unsigned int inri = 0; inri < inrn; ++inri) {
    const vector<unsigned int>& v = mio[inri];
    const vector<unsigned int>& w = miw[inri];
    assert(v.size());
    mapbufn += v.size();
    assert(w.size());
    mapbufn += w.size();
  }

  cumake(&mapbuf, mapbufn);
  unsigned int mapbufi = 0;
  unsigned int *cmapbuf = new unsigned int[mapbufn];

  unsigned int **coimap = new unsigned int *[outrn];
  unsigned int **cowmap = new unsigned int *[outrn];
  unsigned int **ciomap = new unsigned int *[inrn];
  unsigned int **ciwmap = new unsigned int *[inrn];

  for (unsigned int outri = 0; outri < outrn; ++outri) {
    const vector<unsigned int>& v = moi[outri];
    const vector<unsigned int>& w = mow[outri];

    memcpy(cmapbuf + mapbufi, v.data(), v.size() * sizeof(unsigned int));
    coimap[outri] = mapbuf + mapbufi;
    mapbufi += v.size();

    memcpy(cmapbuf + mapbufi, w.data(), w.size() * sizeof(unsigned int));
    cowmap[outri] = mapbuf + mapbufi;
    mapbufi += w.size();
  }

  for (unsigned int inri = 0; inri < inrn; ++inri) {
    const vector<unsigned int>& v = mio[inri];
    const vector<unsigned int>& w = miw[inri];

    memcpy(cmapbuf + mapbufi, v.data(), v.size() * sizeof(unsigned int));
    ciomap[inri] = mapbuf + mapbufi;
    mapbufi += v.size();

    memcpy(cmapbuf + mapbufi, w.data(), w.size() * sizeof(unsigned int));
    ciwmap[inri] = mapbuf + mapbufi;
    mapbufi += w.size();
  }
  assert(mapbufi == mapbufn);

  encude(cmapbuf, mapbufn, mapbuf);
  delete[] cmapbuf; 

  encude(coimap, outrn, oimap);
  encude(cowmap, outrn, owmap);
  encude(ciomap, inrn, iomap);
  encude(ciwmap, inrn, iwmap);
  delete[] ciomap;
  delete[] ciwmap;
  delete[] coimap;
  delete[] cowmap;

  _mow = mow;

  cumake(&wimap, wn);
  encude(mwi.data(), wn, wimap);

  cumake(&womap, wn);
  encude(mwo.data(), wn, womap);
}

void Megatron::randomize(double disp) {
  using namespace std;

  double *cweight = new double[wn];

  for (unsigned int outri = 0; outri < outrn; ++outri) {
    const vector<unsigned int>& w = _mow[outri];
    assert(w.size());

    double iss = disp / sqrt(w.size() - 1);
    double sw = 0;
    for (unsigned int i = 0; i < w.size() - 1; ++i) {
      // double ww = iss * rnd(-1, 1);
      double ww = iss * randgauss();
      assert(w[i] < wn);
      cweight[w[i]] = ww;
      sw += ww;
    }
    assert(w[w.size() - 1] < wn);
//    cweight[w[w.size() - 1]] = 0;
    cweight[w[w.size() - 1]] = -sw/2.0;
  }

  encude(cweight, wn, weight);
  delete[] cweight;

  cuzero(m, wn);
  double *one = new double[wn];
  for (unsigned int wi = 0; wi < wn; ++wi)
    one[wi] = 1.0;
  encude(one, wn, v);
  delete[] one;
}

}
