#define __MAKEMORE_MEGATRON_CU__ 1

#include <stdio.h>

#include <math.h>

#include <vector>
#include <map>

#include "cudamem.hh"
#include "megatron.hh"

__global__ void gpu_megatron_feed(
  const double *in,
  double *fin, double *out, double *fout,
  unsigned int inn, unsigned int outn,
  unsigned int wn,
  unsigned int **iwmap, unsigned int **owmap,
  unsigned int **iomap, unsigned int **oimap,
  unsigned int *wimap, unsigned int *womap,
  double *weight,
  double eta, double kappa,

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

  double q = 1.0 / (1.0 + exp(-sum));

  q -= 0.5;
  q *= kappa;
  q += 0.5;

  out[outi] = q;
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
  double eta, double kappa,

  unsigned int inrn, unsigned int outrn, unsigned int mbn
) {
  unsigned int outi = blockIdx.x * blockDim.x + threadIdx.x;
  if (outi >= outn)
    return;

  double o = out[outi];

  o -= 0.5;
  o /= kappa;
  o += 0.5;

  if (o > 1.0)
    o = 1.0;
  else if (o < 0.0)
    o = 0.0;

  double fo = fout[outi];
  fout[outi] = fo * o * (1.0 - o) / kappa;
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
  double eta, double kappa,

  unsigned int inrn, unsigned int outrn, unsigned int mbn
) {
  unsigned int wi = blockIdx.x * blockDim.x + threadIdx.x;
  if (wi >= wn)
    return;

  double dw = 0;

  unsigned int outri = womap[wi];
  --outri;

  unsigned int inri = wimap[wi];
  if (inri == 0) {
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      unsigned int outi = mbi * outrn + outri;
      dw += eta * fout[outi];
    }
    weight[wi] += dw;
    return;
  }
  --inri;

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    unsigned int outi = mbi * outrn + outri;
    double z = eta * fout[outi];

    unsigned int ini = mbi * inrn + inri;
    dw += z * in[ini];
  }
  weight[wi] += dw;
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
  double eta, double kappa,

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



const double *Megatron::feed(const double *_in, double *_fin) {
  in = _in;
  fin = _fin;

  int bs = 128;
  int gs = (outn + bs  - 1) / bs;

  gpu_megatron_feed<<<gs, bs>>>(
    in, fin, out, fout, inn, outn,
    wn, iwmap, owmap, iomap, oimap, wimap, womap,
    weight, eta, kappa,
    inrn, outrn, mbn
  );

  return out;
}


void Megatron::train(double nu) {
  int bs0 = 128;
  int gs0 = (outn + bs0 - 1) / bs0;
  gpu_megatron_train0<<<gs0, bs0>>>(
    in, fin, out, fout, inn, outn,
    wn, iwmap, owmap, iomap, oimap, wimap, womap,
    weight, nu * eta, kappa,
    inrn, outrn, mbn
  );

  int bs1 = 128;
  int gs1 = (wn + bs1 - 1) / bs1;
  gpu_megatron_train1<<<gs1, bs1>>>(
    in, fin, out, fout, inn, outn,
    wn, iwmap, owmap, iomap, oimap, wimap, womap,
    weight, nu * eta, kappa,
    inrn, outrn, mbn
  );

  if (fin) {
    int bs2 = 128;
    int gs2 = (inn + bs2 - 1) / bs2;
    gpu_megatron_train2<<<gs2, bs2>>>(
      in, fin, out, fout, inn, outn,
      wn, iwmap, owmap, iomap, oimap, wimap, womap,
      weight, nu * eta, kappa,
      inrn, outrn, mbn
    );
  }
}

Megatron::Megatron(const Wiring *_wire, double *_cweight, unsigned int _mbn)
  : Tron(_wire->inn * _mbn, _wire->outn * _mbn)
{
  mbn = _mbn;
  assert(mbn > 0);
  assert(inn % mbn == 0);
  inrn = inn / mbn;
  assert(outn % mbn == 0);
  outrn = outn / mbn;

  wire = _wire;

  cumake(&out, outn);
  cumake(&fout, outn);

  cumake(&owmap, outrn);
  cumake(&oimap, outrn);
  cumake(&iomap, inrn);
  cumake(&iwmap, inrn);

  kappa = 1.0;
  eta = 1.0;

  _makemaps();

  cumake(&weight, wn);
  cweight = _cweight;
  sync(0);
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
  for (auto ci = wire->connected.begin(); ci != wire->connected.end(); ++ci) {
    unsigned int inri = ci->first;
    unsigned int outri = ci->second;

    mio[inri].push_back(outri + 1);
    miw[inri].push_back(wi);
    mow[outri].push_back(wi);
    moi[outri].push_back(inri + 1);

    ++wi;
  }

  for (unsigned int outri = 0; outri < wire->outn; ++outri) {
    moi[outri].push_back(0);
    mow[outri].push_back(wi);

    ++wi;
  }

  for (unsigned int inri = 0; inri < wire->inn; ++inri) {
    mio[inri].push_back(0);
    miw[inri].push_back((unsigned int)-1);
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
      if (*wip != (unsigned)-1)
        mwi[*wip] = inri + 1;
    }
  }
    
  unsigned int *tmp;

  for (unsigned int outri = 0; outri < outrn; ++outri) {
    const vector<unsigned int>& v = moi[outri];
    const vector<unsigned int>& w = mow[outri];
    assert(w.size());

    tmp = cunew<unsigned int>(v.size());
    encude(v.data(), v.size(), tmp);
    encude(&tmp, 1, oimap + outri);

    tmp = cunew<unsigned int>(w.size());
    encude(w.data(), w.size(), tmp);
    encude(&tmp, 1, owmap + outri);
  }

  for (unsigned int inri = 0; inri < inrn; ++inri) {
    const vector<unsigned int>& v = mio[inri];
    const vector<unsigned int>& w = miw[inri];

    tmp = cunew<unsigned int>(v.size());
    encude(v.data(), v.size(), tmp);
    encude(&tmp, 1, iomap + inri);

    tmp = cunew<unsigned int>(w.size());
    encude(w.data(), w.size(), tmp);
    encude(&tmp, 1, iwmap + inri);
  }

  _mow = mow;

  wimap = cunew<unsigned int>(wn);
  encude(mwi.data(), wn, wimap);

  womap = cunew<unsigned int>(wn);
  encude(mwo.data(), wn, womap);
}

void Megatron::randomize(double disp) {
  using namespace std;

  for (unsigned int outri = 0; outri < outrn; ++outri) {
    const vector<unsigned int>& w = _mow[outri];
    assert(w.size());

    double iss = disp / sqrt(w.size() + 1);
    double sw = 0;
    for (unsigned int i = 0; i < w.size() - 1; ++i) {
      double ww = iss * rnd(-1, 1);
      cweight[w[i]] = ww;
      sw += ww;
    }
    cweight[w[w.size() - 1]] = 0; //-sw/2.0;
  }
  sync(0);
}

void Megatron::sync(double t) {
  if (t == 1) {
    ::decude(weight, wn, cweight);
    return;
  }

  if (t == 0) {
    ::encude(cweight, wn, weight);
    return;
  }

  assert(0);
  // cusync(wn, weight, cweight, t);
}
