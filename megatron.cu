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
  double *weight,
  double eta, double kappa,

  unsigned int inrn, unsigned int outrn, unsigned int mbn
) {
  unsigned int outri = blockIdx.x * blockDim.x + threadIdx.x;
  if (outri >= outrn)
    return;

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    unsigned int outi = mbi * outrn + outri;

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
} 

__global__ void gpu_megatron_train1(
  const double *in,
  double *fin, double *out, double *fout,
  unsigned int inn, unsigned int outn,
  unsigned int wn,
  unsigned int **iwmap, unsigned int **owmap,
  unsigned int **iomap, unsigned int **oimap,
  double *weight,
  double eta, double kappa,

  unsigned int inrn, unsigned int outrn, unsigned int mbn
) {
  unsigned int outri = blockIdx.x * blockDim.x + threadIdx.x;
  if (outri >= outn)
    return;

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    unsigned int outi = mbi * outrn + outri;

    unsigned int *inrip = oimap[outri];
    unsigned int *wip = owmap[outri];

    double o = out[outi];

    o -= 0.5;
    o /= kappa;
    o += 0.5;

    if (o > 1.0)
      o = 1.0;
    else if (o < 0.0)
      o = 0.0;
    fout[outi] = fout[outi] * o * (1.0 - o) / kappa;
    double z = eta * fout[outi];

    while (*inrip) {
      unsigned int ini = mbi * inrn + *inrip - 1;
      unsigned int wi = *wip;

      weight[wi] += z * in[ini];

      ++inrip;
      ++wip;
    }

    unsigned int wi = *wip;
    weight[wi] += z;
  }
}

__global__ void gpu_megatron_train2(
  const double *in,
  double *fin, double *out, double *fout,
  unsigned int inn, unsigned int outn,
  unsigned int wn,
  unsigned int **iwmap, unsigned int **owmap,
  unsigned int **iomap, unsigned int **oimap,
  double *weight,
  double eta, double kappa,

  unsigned int inrn, unsigned int outrn, unsigned int mbn
) {
  unsigned int inri = blockIdx.x * blockDim.x + threadIdx.x;
  if (inri >= inrn)
    return;
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    unsigned int ini = mbi * inrn + inri;

    unsigned int *outrip = iomap[inri];
    unsigned int *wip = iwmap[inri];

    double sum = 0;
    while (*outrip) {
      unsigned int outi = mbi * outrn + *outrip - 1;
      assert(outi < outn);
      unsigned int wi = *wip;

      sum += weight[wi] * fout[outi];

      ++outrip;
      ++wip;
    }

    fin[ini] += sum;
  }
}



const double *Megatron::feed(const double *_in, double *_fin) {
  in = _in;
  fin = _fin;

  int bs = 128;
  int gs = (outrn + bs  - 1) / bs;

  gpu_megatron_feed<<<gs, bs>>>(
    in, fin, out, fout, inn, outn,
    wn, iwmap, owmap, iomap, oimap,
    weight, eta, kappa,
    inrn, outrn, mbn
  );

  return out;
}


void Megatron::train(double nu) {
//fprintf(stderr, "kappa=%lf eta=%lf nu=%lf\n", kappa, eta, nu);
  int bs1 = 128;
  int gs1 = (outrn + bs1 - 1) / bs1;

  gpu_megatron_train1<<<gs1, bs1>>>(
    in, fin, out, fout, inn, outn,
    wn, iwmap, owmap, iomap, oimap,
    weight, nu * eta, kappa,
    inrn, outrn, mbn
  );

  if (fin) {
    int bs2 = 128;
    int gs2 = (inrn + bs2 - 1) / bs2;

    gpu_megatron_train2<<<gs2, bs2>>>(
      in, fin, out, fout, inn, outn,
      wn, iwmap, owmap, iomap, oimap,
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

  wn = wire->wn;
  cumake(&weight, wn);

  cweight = _cweight;
  encude(cweight, wn, weight);

  _makemaps();
}

Megatron::~Megatron() {
  cufree(out);
  cufree(fout);

  cufree(owmap);
  cufree(oimap);
  cufree(iwmap);
  cufree(iomap);

  cufree(weight);
}

void Megatron::_makemaps() {
  using namespace std;

  const vector< vector<unsigned int> > &moi = wire->moi;
  const vector< vector<unsigned int> > &mio = wire->mio;
  const vector< vector<unsigned int> > &miw = wire->miw;
  const vector< vector<unsigned int> > &mow = wire->mow;

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
}

void Megatron::randomize(double disp) {
  using namespace std;

  const vector< vector<unsigned int> > &mow = wire->mow;

  for (unsigned int outri = 0; outri < outrn; ++outri) {
    const vector<unsigned int>& w = mow[outri];
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

  encude(cweight, wn, weight);
}

void Megatron::sync() {
  ::decude(weight, wn, cweight);
}
