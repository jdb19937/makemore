#define __MAKEMORE_MEGATRON_CU__ 1

#include <stdio.h>

#include <math.h>

#include <vector>
#include <map>

#include "megatron.hh"

__global__ void gpu_megatron_feed(
  const double *in,
  double *fin, double *out, double *fout,
  unsigned int inn, unsigned int outn,
  unsigned int wn,
  unsigned int **iwmap, unsigned int **owmap,
  unsigned int **iomap, unsigned int **oimap,
  double *weight,
  double eta, double kappa
) {
  unsigned int outi = blockIdx.x * blockDim.x + threadIdx.x;
  if (outi >= outn)
    return;

  unsigned int *inip = oimap[outi];
  unsigned int *wip = owmap[outi];

  double sum = 0;
  while (*inip) {
    unsigned int ini = *inip - 1;
    unsigned int wi = *wip;

    sum += weight[wi] * in[ini];

    ++inip;
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

__global__ void gpu_megatron_train1(
  const double *in,
  double *fin, double *out, double *fout,
  unsigned int inn, unsigned int outn,
  unsigned int wn,
  unsigned int **iwmap, unsigned int **owmap,
  unsigned int **iomap, unsigned int **oimap,
  double *weight,
  double eta, double kappa
) {
  unsigned int outi = blockIdx.x * blockDim.x + threadIdx.x;
  if (outi >= outn)
    return;

  unsigned int *inip = oimap[outi];
  unsigned int *wip = owmap[outi];

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

  while (*inip) {
    unsigned int ini = *inip - 1;
    unsigned int wi = *wip;

    weight[wi] += z * in[ini];

    ++inip;
    ++wip;
  }

  unsigned int wi = *wip;
  weight[wi] += z;
}

__global__ void gpu_megatron_train2(
  const double *in,
  double *fin, double *out, double *fout,
  unsigned int inn, unsigned int outn,
  unsigned int wn,
  unsigned int **iwmap, unsigned int **owmap,
  unsigned int **iomap, unsigned int **oimap,
  double *weight,
  double eta, double kappa
) {
  unsigned int ini = blockIdx.x * blockDim.x + threadIdx.x;
  if (ini >= inn)
    return;

  unsigned int *outip = iomap[ini];
  unsigned int *wip = iwmap[ini];

  double sum = 0;
  while (*outip) {
    unsigned int outi = *outip - 1;
    assert(outi < outn);
    unsigned int wi = *wip;

    sum += weight[wi] * fout[outi];

    ++outip;
    ++wip;
  }

  fin[ini] += sum;
}


const double *Megatron::feed(const double *_in, double *_fin) {
  in = _in;
  fin = _fin;

  int bs = 128;
  int gs = (outn + bs - 1) / bs;
  gpu_megatron_feed<<<gs, bs>>>(
    in, fin, out, fout, inn, outn,
    wn, iwmap, owmap, iomap, oimap,
    weight, eta, kappa
  );

  return out;
}


#include "cudamem.hh"

void Megatron::train(double nu) {
//fprintf(stderr, "kappa=%lf eta=%lf nu=%lf\n", kappa, eta, nu);
  int bs1 = 128;
  int gs1 = (outn + bs1 - 1) / bs1;
  gpu_megatron_train1<<<gs1, bs1>>>(
    in, fin, out, fout, inn, outn,
    wn, iwmap, owmap, iomap, oimap,
    weight, nu * eta, kappa
  );

  if (fin) {
    int bs2 = 128;
    int gs2 = (inn + bs2 - 1) / bs2;
    gpu_megatron_train2<<<gs2, bs2>>>(
      in, fin, out, fout, inn, outn,
      wn, iwmap, owmap, iomap, oimap,
      weight, nu * eta, kappa
    );
  }
}

Megatron::Megatron(const Wiring *_wire) : Tron(_wire->inn, _wire->outn) {
  wire = _wire;
  inl = wire->inl;
  outl = wire->outl;

  cumake(&out, outn);
  cumake(&fout, outn);

  cumake(&owmap, outn);
  cumake(&oimap, outn);
  cumake(&iomap, inn);
  cumake(&iwmap, inn);

  kappa = 1.0;
  eta = 1.0;

  double disp = 4.0;
  _makemaps(disp);
}

Megatron::~Megatron() {
  cufree(out);
  cufree(fout);

  cufree(owmap);
  cufree(oimap);
  cufree(iwmap);
  cufree(iomap);
}

void Megatron::_makemaps(double disp) {
  using namespace std;

  const vector< vector<unsigned int> > &moi = wire->moi;
  const vector< vector<unsigned int> > &mio = wire->mio;
  const vector< vector<unsigned int> > &miw = wire->miw;
  const vector< vector<unsigned int> > &mow = wire->mow;

  wn = wire->wn;
  cumake(&weight, wn);
  double *cweight = new double[wn];

  unsigned int *tmp;

  for (unsigned int outi = 0; outi < outn; ++outi) {
    const vector<unsigned int>& v = moi[outi];
    const vector<unsigned int>& w = mow[outi];
    assert(w.size());

    double iss = disp / sqrt(w.size() + 1);
    double sw = 0;
    for (unsigned int i = 0; i < w.size() - 1; ++i) {
      double ww = iss * rnd(-1, 1);
      cweight[w[i]] = ww;
      sw += ww;
    }
    cweight[w[w.size() - 1]] = 0; //-sw/2.0;

    tmp = cunew<unsigned int>(v.size());
    encude(v.data(), v.size(), tmp);
    encude(&tmp, 1, oimap + outi);

    tmp = cunew<unsigned int>(w.size());
    encude(w.data(), w.size(), tmp);
    encude(&tmp, 1, owmap + outi);
  }

  encude(cweight, wn, weight);
  delete[] cweight;

  for (unsigned int ini = 0; ini < inn; ++ini) {
    const vector<unsigned int>& v = mio[ini];
    const vector<unsigned int>& w = miw[ini];

    tmp = cunew<unsigned int>(v.size());
    encude(v.data(), v.size(), tmp);
    encude(&tmp, 1, iomap + ini);

    tmp = cunew<unsigned int>(w.size());
    encude(w.data(), w.size(), tmp);
    encude(&tmp, 1, iwmap + ini);
  }
}

