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

Megatron::Megatron(const Layout *_inl, const Layout *_outl) : Tron(_inl->n, _outl->n) {
  inl = _inl;
  outl = _outl;

  cudaMalloc((void **)&out, sizeof(double) * outn);
  cudaMalloc((void **)&fout, sizeof(double) * outn);

  cudaMalloc((void **)&owmap, sizeof(unsigned int *) * outn);
  cudaMalloc((void **)&oimap, sizeof(unsigned int *) * outn);
  cudaMalloc((void **)&iwmap, sizeof(unsigned int *) * inn);
  cudaMalloc((void **)&iomap, sizeof(unsigned int *) * inn);

  kappa = 1.0;
  eta = 1.0;
}

Megatron::~Megatron() {
  cudaFree((void *)out);
  cudaFree((void *)fout);

  cudaFree((void *)owmap);
  cudaFree((void *)oimap);
  cudaFree((void *)iwmap);
  cudaFree((void *)iomap);
}

void Megatron::makemaps(unsigned int minv, unsigned int maxv, double disp) {
  using namespace std;
  vector<vector<unsigned int> > moi, mio;
  vector<vector<unsigned int> > mow, miw;

  moi.resize(outn);
  mow.resize(outn);
  mio.resize(inn);
  miw.resize(inn);

  const double *inx = inl->x;
  const double *iny = inl->y;
  const double *inr = inl->r;
  const double *outx = outl->x;
  const double *outy = outl->y;
  const double *outr = outl->r;

  wn = 0;

  for (unsigned int outi = 0; outi < outn; ++outi) {
    multimap<double, unsigned int> dini;
    for (unsigned int ini = 0; ini < inn; ++ini) {
      double dx = outx[outi] - inx[ini];
      double dy = outy[outi] - iny[ini];
      double d = sqrt(dx * dx + dy * dy);

      if (inr)
        d -= inr[ini];
      if (inr)
        d -= outr[ini];

      dini.insert(make_pair(d, ini));
    }

     auto q = dini.begin();
     unsigned int j = 0;
     while (q != dini.end() && j < maxv && (j < minv || q->first < 0)) {
       unsigned int ini = q->second;
       moi[outi].push_back(ini + 1);
       mio[ini].push_back(outi + 1);

       mow[outi].push_back(wn);
       miw[ini].push_back(wn);

       ++q;
       ++j;

       ++wn;
     }

     moi[outi].push_back(0);
     mow[outi].push_back(wn);
     ++wn;
  }

  cudaMalloc((void **)&weight, sizeof(double) * wn);

  double *cweight = new double[wn];

  for (auto q = mio.begin(); q != mio.end(); ++q)
    q->push_back(0);

  for (unsigned int outi = 0; outi < outn; ++outi) {
    vector<unsigned int>& v = moi[outi];
    vector<unsigned int>& w = mow[outi];

    assert(w.size());

    double iss = disp / sqrt(w.size() + 1);
    double sw = 0;
    for (unsigned int i = 0; i < w.size() - 1; ++i) {
      double ww = iss * rnd(-1, 1);
      cweight[w[i]] = ww;
      sw += ww;
    }
    cweight[w[w.size() - 1]] = 0; //-sw/2.0;

    void *rr;
    cudaMalloc((void **)&rr, sizeof(unsigned int) * v.size());
    cudaMemcpy(rr, v.data(), sizeof(unsigned int) * v.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(oimap + outi, &rr, sizeof(unsigned int *), cudaMemcpyHostToDevice);


    cudaMalloc((void **)&rr, sizeof(unsigned int) * w.size());
    cudaMemcpy(rr, w.data(), sizeof(unsigned int) * w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(owmap + outi, &rr, sizeof(unsigned int *), cudaMemcpyHostToDevice);
  }

  cudaMemcpy(weight, cweight, sizeof(double) * wn, cudaMemcpyHostToDevice);
  delete[] cweight;

  for (unsigned int ini = 0; ini < inn; ++ini) {
    vector<unsigned int>& v = mio[ini];
    vector<unsigned int>& w = miw[ini];

    void *rr;
    cudaMalloc((void **)&rr, sizeof(unsigned int) * v.size());
    cudaMemcpy(rr, v.data(), sizeof(unsigned int) * v.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(iomap + ini, &rr, sizeof(unsigned int *), cudaMemcpyHostToDevice);


    cudaMalloc((void **)&rr, sizeof(unsigned int) * w.size());
    cudaMemcpy(rr, w.data(), sizeof(unsigned int) * w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(iwmap + ini, &rr, sizeof(unsigned int *), cudaMemcpyHostToDevice);
  }
}

#if MEGATEST_MAIN
#include "ppm.hh"
int main() {
  unsigned int ni = 1024;
  unsigned int na = 40;
  unsigned int nc = 1024;

  std::vector<double*> inputs;
  std::vector<double*> attributes;

  fprintf(stderr, "ni=%u na=%u nc=%u\n", ni, na, nc);
  {
    int i = 0;

    fprintf(stderr, "reading samples\n");
    while (!feof(stdin)) {
      double *input = new double[ni];
      int ret = fread(input, sizeof(double), ni, stdin);
      if (ret == 0)
        break;
      assert(ret == ni);
      inputs.push_back(input);

      double *attribute = new double[na];
      ret = fread(attribute, sizeof(double), na, stdin);
      assert(ret == na);
      attributes.push_back(attribute);

      if (i % 1000 == 0)
        fprintf(stderr, "still reading samples i=%d\n", i);
      ++i;
    }
    fprintf(stderr, "finished reading samples i=%d\n", i);

  }

  unsigned int minibatch = 8;



  Layout *inl = Layout::new_square_grid(32);
  Layout *hidl1 = Layout::new_square_random(512);
  Layout *hidl2 = Layout::new_square_random2(512);
  Layout *hidl3 = Layout::new_square_random(512);
  Layout *hidl4 = Layout::new_square_random(512);
  Layout *outl = Layout::new_square_grid(32);
  Megatron *m1 = new Megatron(inl, hidl1); m1->makemaps(10);
  Megatron *m2 = new Megatron(hidl1, hidl2); m2->makemaps(40);
  Megatron *m3 = new Megatron(hidl2, hidl3); m3->makemaps(40);
  Megatron *m4 = new Megatron(hidl3, hidl4); m4->makemaps(20);
  Megatron *m5 = new Megatron(hidl4, outl); m5->makemaps(10);
  m5->kappa = 4.0;

  Tron *m = compositron(m1, m2);
  m = compositron(m, m3);
  m = compositron(m, m4);
  m = compositron(m, m5);

  Tron *cm = compositron(encudatron(1024), m);
  cm = compositron(cm, decudatron(1024));

  int i = 0;

  double cerr2 = 0.5, cerr3 = 0.5;
  while (1) {
    int which = rand() % inputs.size();
 
    double *in = inputs[which];

    const double *out = cm->feed(in);
    cm->target(in);
    cerr2 *= 0.999;
    cerr2 += 0.001 * cm->err2();
    cerr3 *= 0.999;
    cerr3 += 0.001 * cm->err3();
    cm->train(0.1);

    if (i % 1000 == 0) {
      fprintf(stderr, "i=%d in[0]=%lf out[0]=%lf cerr2=%lf cerr3=%lf\n", i, in[0], out[0], cerr2, cerr3);
      PPM ppm1;
      ppm1.unvectorizegray(in, 32, 32);

      PPM ppm2;
      ppm2.unvectorizegray(out, 32, 32);

      PPM ppm3;
      ppm3.w = 32;
      ppm3.h = 64;
      ppm3.data = new uint8_t[ppm3.w * ppm3.h * 3];
      memcpy(ppm3.data, ppm1.data, 32 * 32 * 3);
      memcpy(ppm3.data + 32*32*3, ppm2.data, 32 * 32 * 3);
      ppm3.write(stdout);
    }

   ++i;
  }
}
#endif
