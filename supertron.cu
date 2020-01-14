#define __MAKEMORE_SUPERTRON_CU__ 1

#include <stdio.h>
#include <assert.h>

#include <math.h>

#include <vector>
#include <map>

#include "cudamem.hh"
#include "random.hh"
#include "supertron.hh"
#include "mapfile.hh"

namespace makemore {

inline __device__ int  shl(int a, int b) { return ((b < 0) ? (a >> -b) : (a << b)); }
inline            int dshl(int a, int b) { return ((b < 0) ? (a >> -b) : (a << b)); }

inline __device__ int  shr(int a, int b) { return ((b < 0) ? (a << -b) : (a >> b)); }
inline            int dshr(int a, int b) { return ((b < 0) ? (a << -b) : (a >> b)); }


inline __device__ bool dev_outi_to_ini_wi(const Supertron::Layer::Head* head, int outi, int i, int *inip, int *wip)
#include "outi_to_ini_wi.hh"
inline __device__ bool dev_ini_to_outi_wi(const Supertron::Layer::Head* head, int ini, int i, int *outip, int *wip)
#include "ini_to_outi_wi.hh"
inline __device__ bool dev_wi_to_ini_outi(const Supertron::Layer::Head* head, int wi, int i, int *inip, int *outip)
#include "wi_to_ini_outi.hh"

inline __device__ int dev_get_oiwtabn(const Supertron::Layer::Head* head)
#include "get_oiwtabn.hh"
inline __device__ int dev_get_iowtabn(const Supertron::Layer::Head* head)
#include "get_iowtabn.hh"
inline __device__ int dev_get_wiotabn(const Supertron::Layer::Head* head)
#include "get_wiotabn.hh"


#define shr dshr
#define shl dshl

inline bool outi_to_ini_wi(const Supertron::Layer::Head* head, int outi, int i, int *inip, int *wip)
#include "outi_to_ini_wi.hh"
inline bool ini_to_outi_wi(const Supertron::Layer::Head* head, int ini, int i, int *outip, int *wip)
#include "ini_to_outi_wi.hh"
inline bool wi_to_ini_outi(const Supertron::Layer::Head* head, int wi, int i, int *inip, int *outip)
#include "wi_to_ini_outi.hh"

inline int get_oiwtabn(const Supertron::Layer::Head* head)
#include "get_oiwtabn.hh"
inline int get_iowtabn(const Supertron::Layer::Head* head)
#include "get_iowtabn.hh"
inline int get_wiotabn(const Supertron::Layer::Head* head)
#include "get_wiotabn.hh"





__global__ void gpu_supertron_feed(
  Supertron::Layer layer
) {
  Supertron::Layer::Head *head = layer.head;
  int outn = head->outn;
  int outi = blockIdx.x * blockDim.x + threadIdx.x;
  if (outi >= outn)
    return;

  double *weight = layer.weight;
  const double *in = layer.in;

  double sum = 0;

  if (head->type == Supertron::Layer::TYPE_NOISEPAD || head->type == Supertron::Layer::TYPE_FILLPAD) {
    int tmp = outi;
    int oz = tmp % head->oc; tmp /= head->oc;
    int ox = tmp % head->ow; tmp /= head->ow;
    int oy = tmp;

    int ix = ox;
    int iy = oy;
    int iz = oz;

    if (oz < head->ic) {
      int ini = iz + head->ic * (ix + head->iw * iy);
      sum = in[ini];
    } else if (head->type == Supertron::Layer::TYPE_NOISEPAD) {
      int nni = (oz - head->ic) + (head->oc - head->ic) * (ix + head->iw * iy);
      sum = layer.noise[nni];
    } else if (head->type == Supertron::Layer::TYPE_FILLPAD) {
      int nni = (oz - head->ic);
      sum = layer.noise[nni];
    }
  } else if (head->type == Supertron::Layer::TYPE_IDENTITY) {
    int tmp = outi;
    int oz = tmp % head->oc; tmp /= head->oc;
    int ox = tmp % head->ow; tmp /= head->ow;
    int oy = tmp;

    int ix = ox;
    int iy = oy;
    int iz = oz;

    int ini = iz + head->ic * (ix + head->iw * iy);
    sum = in[ini];
  } else if (head->type == Supertron::Layer::TYPE_MPOOL) {
    int tmp = outi;
    int oz = tmp % head->oc; tmp /= head->oc;
    int ox = tmp % head->ow; tmp /= head->ow;
    int oy = tmp;

    if (head->s >= 0)
      return;
    int md = (1 << -head->s);
    int max_ini = -1;
  
    for (int dy = 0; dy < md; ++ dy) {
      for (int dx = 0; dx < md; ++ dx) {
        int ix = dx + ox * md;
        int iy = dy + oy * md;
        int iz = oz;

        int ini = iz + head->ic * (ix + head->iw * iy);
        if (max_ini < 0 || in[ini] > in[max_ini])
          max_ini = ini;
      }
    }

    sum = in[max_ini];
  } else if (int *oiwtab = layer.oiwtab) {
    unsigned int oiwtabn = dev_get_oiwtabn(head);

    oiwtab += outi * oiwtabn * 2;
    for (int i = 0; i < oiwtabn; ++i, oiwtab += 2) {
      int ini = oiwtab[0];
      int wi = oiwtab[1];
      if (wi >= 0) {
        if (ini >= 0)
          sum += weight[wi] * in[ini];
        else if (ini == -1)
          sum += weight[wi];
      }
    }
  } else {
    int ini, wi;
    for (int i = 0; dev_outi_to_ini_wi(head, outi, i, &ini, &wi); ++i) {
      if (wi >= 0) {
        if (ini >= 0)
          sum += weight[wi] * in[ini];
        else if (ini == -1)
          sum += weight[wi];
      }
    }
  }

  switch (head->activated) {
  case Supertron::Layer::ACTIVATION_NONE:
    layer.out[outi] = sum;
    break;
  case Supertron::Layer::ACTIVATION_SIGMOID:
    layer.out[outi] = 1.0 / (1.0 + exp(-sum));
    break;
  case Supertron::Layer::ACTIVATION_RELU:
    layer.out[outi] = (sum > 0) ? sum : 0;
    break;
  case Supertron::Layer::ACTIVATION_SOFTPLUS:
    layer.out[outi] = (sum > 64) ? sum : log(1 + exp(sum));
    break;
  default:
    layer.out[outi] = 0;
    break;
  }

  layer.fout[outi] = 0.0;
} 

__global__ void gpu_supertron_train0(
  Supertron::Layer layer, double a
) {
  Supertron::Layer::Head *head = layer.head;
  int outn = head->outn;
  int outi = blockIdx.x * blockDim.x + threadIdx.x;
  if (outi >= outn)
    return;

  switch (head->activated) {
  case Supertron::Layer::ACTIVATION_NONE:
    break;
  case Supertron::Layer::ACTIVATION_SIGMOID:
    {
      double o = layer.out[outi];
      double fo = layer.fout[outi];

      if (o > 1.0)
        o = 1.0;
      else if (o < 0.0)
        o = 0.0;

      layer.fout[outi] = fo * o * (1.0 - o);
    }
    break;
  case Supertron::Layer::ACTIVATION_RELU:
    if (layer.out[outi] <= 0)
      layer.fout[outi] = 0;
    break;
  case Supertron::Layer::ACTIVATION_SOFTPLUS:
    layer.fout[outi] = layer.fout[outi] * (1 - exp(-layer.out[outi]));
    break;
  default:
    break;
  }
}




__global__ void gpu_supertron_train1(
  Supertron::Layer layer
) {
  Supertron::Layer::Head *head = layer.head;
  int inn = head->inn;
  unsigned int ini = blockIdx.x * blockDim.x + threadIdx.x;
  if (ini >= inn)
    return;

  double sum = 0;
  int outi, wi;

  double *weight = layer.weight;
  double *fout = layer.fout;

  if (head->type == Supertron::Layer::TYPE_NOISEPAD || head->type == Supertron::Layer::TYPE_IDENTITY || head->type == Supertron::Layer::TYPE_FILLPAD) {
    int tmp = ini;
    int iz = tmp % head->ic; tmp /= head->ic;
    int ix = tmp % head->iw; tmp /= head->iw;
    int iy = tmp;

    int oz = iz;
    int ox = ix;
    int oy = iy;

    int outi = oz + head->oc * (ox + head->ow * oy);
    layer.fin[ini] += layer.fout[outi];
    return;
  } else if (head->type == Supertron::Layer::TYPE_MPOOL) {
    int tmp = ini;
    int iz = tmp % head->ic; tmp /= head->ic;
    int ix = tmp % head->iw; tmp /= head->iw;
    int iy = tmp;

    if (head->s >= 0)
      return;
    int md = (1 << -head->s);

    int oz = iz;
    int ox = ix / md;
    int oy = iy / md;

    int max_inj = -1;
  
    for (int dy = 0; dy < md; ++ dy) {
      for (int dx = 0; dx < md; ++ dx) {
        int jx = dx + ox * md;
        int jy = dy + oy * md;
        int jz = oz;

        int inj = jz + head->ic * (jx + head->iw * jy);
        if (max_inj < 0 || layer.in[inj] > layer.in[max_inj])
          max_inj = inj;
      }
    }

    if (max_inj != ini)
      return;

    int outi = oz + head->oc * (ox + head->ow * oy);
    layer.fin[ini] += layer.fout[outi];
    return;
  }

  if (int *iowtab = layer.iowtab) {
    unsigned int iowtabn = dev_get_iowtabn(head);

    iowtab += ini * iowtabn * 2;
    for (int i = 0; i < iowtabn; ++i, iowtab += 2) {
      outi = iowtab[0];
      wi = iowtab[1];

      if (outi >= 0 && wi >= 0)
        sum += weight[wi] * fout[outi];
    }
  } else {
    for (int i = 0; dev_ini_to_outi_wi(head, ini, i, &outi, &wi); ++i) {
      if (outi >= 0 && wi >= 0)
        sum += weight[wi] * fout[outi];
    }
  }

  layer.fin[ini] += sum;
}


__global__ void gpu_supertron_train2(
  Supertron::Layer layer, double a
) {
  Supertron::Layer::Head *head = layer.head;
  int wn = head->wn;

  if (head->type == Supertron::Layer::TYPE_MPOOL)
    return;
  if (head->type == Supertron::Layer::TYPE_IDENTITY)
    return;
  if (head->type == Supertron::Layer::TYPE_NOISEPAD)
    return;
  if (head->type == Supertron::Layer::TYPE_FILLPAD)
    return;
  if (a == 0)
    return;

  unsigned int wi;
  int k;
  int i0, i1;
  unsigned int n = dev_get_wiotabn(head);

  if (layer.wbufk) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    wi = q % wn;
    k = q / wn;
    if (k >= layer.wbufk)
      return;

    i0 = (k * n) / layer.wbufk;
    i1 = ((k + 1) * n) / layer.wbufk;
  } else {
    wi = blockIdx.x * blockDim.x + threadIdx.x;
    if (wi >= wn)
      return;
    k = -1;
    i0 = 0;
    i1 = n;
  }

  double *fout = layer.fout;
  const double *in = layer.in;

  double dw = 0;
  int nw = 0;


  if (int *wiotab = layer.wiotab) {
    unsigned int wiotabn = dev_get_wiotabn(head);

    wiotab += wi * wiotabn * 2 + i0 * 2;

    for (int i = i0; i < i1; ++i, wiotab += 2) {
      int ini = wiotab[0];
      int outi = wiotab[1];

      if (outi >= 0 && ini != -2) {
        dw += fout[outi] * (ini == -1 ? 1.0 : in[ini]);
        ++nw;
      }
    }

  } else {
    int ini, outi;

    for (int i = i0; i < i1; ++i) {
      dev_wi_to_ini_outi(head, wi, i, &ini, &outi);
      if (outi >= 0 && ini != -2) {
        dw += fout[outi] * (ini == -1 ? 1.0 : in[ini]);
        ++nw;
      }
    }
  }

  if (layer.wbufk) {
    layer.wbuf[wi * layer.wbufk + k] = dw;

    __syncthreads();

    if (k)
      return;

    double sdw = 0;
    for (k = 0; k < layer.wbufk; ++k)
      sdw += layer.wbuf[wi * layer.wbufk + k];
    dw = sdw;
  }

  // if (nw > 1)
  //   dw /= (double)nw;

  if (head->adam) {
    double adam_b1 = layer.head->adam_b1;
    double adam_b2 = layer.head->adam_b2;
    double adam_b3 = layer.head->adam_b3;
    double adam_eps = layer.head->adam_eps;


    layer.m[wi] = adam_b1 * layer.m[wi] + (1.0 - adam_b1) * dw;
    layer.v[wi] = adam_b2 * layer.v[wi] + (1.0 - adam_b2) * dw * dw;

    double dx = layer.m[wi] / (sqrt(layer.v[wi]) + adam_eps);
    // gradient clamp sigma
    double ddq = 1e6;
    if (dx < -ddq) {
      dx = -ddq;
    } else if (dx > ddq) {
      dx = ddq;
    }

    layer.weight[wi] += a * dx;
  } else {
    layer.weight[wi] += a * dw;
  }
}

const double *Supertron::feed(const double *_in, double *_fin) {
  assert(layers.size() > 0);

  for (unsigned int li = 0; li < layers.size(); ++li) {
    Supertron::Layer &lay = *layers[li];

    if (li > 0) {
      lay.in = layers[li - 1]->out;
      lay.fin = layers[li - 1]->fout;
    } else {
      lay.in = _in;
      lay.fin = _fin;
    }

    if (lay.type == Supertron::Layer::TYPE_NOISEPAD || lay.type == Supertron::Layer::TYPE_FILLPAD) {
      unsigned int nn;
      if (lay.type == Supertron::Layer::TYPE_NOISEPAD)
        nn = lay.outn - lay.inn;
      else if (lay.type == Supertron::Layer::TYPE_FILLPAD)
        nn = 256; // !!!
        
      if (!lay.noise)
        cumake(&lay.noise, nn);
      double *noisebuf = new double[nn];
      for (unsigned int j = 0; j < nn; ++j)
        noisebuf[j] = randgauss();
      encude(noisebuf, nn, lay.noise);
      delete[] noisebuf;
    }

    int bs = 256;
    int gs = (lay.outn + bs - 1) / bs;
    gpu_supertron_feed<<<gs, bs>>>(lay);
  }
  
  return layers[layers.size() - 1]->out;
}

void Supertron::target(const double *tgt) {
  assert(layers.size());
  Supertron::Layer *lay = layers[layers.size() - 1];
  cusubvec(tgt, lay->out, lay->outn, lay->fout);
}


void Supertron::update_stats() {
  assert(layers.size());
  Supertron::Layer *lay = layers[layers.size() - 1];

  double z = pow(1.0 - errdecay, (double)rounds);

  double nerr2 = sqrt(cusumsq(lay->fout, lay->outn) / (double)lay->outn);
//fprintf(stderr, "nerr2=%lf\n", nerr2);
  err2 *= (1.0 - z);
  err2 *= (1.0 - errdecay);
  err2 += errdecay * nerr2;
  err2 *= 1.0 / (1.0 - z * (1.0 - errdecay));

  double nerrm = cumaxabs(lay->fout, lay->outn);
  errm *= (1.0 - z);
  errm *= (1.0 - errdecay);
  errm += errdecay * nerrm;
  errm *= 1.0 / (1.0 - z * (1.0 - errdecay));

  ++rounds;
}

void Supertron::train(double nu) {

  for (auto li = layers.rbegin(); li != layers.rend(); ++li) {
    Supertron::Layer &lay = **li;

    {
      int bs0 = 256;
      int gs0 = (lay.outn + bs0 - 1) / bs0;
      gpu_supertron_train0<<<gs0, bs0>>>(lay, nu);
    }

    if (lay.fin) {
      int bs1 = 256;
      int gs1 = (lay.inn + bs1 - 1) / bs1;
      gpu_supertron_train1<<<gs1, bs1>>>(lay);
    }


    if (lay.wbufk) {
      int bs2 = 256;
      int gs2 = (lay.wn * lay.wbufk + bs2 - 1) / bs2;
      gpu_supertron_train2<<<gs2, bs2>>>(lay, nu);
    } else {
      int bs2 = 256;
      int gs2 = (lay.wn + bs2 - 1) / bs2;
      gpu_supertron_train2<<<gs2, bs2>>>(lay, nu);
    }
  }
}

Supertron::Supertron(Mapfile *_mapfile) {
  inn = 0;
  outn = 0;
  err2 = 0;
  errm = 0;
  rounds = 0;
  errdecay = 0.01;

  mapfile = _mapfile;

  cumake(&cunlayers, 1);
  mapfile->map(cunlayers, 1);
  mapfile->load(cunlayers);

  uint64_t nlayers;
  decude(cunlayers, 1, &nlayers);
  layers.resize(nlayers);

  for (unsigned int i = 0; i < nlayers; ++i) {
    Supertron::Layer *lay = new Supertron::Layer;
    cumake(&lay->head, 1);
    mapfile->map(lay->head, 1);
    mapfile->load(lay->head);

    Supertron::Layer::Head head;
    decude(lay->head, 1, &head);
    lay->type = head.type;
    lay->wn = head.wn;
    lay->inn = head.inn;
    lay->outn = head.outn;

    cumake(&lay->out, head.outn);
    cumake(&lay->fout, head.outn);

    cumake(&lay->weight, head.wn);
    mapfile->map(lay->weight, head.wn);
    mapfile->load(lay->weight);

    cumake(&lay->m, head.wn);
    mapfile->map(lay->m, head.wn);
    mapfile->load(lay->m);

    cumake(&lay->v, head.wn);
    mapfile->map(lay->v, head.wn);
    mapfile->load(lay->v);

    lay->iowtab = NULL;
    lay->oiwtab = NULL;
    lay->wiotab = NULL;

    if (head.type == Supertron::Layer::TYPE_CONV) {

      unsigned int mwn = 65536;
      if (lay->wn < mwn) {
        lay->wbufk = 2 * mwn / lay->wn;
        if (lay->wbufk > 1024)
          lay->wbufk = 1024;
      }

      cumake(&lay->wbuf, lay->wn * lay->wbufk);
    }

#if 0
    {
      unsigned int oiwtabn = get_oiwtabn(&head);
      cumake(&lay->oiwtab, oiwtabn * head.outn * 2);
      mapfile->map(lay->oiwtab, oiwtabn * head.outn * 2);
      mapfile->load(lay->oiwtab);
    }

    {
      unsigned int iowtabn = get_iowtabn(&head);
      cumake(&lay->iowtab, iowtabn * head.inn * 2);
      mapfile->map(lay->iowtab, iowtabn * head.inn * 2);
      mapfile->load(lay->iowtab);
    }

    {
      unsigned int wiotabn = get_wiotabn(&head);
      cumake(&lay->wiotab, wiotabn * head.wn * 2);
      mapfile->map(lay->wiotab, wiotabn * head.wn * 2);
      mapfile->load(lay->wiotab);
    }
#endif


    layers[i] = lay;
    if (i > 0) {
fprintf(stderr, "i=%d\n", i);
      assert(layers[i - 1]->outn == layers[i]->inn);
}
  }

  if (nlayers) {
    inn = layers[0]->inn;
    outn = layers[layers.size() - 1]->outn;
  }
}

Supertron::~Supertron() {
  for (auto lay : layers) {
    cufree(lay->head);
    cufree(lay->out);
    cufree(lay->fout);
    cufree(lay->weight);
    cufree(lay->m);
    cufree(lay->v);
    delete lay;
  }
  if (cunlayers)
    cufree(cunlayers);
}

void Supertron::add_layer(
   Supertron::Layer::Type type,
   unsigned int iw, unsigned int ih, unsigned int ic,
   unsigned int ow, unsigned int oh, unsigned int oc,
   unsigned int d, int s,
   Supertron::Layer::Activation activated
) {
  Supertron::Layer::Head head;

  head.type = type;
  head.iw = iw;
  head.ih = ih;
  head.ic = ic;
  head.ow = ow;
  head.oh = oh;
  head.oc = oc;
  head.d = d;
  head.s = s;
  head.activated = activated;

  head.inn = iw * ih * ic;
  head.outn = ow * oh * oc;

  head.adam = true;
  head.adam_b1 = 0.9;
  head.adam_b2 = 0.999;
//  head.adam_b2 = 0.99;
  head.adam_b3 = 0.5;
  head.adam_eps = 1e-8;

  int d21 = (d * 2 + 1);
  int m = (d21 * d21 * ic + 1);
  int s2 = (s > 0) ? (1 << s) : 1;

  switch (head.type) {
  case Supertron::Layer::TYPE_IDENTITY:
    head.wn = 1;
    assert(iw == ow);
    assert(ih == oh);
    assert(ic == oc);
    assert(d == 0);
    assert(s == 0);
    break;

  case Supertron::Layer::TYPE_NOISEPAD:
  case Supertron::Layer::TYPE_FILLPAD:
    head.wn = 1;
    assert(d == 0);
    assert(s == 0);
    assert(iw == ow);
    assert(ih == oh);
    assert(oc > ic);
    break;

  case Supertron::Layer::TYPE_FULL:
    head.wn = (head.inn + 1) * head.outn;
    assert(d == 0);
    assert(s == 0);
    break;

  case Supertron::Layer::TYPE_LOCAL:
    head.wn = head.outn * m;
fprintf(stderr, "%d, %d, %d, %d\n", ow, iw, s, (iw>>-s));
    assert(ow == shl(iw, s));
    assert(oh == shl(ih, s));
    break;

  case Supertron::Layer::TYPE_CONV:
    head.wn = s2 * s2 * head.oc * m;
    assert(ow == shl(iw, s));
    assert(oh == shl(ih, s));
    break;

  case Supertron::Layer::TYPE_YCONV:
    head.wn = s2 * oc * ow * m;
//fprintf(stderr, "s=%d iw=%u ow=%u\n", s, iw, ow);
    assert(ow == shl(iw, s));
    assert(oh == shl(ih, s));
    break;

  case Supertron::Layer::TYPE_MPOOL:
    head.wn = 1;
    assert(s < 0);
    assert(d == 0);
    assert(ow == shl(iw, s));
    assert(oh == shl(ih, s));
    break;

  default:
    assert(0);
  }

fprintf(stderr, "wn=%d\n", head.wn);

  add_layer(head);
}

void Supertron::add_layer(const Supertron::Layer::Head &head) {
  uint64_t nlayers = layers.size();
  if (nlayers) {
    Layer *ll = layers[nlayers - 1];
    assert(ll->outn == head.inn);
  }

  ++nlayers;
  encude(&nlayers, 1, cunlayers);

  Supertron::Layer *lay = new Supertron::Layer;
  cumake(&lay->head, 1);
  mapfile->map(lay->head, 1);
  encude(&head, 1, lay->head);

  cumake(&lay->out, head.outn);
  cumake(&lay->fout, head.outn);

  cumake(&lay->weight, head.wn);
  mapfile->map(lay->weight, head.wn);
  cumake(&lay->m, head.wn);
  mapfile->map(lay->m, head.wn);
  cumake(&lay->v, head.wn);
  mapfile->map(lay->v, head.wn);

  lay->type = head.type;

  lay->wn = head.wn;
  lay->inn = head.inn;
  lay->outn = head.outn;

  lay->iowtab = NULL;
  lay->oiwtab = NULL;
  lay->wiotab = NULL;

#if 0
  {
    unsigned int oiwtabn = get_oiwtabn(&head);
    cumake(&lay->oiwtab, oiwtabn * head.outn * 2);
    mapfile->map(lay->oiwtab, oiwtabn * head.outn * 2);

    int *oiwtab = new int[oiwtabn * head.outn * 2], *p = oiwtab;
    for (unsigned int outi = 0; outi < head.outn; ++outi) {
      for (int i = 0; i < oiwtabn; ++i, p += 2) {
        assert(outi_to_ini_wi(&head, outi, i, p + 0, p + 1));
      }

      int tmp0, tmp1;
      assert(!outi_to_ini_wi(&head, outi, oiwtabn, &tmp0, &tmp1));
    }
    encude(oiwtab, oiwtabn * head.outn * 2, lay->oiwtab);
    delete[] oiwtab;
  }

  {
    unsigned int iowtabn = get_iowtabn(&head);
    cumake(&lay->iowtab, iowtabn * head.inn * 2);
    mapfile->map(lay->iowtab, iowtabn * head.inn * 2);

    int *iowtab = new int[iowtabn * head.inn * 2], *p = iowtab;
    for (unsigned int ini = 0; ini < head.inn; ++ini) {
      for (int i = 0; i < iowtabn; ++i, p += 2) {
        assert(ini_to_outi_wi(&head, ini, i, p + 0, p + 1));
      }

      int tmp0, tmp1;
      assert(!ini_to_outi_wi(&head, ini, iowtabn, &tmp0, &tmp1));
    }
    encude(iowtab, iowtabn * head.inn * 2, lay->iowtab);
    delete[] iowtab;
  }

  {
    unsigned int wiotabn = get_wiotabn(&head);
    cumake(&lay->wiotab, wiotabn * head.wn * 2);
    mapfile->map(lay->wiotab, wiotabn * head.wn * 2);

    int *wiotab = new int[wiotabn * head.wn * 2], *p = wiotab;
    for (unsigned int wi = 0; wi < head.wn; ++wi) {
      for (int i = 0; i < wiotabn; ++i, p += 2) {
        assert(wi_to_ini_outi(&head, wi, i, p + 0, p + 1));
      }

      int tmp0, tmp1;
      assert(!wi_to_ini_outi(&head, wi, wiotabn, &tmp0, &tmp1));
    }
    encude(wiotab, wiotabn * head.wn * 2, lay->wiotab);
    delete[] wiotab;
  }
#endif

  layers.push_back(lay);

  if (layers.size() == 1)
    inn = lay->inn;
  outn = lay->outn;
}

void Supertron::randomize(double disp) {
  for (auto lay : layers) {
    Supertron::Layer::Head head;
    decude(lay->head, 1, &head);

    unsigned int s = get_oiwtabn(&head);
    double f = disp / sqrt((double)s);
    double *weight = new double[lay->wn];

    switch (head.type) {
    case Supertron::Layer::TYPE_FULL:
    case Supertron::Layer::TYPE_LOCAL:
      for (unsigned int outi = 0; outi < head.outn; ++outi) {

        double sw = 0;
        int ini, wi;
        for (int i = 0; i < s; ++i) {
          outi_to_ini_wi(&head, outi, i, &ini, &wi);
          if (ini != -1) {
            weight[wi] = randgauss() * f;
            sw += weight[wi];
          }
        }
        for (int i = 0; i < s; ++i) {
          outi_to_ini_wi(&head, outi, i, &ini, &wi);
          if (ini == -1 && wi != -1) {
            weight[wi] = -sw/2.0;
            break;
          }
        }
      }
      break;

    default:
      for (unsigned int wi = 0; wi < lay->wn; ++wi)
        weight[wi] = randgauss() * f;
      break;
    }

    encude(weight, lay->wn, lay->weight);
    delete[] weight;
  }
}

}

#if MAIN
using namespace makemore;

int main() {
  Mapfile mapfile("foos.map");
  Supertron s(&mapfile);

fprintf(stderr, "%lu\n", s.layers.size());
for (auto lay : s.layers) {
  Supertron::Layer::Head head;
  decude(lay->head, 1, &head);
  fprintf(stderr, "%u %u %u\n", head.inn, head.outn, head.wn);
}

#if 1
double *cuin;
cumake(&cuin, 256);
double *in = new double[256];
double *out = new double[256];

while (1) {
  for (int j = 0;j < 256; ++j) {
    in[j] = randrange(0, 1);
  }

  encude(in, 256, cuin);
  decude(s.feed(cuin), 256, out);

  s.target(cuin);
  s.update_stats();
  s.train(0.0001);

  if (s.rounds % 500 == 0) {
    for (unsigned int k = 0; k < 5; ++k)
      fprintf(stderr, "%lf\t", out[k]);
    fprintf(stderr, "\n");
    fprintf(stderr, "err2=%lf\n", s.err2);

    s.mapfile->save();
  }
}

#else
  s.add_layer(Supertron::Layer::TYPE_YCONV,
    16, 16, 1,
    32, 32, 1,
    1, 1, true
  );

  s.add_layer(Supertron::Layer::TYPE_YCONV,
    32, 32, 1,
    16, 16, 1,
    1, -1, false
  );
    
  s.randomize(1.0);
  s.mapfile->save();
#endif
  return 0;
}
#endif
