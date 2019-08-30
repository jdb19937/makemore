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

static inline __device__ int shl(int a, int b) {
  if (b >= 0) {
    return (a << b);
  } else {
    return (a >> (-b));
  }
}
static inline __device__ int shr(int a, int b) {
  if (b >= 0) {
    return (a >> b);
  } else {
    return (a << (-b));
  }
}

static inline __device__ bool outi_to_ini_wi(const Supertron::Layer::Head* head, int outi, int i, int *inip, int *wip) {
  switch (head->type) {
  case Supertron::Layer::TYPE_FULL:
    if (i < 0 || i >= head->inn) {
      *inip = -1;
      *wip = -1;
      return false;
    }

    *inip = i;
    *wip = *inip * head->outn + outi;
    return true;

  case Supertron::Layer::TYPE_LOCAL:
    {
      int d21 = (head->d * 2 + 1);
      int m = (d21 * d21 * head->ic + 1);
      if (i >= m) {
        *inip = -1;
        *wip = -1;
        return false;
      }

      *wip = outi + head->outn * i;

      if (i >= m - 1) {
        *inip = -1;
        return true;
      }

      int dx0 = (int)(i / head->ic) % d21;
      int dy0 = (int)(i / head->ic) / d21;
      int dx = dx0 - head->d;
      int dy = dy0 - head->d;

      // int oz = (outi % head->oc);
      int ox = (int)(outi / head->oc) % head->ow;
      int oy = (int)(outi / head->oc) / head->ow;

      int iz = i % head->ic;
      int ix = shr(ox, head->s) - dx;
      int iy = shr(oy, head->s) - dy;
  
      if (ix < 0 || ix >= head->iw || iy < 0 || iy >= head->ih) {
        *inip = -1;
        return true;
      }

      *inip = iz + head->ic * (ix + head->iw * iy);
      return true;
    }

  case Supertron::Layer::TYPE_CONV:
    {
      int d21 = (head->d * 2 + 1);
      int m = (d21 * d21 * head->ic + 1);
      if (i >= m) {
        *inip = -1;
        *wip = -1;
        return false;
      }

      // int oz = (outi % head->oc);
      int ox = (int)(outi / head->oc) % head->ow;
      int oy = (int)(outi / head->oc) / head->ow;

      int s2 = (head->s > 0) ? (1 << head->s) : 1;
      int rxy = s2 * (oy % s2) + (ox % s2);
      *wip = rxy + s2 * s2 * i;

      if (i >= m - 1) {
        *inip = -1;
        return true;
      }

      int dx0 = (int)(i / head->ic) % d21;
      int dy0 = (int)(i / head->ic) / d21;
      int dx = dx0 - head->d;
      int dy = dy0 - head->d;

      int iz = i % head->ic;
      int ix = shr(ox, head->s) - dx;
      int iy = shr(oy, head->s) - dy;
  
      if (ix < 0 || ix >= head->iw || iy < 0 || iy >= head->ih) {
        *inip = -1;
        return true;
      }

      *inip = iz + head->ic * (ix + head->iw * iy);
      return true;
    }

  case Supertron::Layer::TYPE_YCONV:
    {
      int d21 = (head->d * 2 + 1);
      int m = (d21 * d21 * head->ic + 1);
      if (i >= m) {
        *inip = -1;
        *wip = -1;
        return false;
      }

      int oz = (outi % head->oc);
      int ox = (int)(outi / head->oc) % head->ow;
      int oy = (int)(outi / head->oc) / head->ow;

      int s2 = (head->s > 0) ? (1 << head->s) : 1;
      int ry = (oy % s2);

      *wip = ry + s2 * (oz + head->oc * (ox + head->ow * (i)));

      if (i >= m - 1) {
        *inip = -1;
        return true;
      }

      int dx0 = (int)(i / head->ic) % d21;
      int dy0 = (int)(i / head->ic) / d21;
      int dx = dx0 - head->d;
      int dy = dy0 - head->d;

      int iz = i % head->ic;
      int ix = shr(ox, head->s) - dx;
      int iy = shr(oy, head->s) - dy;
  
      if (ix < 0 || ix >= head->iw || iy < 0 || iy >= head->ih) {
        *inip = -1;
        return true;
      }

      *inip = iz + head->ic * (ix + head->iw * iy);
      return true;
    }

  default:
    *inip = -1;
    *wip = -1;
    return false;
  }

  return false;
}

__device__ bool ini_to_outi_wi(const Supertron::Layer::Head* head, int ini, int i, int *outip, int *wip) {
  switch (head->type) {
  case Supertron::Layer::TYPE_FULL:
    if (i < 0 || i >= head->outn) {
      *outip = -1;
      *wip = -1;
      return false;
    }

    *outip = i;
    *wip = ini * head->outn + *outip;
    return true;

  case Supertron::Layer::TYPE_LOCAL:
    {
      int iz = (ini % head->ic);
      int ix = (int)(ini / head->ic) % head->iw;
      int iy = (int)(ini / head->ic) / head->iw;

      int d21 = (head->d * 2 + 1);
      int s2 = (head->s > 0) ? (1 << head->s) : 1;
      int m = (d21 * d21 * head->oc);
      if (i >= m * s2 * s2) {
        *outip = -1;
        *wip = -1;
        return false;
      }

      int rxy = i / m;
      int rx = rxy % s2;
      int ry = rxy / s2;
      i %= m;

      int dx0 = (int)(i / head->oc) % d21;
      int dy0 = (int)(i / head->oc) / d21;
      int dx = dx0 - head->d;
      int dy = dy0 - head->d;

      int oz = i % head->oc;
      int ox = shl((ix + dx), head->s) + rx;
      int oy = shl((iy + dy), head->s) + ry;

      if (ox < 0 || ox >= head->ow || oy < 0 || oy >= head->oh) {
        *outip = -1;
        *wip = -1;
        return true;
      }
      
      *outip = oz + head->oc * (ox + head->ow * oy);
      *wip = *outip + head->outn * (iz + head->ic * (dx0 + d21 * (dy0)));
      return true;
    }

  case Supertron::Layer::TYPE_CONV:
    {
      int iz = (ini % head->ic);
      int ix = (int)(ini / head->ic) % head->iw;
      int iy = (int)(ini / head->ic) / head->iw;

      int d21 = (head->d * 2 + 1);
      int s2 = (head->s > 0) ? (1 << head->s) : 1;
      int m = (d21 * d21 * head->oc);
      if (i >= m * s2 * s2) {
        *outip = -1;
        *wip = -1;
        return false;
      }

      int rxy = i / m;
      int rx = rxy % s2;
      int ry = rxy / s2;
      i %= m;

      int dx0 = (int)(i / head->oc) % d21;
      int dy0 = (int)(i / head->oc) / d21;
      int dx = dx0 - head->d;
      int dy = dy0 - head->d;

      int oz = i % head->oc;
      int ox = shl((ix + dx), head->s) + rx;
      int oy = shl((iy + dy), head->s) + ry;

      if (ox < 0 || ox >= head->ow || oy < 0 || oy >= head->oh) {
        *outip = -1;
        *wip = -1;
        return true;
      }

      *outip = oz + head->oc * (ox + head->ow * oy);
      *wip = rxy + s2 * s2 * (iz + head->ic * (dx0 + d21 * (dy0)));
      return true;
    }

  case Supertron::Layer::TYPE_YCONV:
    {
      int iz = (ini % head->ic);
      int ix = (int)(ini / head->ic) % head->iw;
      int iy = (int)(ini / head->ic) / head->iw;

      int d21 = (head->d * 2 + 1);
      int s2 = (head->s > 0) ? (1 << head->s) : 1;
      int m = (d21 * d21 * head->oc);
      if (i >= m * s2 * s2) {
        *outip = -1;
        *wip = -1;
        return false;
      }

      int rxy = i / m;
      int rx = rxy % s2;
      int ry = rxy / s2;
      i %= m;

      int dx0 = (int)(i / head->oc) % d21;
      int dy0 = (int)(i / head->oc) / d21;
      int dx = dx0 - head->d;
      int dy = dy0 - head->d;

      int oz = i % head->oc;
      int ox = shl((ix + dx), head->s) + rx;
      int oy = shl((iy + dy), head->s) + ry;

      if (ox < 0 || ox >= head->ow || oy < 0 || oy >= head->oh) {
        *outip = -1;
        *wip = -1;
        return true;
      }

      *outip = oz + head->oc * (ox + head->ow * oy);

      *wip = ry + s2 *
        (oz + (head->oc * (ox + head->ow *
          (iz + head->ic * (dx0 + d21 * (dy0)))
        )));

      return true;
    }

  default:
    *outip = -1;
    *wip = -1;
    return false;
  }

  //return false;
}

__device__ bool wi_to_ini_outi(const Supertron::Layer::Head* head, int wi, int i, int *inip, int *outip) {
  switch (head->type) {
  case Supertron::Layer::TYPE_FULL:
    if (i == 0) {
      *inip = wi / head->outn;
      if (*inip >= head->inn)
        *inip = -1;
      *outip = wi % head->outn;
    } else {
      *inip = -1;
      *outip = -1;
    }
    break;

  case Supertron::Layer::TYPE_LOCAL:
    {
      if (i != 0) {
        *inip = -1;
        *outip = -1;
        break;
      }

      int outi = wi % head->outn;
      // int oz = (outi % head->oc);
      int ox = (int)(outi / head->oc) % head->ow;
      int oy = (int)(outi / head->oc) / head->ow;
      *outip = outi;

      int d21 = (head->d * 2 + 1);
      int m = (d21 * d21 * head->ic + 1);
      int di = wi / head->outn;
      if (di >= m - 1) {
        *inip = -1;
        break;
      }

      int dx0 = (int)(di / head->ic) % d21;
      int dy0 = (int)(di / head->ic) / d21;
      int dx = dx0 - head->d;
      int dy = dy0 - head->d;

      int iz = di % head->ic;
      int ix = shr(ox, head->s) - dx;
      int iy = shr(oy, head->s) - dy;
  
      if (ix < 0 || ix >= head->iw || iy < 0 || iy >= head->ih) {
        *inip = -1;
        break;
      }

      *inip = iz + head->ic * (ix + head->iw * iy);
      break;
    }
  case Supertron::Layer::TYPE_CONV:
    {
      int s2 = (head->s > 0) ? (1 << head->s) : 1;
      int n = head->outn / s2 / s2;

      if (i < 0 || i >= n) {
        *inip = -1;
        *outip = -1;
        break;
      }

      int d21 = (head->d * 2 + 1);
      int m = (d21 * d21 * head->ic + 1);

      int tmp = wi;
      int rxy = tmp % (s2 * s2); tmp /= (s2 * s2);
      int ry = rxy / s2;
      int rx = rxy % s2;
      int di = tmp;

      int oz = i % head->oc;
      int ox = rx + s2 * ((i / head->oc) % (int)(head->ow / s2));
      int oy = ry + s2 * ((i / head->oc) / (int)(head->oh / s2));

      int outi = oz + head->oc * (ox + head->ow * oy);
      *outip = outi;

      if (di >= m - 1) {
        *inip = -1;
        break;
      }

      int dx0 = (int)(di / head->ic) % d21;
      int dy0 = (int)(di / head->ic) / d21;
      int dx = dx0 - head->d;
      int dy = dy0 - head->d;

      int iz = di % head->ic;
      int ix = shr(ox, head->s) - dx;
      int iy = shr(oy, head->s) - dy;
  
      if (ix < 0 || ix >= head->iw || iy < 0 || iy >= head->ih) {
        *inip = -1;
        break;
      }

      *inip = iz + head->ic * (ix + head->iw * iy);
      break;
    }
  case Supertron::Layer::TYPE_YCONV:
    {
      int s2 = (head->s > 0) ? (1 << head->s) : 1;
      int n = head->oh / s2;

      if (i < 0 || i >= n) {
        *inip = -1;
        *outip = -1;
        break;
      }

      int d21 = (head->d * 2 + 1);
      int m = (d21 * d21 * head->ic + 1);

      int tmp = wi;
      int ry = tmp % s2; tmp /= s2;
      int oz = tmp % head->oc; tmp /= head->oc;
      int ox = tmp % head->ow; tmp /= head->ow;
      int oy = ry + s2 * i;
      int di = tmp;

      int outi = oz + head->oc * (ox + head->ow * oy);
      *outip = outi;

      if (di >= m - 1) {
        *inip = -1;
        break;
      }

      int dx0 = (int)(di / head->ic) % d21;
      int dy0 = (int)(di / head->ic) / d21;
      int dx = dx0 - head->d;
      int dy = dy0 - head->d;

      int iz = di % head->ic;
      int ix = shr(ox, head->s) - dx;
      int iy = shr(oy, head->s) - dy;
  
      if (ix < 0 || ix >= head->iw || iy < 0 || iy >= head->ih) {
        *inip = -1;
        break;
      }

      *inip = iz + head->ic * (ix + head->iw * iy);
      break;
    }
  default:
    *inip = -1;
    *outip = -1;
    break;
  }

  return (*outip != -1 || *inip != -1);
}

__global__ void gpu_supertron_feed(
  Supertron::Layer layer
) {
  Supertron::Layer::Head *head = layer.head;
  int outn = head->outn;
  int outi = blockIdx.x * blockDim.x + threadIdx.x;
  if (outi >= outn)
    return;

  int ini, wi;
  double sum = 0;

  for (int i = 0; outi_to_ini_wi(head, outi, i, &ini, &wi); ++i) {
    if (wi != -1)
      sum += layer.weight[wi] * (ini == -1 ? 1.0 : layer.in[ini]);
  }

  if (head->activated) {
    double q = 1.0 / (1.0 + exp(-sum));
    layer.out[outi] = q;
  } else {
    layer.out[outi] = sum;
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
  if (!head->activated)
    return;

  double o = layer.out[outi];
  double fo = layer.fout[outi];

  if (o > 1.0)
    o = 1.0;
  else if (o < 0.0)
    o = 0.0;

  layer.fout[outi] = fo * o * (1.0 - o);
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

  for (int i = 0; ini_to_outi_wi(head, ini, i, &outi, &wi); ++i) {
    if (outi != -1)
      sum += layer.weight[wi] * layer.fout[outi];
  }

  layer.fin[ini] += sum;
}


__global__ void gpu_supertron_train2(
  Supertron::Layer layer, double a
) {
  Supertron::Layer::Head *head = layer.head;
  int wn = head->wn;
  unsigned int wi = blockIdx.x * blockDim.x + threadIdx.x;
  if (wi >= wn)
    return;

  double dw = 0;
  int ini, outi;

  for (int i = 0; wi_to_ini_outi(head, wi, i, &ini, &outi); ++i) {
    if (outi != -1)
      dw += layer.fout[outi] * (ini == -1 ? 1.0 : layer.in[ini]);
  }

  layer.m[wi] = layer.head->adam_b1 * layer.m[wi] + (1.0 - layer.head->adam_b1) * dw;
  layer.v[wi] = layer.head->adam_b2 * layer.v[wi] + (1.0 - layer.head->adam_b2) * dw * dw;
  layer.weight[wi] += a * layer.m[wi] / (pow(layer.v[wi], layer.head->adam_b3) + layer.head->adam_eps);
}

const double *Supertron::feed(const double *_in, double *_fin) {
  assert(layers.size() > 0);

  for (unsigned int li = 0; li < layers.size(); ++li) {
    Layer &lay = *layers[li];

    if (li > 0) {
      lay.in = layers[li - 1]->out;
      lay.fin = layers[li - 1]->fout;
    } else {
      lay.in = _in;
      lay.fin = _fin;
    }

    int bs = 256;
    int gs = (lay.outn + bs - 1) / bs;
    gpu_supertron_feed<<<gs, bs>>>(lay);
  }
  
  return layers[layers.size() - 1]->out;
}

void Supertron::target(const double *tgt) {
  assert(layers.size());
  Layer *lay = layers[layers.size() - 1];
  cusubvec(tgt, lay->out, lay->outn, lay->fout);
}


void Supertron::update_stats() {
  assert(layers.size());
  Layer *lay = layers[layers.size() - 1];

  double z = pow(1.0 - errdecay, (double)rounds);

  double nerr2 = sqrt(cusumsq(lay->fout, lay->outn) / (double)lay->outn);
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
    Layer &lay = **li;

    int bs0 = 256;
    int gs0 = (lay.outn + bs0 - 1) / bs0;
    gpu_supertron_train0<<<gs0, bs0>>>(lay, nu);

    if (lay.fin) {
      int bs1 = 256;
      int gs1 = (lay.inn + bs1 - 1) / bs1;
      gpu_supertron_train1<<<gs1, bs1>>>(lay);
    }

    int bs2 = 256;
    int gs2 = (lay.wn + bs2 - 1) / bs2;
    gpu_supertron_train2<<<gs2, bs2>>>(lay, nu);
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
    Layer *lay = new Layer;
    cumake(&lay->head, 1);
    mapfile->map(lay->head, 1);
    mapfile->load(lay->head);

    Layer::Head head;
    decude(lay->head, 1, &head);
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

    layers[i] = lay;
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
   Layer::Type type,
   unsigned int iw, unsigned int ih, unsigned int ic,
   unsigned int ow, unsigned int oh, unsigned int oc,
   unsigned int d, int s,
   bool activated
) {
  Layer::Head head;

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

  head.adam_b1 = 0.9;
  head.adam_b2 = 0.999;
  head.adam_b3 = 0.5;
  head.adam_eps = 1e-8;

  int d21 = (d * 2 + 1);
  int m = (d21 * d21 * ic + 1);
  int s2 = (s > 0) ? (1 << s) : 1;

  switch (head.type) {
  case Layer::TYPE_FULL:
    head.wn = head.inn * head.outn;
    assert(d == 0);
    assert(s == 0);
    break;

  case Layer::TYPE_LOCAL:
    head.wn = head.outn * m;
    if (s < 0) {
      assert((ow << -s) == iw);
      assert((oh << -s) == ih);
    } else {
      assert(ow == (iw << s));
      assert(oh == (ih << s));
    }
    break;

  case Layer::TYPE_CONV:
    head.wn = s2 * s2 * m;
    if (s < 0) {
      assert((ow << -s) == iw);
      assert((oh << -s) == ih);
    } else {
      assert(ow == (iw << s));
      assert(oh == (ih << s));
    }
    break;

  case Layer::TYPE_YCONV:
    head.wn = s2 * oc * ow * m;
//fprintf(stderr, "s=%d iw=%u ow=%u\n", s, iw, ow);
    if (s < 0) {
      assert((ow << -s) == iw);
      assert((oh << -s) == ih);
    } else {
      assert(ow == (iw << s));
      assert(oh == (ih << s));
    }
    break;

  default:
    assert(0);
  }

  add_layer(head);
}

void Supertron::add_layer(const Supertron::Layer::Head &head) {
  uint64_t nlayers = layers.size();
  ++nlayers;
  encude(&nlayers, 1, cunlayers);

  Layer *lay = new Layer;
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

  lay->wn = head.wn;
  lay->inn = head.inn;
  lay->outn = head.outn;

  layers.push_back(lay);

  if (layers.size() == 1)
    inn = lay->inn;
  outn = lay->outn;
}

void Supertron::randomize(double disp) {
  for (auto lay : layers) {
    double *weight = new double[lay->wn];
    for (unsigned int wi = 0; wi < lay->wn; ++wi)
      weight[wi] = randgauss() * disp;
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
