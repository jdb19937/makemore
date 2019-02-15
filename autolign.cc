#include "project.hh"
#include "topology.hh"
#include "random.hh"
#include "cudamem.hh"
#include "project.hh"
#include "pipeline.hh"

#include "ppm.hh"
#include "warp.hh"

#include <math.h>

#include <map>

using namespace makemore;

Pipeline *open_pipeline(unsigned int mbn) {
  Pipeline *pipe = new Pipeline(mbn);
  pipe->add_stage(new Project("test8.proj", mbn));
  pipe->add_stage(new Project("test16.proj", mbn));
  pipe->add_stage(new Project("test32.proj", mbn));
  pipe->add_stage(new Project("test64.proj", mbn));
  return pipe;
}


int usage() {
  fprintf(stderr,
    "Usage: errstats\n"
  );
  return 1;
}

void focus(double *a, const double *xp, const double *yp, unsigned int n) {
  for (unsigned int i = 0; i < n; ++i) {
    if (i >= n)
      return;

    double x = xp[i];
    double y = yp[i];

    double d2 = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5);
    double f = 1.0 - 2.0 * d2;
    if (f < 0.0)
      f = 0.0;
    if (f > 1.0)
      f = 1.0;


// In[2]:= f[x_,y_] := 1 - 2 * ((x-1/2)^2 + (y-1/2)^2) 
// In[6]:= Integrate[f[x,y], {x,0,1},{y,0,1}]
//
//         2
// Out[6]= -
//         3
//  a[i] *= f * 1.5;


// In[26]:= Integrate[f[x,y]^2, {x,0,1},{y,0,1}]
// 
//          22
// Out[26]= --
//          45

    a[i] *= f * f * (45.0 / 22.0);
  }
}

void subvec(double *outbuf, double *tgtbuf) {
  for (unsigned int j = 0; j < 64 * 64 * 3; ++j)
    outbuf[j] -= tgtbuf[j];
}

double err2(double *buf) {
  double e2 = 0;
  for (unsigned int j = 0; j < 64 * 64 * 3; ++j)
    e2 += buf[j] * buf[j];
  e2 /= (64.0 * 64.0 * 3.0);
  return e2;
}

int main(int argc, char **argv) {
  seedrand();

  Pipeline *pipe = open_pipeline(1);
  double srcbuf[320*400*3];
  double tgtbuf[320*400*3];
  double errbuf[64*64*3];
  double best[64*64*3];

  int ret = fread(srcbuf, sizeof(double), 320 * 400 * 3, stdin);
  assert(ret == 320 * 400 * 3);

  assert(pipe->ctxlay->n == 72);
  for (unsigned int i = 0; i < pipe->ctxlay->n; ++i)
    pipe->ctxbuf[i] = 0.5;

  pipe->ctxbuf[68] = 1.0;
  pipe->ctxbuf[69] = 0;
  pipe->ctxbuf[70] = 0;
  pipe->ctxbuf[71] = 0;

  pipe->ctrlock = 0;
  pipe->tgtlock = -1;

  double beste2 = -1;

  double guessx0 = -30;
  double guessy0 = 0;

  PPM ppm(192, 64, 0);

  jwarp(srcbuf, 320, 400, 0, 0, 320, 0, 64, 64, tgtbuf);
  ppm.pastelab(tgtbuf, 64, 64, 0, 0);

  for (unsigned int i = 0; i < 1024; ++i) { 
    double x0 = guessx0 + randrange(-30, 30);
    double y0 = guessy0 + randrange(-30, 30);
    double x1 = (320 - guessx0) + randrange(-5, 5);
    double y1 = y0;

    double ddy = randrange(-20.0, 20.0);
    y1 += ddy;
    y0 -= ddy;
    
    jwarp(srcbuf, 320, 400, (int)x0, (int)y0, (int)x1, (int)y1, 64, 64, tgtbuf);

    memcpy(pipe->outbuf, tgtbuf, sizeof(double) * 64 * 64 * 3);
    pipe->reencode();
    pipe->generate();

    memcpy(errbuf, pipe->outbuf, sizeof(double)*64*64*3);
    subvec(errbuf, tgtbuf);
assert(pipe->outlay->n == 64*64*3);
    focus(errbuf, pipe->outlay->x, pipe->outlay->y, pipe->outlay->n);
    double e2 = err2(errbuf);

//    for (int k = 1; k < 4; ++k) {
//      pipe->stages[k]->gen->reset_stats();
//      encude(pipe->stages[k]->tgtbuf, pipe->stages[k]->tgtlay->n, cubuf);
//      pipe->stages[k]->gen->target(cubuf);
//      e2 += pipe->stages[k]->gen->err2;
//    }

    if (beste2 < 0 || e2 < beste2) {
      beste2 = e2;
      ppm.pastelab(tgtbuf, 64, 64, 64, 0);
      ppm.pastelab(pipe->outbuf, 64, 64, 128, 0);
    }

fprintf(stderr, "%lf\tx0=%lf y0=%lf x1=%lf y1=%lf\n", e2, x0, y0, x1, y1);
  }

  ppm.write(stdout);
  return 0;
}
