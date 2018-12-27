#include <stdio.h>

#include <math.h>

#include <vector>
#include <map>

#include "dataset.hh"
#include "megatron.hh"
#include "ppm.hh"

int main() {
  Dataset samples("face-32x32-gray-full.dat");


  Layout *inl = Layout::new_square_grid(32);
  Layout *hidl1 = Layout::new_square_random(1024);
  Layout *hidl2 = Layout::new_square_random(512);
  Layout *hidl3 = Layout::new_square_random(1024);
//  Layout *hidl4 = Layout::new_square_random(512);
  Layout *outl = Layout::new_square_grid(32);

  Wiring *w1 = new Wiring(inl, hidl1, 8, 8);
  Wiring *w2 = new Wiring(hidl1, hidl2, 8, 8);
  Wiring *w3 = new Wiring(hidl2, hidl3, 8, 8);
  Wiring *w4 = new Wiring(hidl3, outl, 8, 8);
//  Wiring *w5 = new Wiring(hidl4, outl, 10);

  Megatron *m1 = new Megatron(w1);
  Megatron *m2 = new Megatron(w2);
  Megatron *m3 = new Megatron(w3);
  Megatron *m4 = new Megatron(w4);
//  Megatron *m5 = new Megatron(w5);
//  m5->kappa = 4.0;
  m4->kappa = 4.0;

  Tron *m = compositron(m1, m2);
  m = compositron(m, m3);
  m = compositron(m, m4);
//  m = compositron(m, m5);

  Tron *cm = compositron(encudatron(1024), m);
  cm = compositron(cm, decudatron(1024));

  int i = 0;

  double cerr2 = 0.5, cerr3 = 0.5;
  while (1) {
    int which = samples.pick();

    const double *in = samples.data(which);

    const double *out = cm->feed(in);
    cm->target(in);
    cerr2 *= 0.999;
    cerr2 += 0.001 * cm->err2();
    cerr3 *= 0.999;
    cerr3 += 0.001 * cm->err3();
    cm->train(0.03);

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
