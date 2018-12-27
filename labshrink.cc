#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

int main(int argc, char **argv) {
  assert(argc >= 3);
  int w = atoi(argv[1]);
  int h = atoi(argv[2]);
  int n = 1;
  if (argc >= 4)
    n = atoi(argv[3]);
  unsigned int nw, nh;

  double *data = new double[w * h * 3];
  int ret = fread(data, sizeof(double), w * h * 3, stdin);
  assert(ret == w * h * 3);

  for (unsigned int i = 0; i < n; ++i) {
    assert(w % 2 == 0);
    assert(h % 2 == 0);
  
    nw = w / 2;
    nh = h / 2;
  
    assert(nw > 0);
    assert(nh > 0);
    assert(data);
  
    double *ndata = new double[nw * nh * 3];
  
    for (unsigned int y = 0; y < nh; ++y) {
      for (unsigned int x = 0; x < nw; ++x) {
        for (unsigned int c = 0; c < 3; ++c) {
  
          double z = 0;
          z += data[y * w * 6 + x * 6 + c + 0];
          z += data[y * w * 6 + x * 6 + c + 3];
          z += data[y * w * 6 + x * 6 + c + w * 3 + 0];
          z += data[y * w * 6 + x * 6 + c + w * 3 + 3];
          z /= 4.0;
  
          ndata[y * nw * 3 + x * 3 + c] = z;
        }
      }
    }

    delete[] data;
    data = ndata;
    w = nw;
    h = nh;
  }
  
  ret = fwrite(data, sizeof(double), w * h * 3, stdout);
  assert(ret == w * h * 3);
  return 0;
}
