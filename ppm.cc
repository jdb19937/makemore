#include <stdio.h>
#include <ctype.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <string>

#include <math.h>

#include "imgutils.hh"

#if 1
#include <jpeglib.h>
#endif

#include "ppm.hh"

namespace makemore {

double PPM::centerlight() {
  unsigned int x0 = w / 4;
  unsigned int x1 = w * 3 / 4;
  unsigned int y0 = h / 4;
  unsigned int y1 = h * 3 / 4;

  double s = 0;
  unsigned int tot = 0;

  for (unsigned int y = y0; y < y1; ++y) {
    for (unsigned int x = x0; x < x1; ++x) {
      s += data[y * w * 3 + x * 3 + 0];
      s += data[y * w * 3 + x * 3 + 1];
      s += data[y * w * 3 + x * 3 + 2];
      ++tot;
    }
  }
  assert(tot > 0);
  s /= (double)tot;
  return s;
}

bool PPM::read_jpeg(const std::string &jpeg) {
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;

  cinfo.err = jpeg_std_error(&jerr);	
  jpeg_create_decompress(&cinfo);

  jpeg_mem_src(&cinfo, (uint8_t *)jpeg.data(), jpeg.length());
  int rc = jpeg_read_header(&cinfo, TRUE);
  if (rc != 1) {
    jpeg_abort_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return false;
  }

  jpeg_start_decompress(&cinfo);
	
  w = cinfo.output_width;
  h = cinfo.output_height;
  if (cinfo.output_components != 3) {
    jpeg_abort_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return false;
  }

  if (data)
    delete[] data;
  data = new uint8_t[w * h * 3];
  unsigned long w3 = w * 3;

  while (cinfo.output_scanline < cinfo.output_height) {
    unsigned char *buffer_array[1];
    buffer_array[0] = data + cinfo.output_scanline * w3;
    jpeg_read_scanlines(&cinfo, buffer_array, 1);
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);

  return true;
}


void PPM::write_jpeg(std::string *jpeg) {
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;

  JSAMPROW row_pointer[1];
  int row_stride;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  uint8_t *mem = NULL;
  unsigned long mem_size;
  jpeg_mem_dest(&cinfo, &mem, &mem_size);

  cinfo.image_width = w;
  cinfo.image_height = h;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, 90, TRUE);

  jpeg_start_compress(&cinfo, TRUE);

  row_stride = w * 3;
  while (cinfo.next_scanline < cinfo.image_height) {
    row_pointer[0] = & data[cinfo.next_scanline * row_stride];
    (void)jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);

  jpeg->resize(mem_size);
  memcpy((uint8_t *)jpeg->data(), mem, mem_size);
  ::free(mem);
}

void PPM::write_jpeg(FILE *outfile) {
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;

  JSAMPROW row_pointer[1];
  int row_stride;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  jpeg_stdio_dest(&cinfo, outfile);

  cinfo.image_width = w;
  cinfo.image_height = h;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, 90, TRUE);

  jpeg_start_compress(&cinfo, TRUE);

  row_stride = w * 3;
  while (cinfo.next_scanline < cinfo.image_height) {
    row_pointer[0] = & data[cinfo.next_scanline * row_stride];
    (void)jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);
}

bool PPM::read(FILE *fp) {
  int ret = getc(fp);
  if (ret == EOF)
    return 0;
  assert(ret == 'P');
  assert(getc(fp) == '6');
  assert(isspace(getc(fp)));

  int c;
  char buf[256], *p = buf;
  do { *p = getc(fp); } while (isspace(*p));
  ++p;

  while (!isspace(*p = getc(fp))) {
    assert(isdigit(*p));
    assert(p - buf < 32);
    ++p;
  }
  *p = 0;
  w = atoi(buf);

  p = buf;
  while (!isspace(*p = getc(fp))) {
    assert(isdigit(*p));
    assert(p - buf < 32);
    ++p;
  }
  *p = 0;
  h = atoi(buf);

  p = buf;
  while (!isspace(*p = getc(fp))) {
    assert(isdigit(*p));
    assert(p - buf < 32);
    ++p;
  }
  *p = 0;
  assert(!strcmp(buf, "255"));

  if (data)
    delete[] data;
  data = new uint8_t[3 * w * h];

  assert(3 * w * h == fread(data, 1, 3 * w * h, fp));
  return 1;
}

void PPM::write(FILE *fp) {
  fprintf(fp, "P6\n%u %u\n255\n", w, h);
  assert(3 * w * h == fwrite(data, 1, 3 * w * h, fp));
  fflush(fp);
}


static double squish(uint8_t x, double d = 0.02) {
  return (d + ((double)x/255.999) * (1 - 2.0 * d));
}

static uint8_t unsquish(double x, double d = 0.02) {
  double y = ((x - d) / (1 - 2.0 * d));
  if (y > 1.0) y = 1.0;
  if (y < 0.0) y = 0.0;
  return (uint8_t)(y * 255.999);
}

void PPM::vectorize(std::vector<double> *vecp) {
  unsigned int n = w * h * 3;
  vecp->resize(n);
  for (unsigned int i = 0; i < n; i += 3) {
    double l, a, b;
    rgbtolab(data[i+0], data[i+1], data[i+2], &l, &a, &b);
    (*vecp)[i+0] = l;
    (*vecp)[i+1] = a;
    (*vecp)[i+2] = b;
  }
}

void PPM::vectorize(std::vector<uint8_t> *vecp) {
  unsigned int n = w * h * 3;
  vecp->resize(n);
  vectorize(vecp->data());
}

void PPM::vectorize(uint8_t *vecp) {
  unsigned int n = w * h * 3;
  for (unsigned int i = 0; i < n; i += 3) {
    double l, a, b;
    rgbtolab(data[i+0], data[i+1], data[i+2], &vecp[i+0], &vecp[i+1], &vecp[i+2]);
  }
}

void PPM::unvectorizegray(const double *vec, unsigned int _w, unsigned int _h) {
  unsigned int n = _w * _h * 3;

  w = _w;
  h = _h;
  if (data)
    delete[] data;
  data = new uint8_t[n];

  for (unsigned int i = 0; i < n; i += 3) {
    uint8_t r, g, b;
    labtorgb(vec[i/3], 0.5, 0.5, &r, &g, &b);
    data[i+0] = r;
    data[i+1] = g;
    data[i+2] = b;
  }
}

void PPM::border(unsigned int x0, unsigned int y0, unsigned int x1, unsigned int y1, uint8_t cr, uint8_t cg, uint8_t cb) {
  if (x1 >= w)
    x1 = w - 1;
  if (x0 >= x1)
    x0 = x1;
  if (y1 >= h)
    y1 = h - 1;
  if (y0 >= y1)
    y0 = y1;

  for (unsigned int x = x0; x <= x1; ++x) {
    data[y0 * w * 3 + x * 3 + 0] = cr;
    data[y0 * w * 3 + x * 3 + 0] = cg;
    data[y0 * w * 3 + x * 3 + 0] = cb;
  }

  for (unsigned int y = y0 + 1; y <= y1 - 1; ++y) {
    data[y * w * 3 + x0 * 3 + 0] = cr;
    data[y * w * 3 + x0 * 3 + 0] = cg;
    data[y * w * 3 + x0 * 3 + 0] = cb;

    data[y * w * 3 + x1 * 3 + 0] = cr;
    data[y * w * 3 + x1 * 3 + 0] = cg;
    data[y * w * 3 + x1 * 3 + 0] = cb;
  }

  for (unsigned int x = x0; x <= x1; ++x) {
    data[y1 * w * 3 + x * 3 + 0] = cr;
    data[y1 * w * 3 + x * 3 + 0] = cg;
    data[y1 * w * 3 + x * 3 + 0] = cb;
  }
}

void PPM::make(unsigned int _w, unsigned int _h, uint8_t v) {
  unsigned int n = _w * _h * 3;
  w = _w;
  h = _h;

  if (data)
    delete[] data;
  data = new uint8_t[n];

  for (unsigned int i = 0; i < n; ++i)
    data[i] = v;
}

void PPM::pastealpha(const uint8_t *vec, const uint8_t *a, unsigned int vw, unsigned int vh, unsigned int x0, unsigned int y0) {
  // assert(y0 + vh <= h);
  // assert(x0 + vw <= w);

  unsigned int i = 0;
  for (unsigned int vy = 0; vy < vh; ++vy) {
    unsigned int y = y0 + vy;
    unsigned int yw3 = y * w * 3;

    for (unsigned int vx = 0; vx < vw; ++vx) {
      unsigned int x = x0 + vx;
      unsigned int x3 = x * 3;

      if (x < w && y < h) {
        double ax = (double)a[i / 3] / 255.0;

//ax = (ax - 0.5) / 0.01;
if (ax > 1) ax = 1;
if (ax < 0) ax = 0;

        data[yw3 + x3 + 0] = data[yw3 + x3 + 0] * (1 - ax) + ax * vec[i + 0];
        data[yw3 + x3 + 1] = data[yw3 + x3 + 1] * (1 - ax) + ax * vec[i + 1];
        data[yw3 + x3 + 2] = data[yw3 + x3 + 2] * (1 - ax) + ax * vec[i + 2];
      }

      i += 3;
    }
  }
}

void PPM::pastelab(const double *vec, unsigned int vw, unsigned int vh, unsigned int x0, unsigned int y0) {
  assert(y0 + vh <= h);
  assert(x0 + vw <= w);

  unsigned int i = 0;
  for (unsigned int vy = 0; vy < vh; ++vy) {
    unsigned int y = y0 + vy;
    unsigned int yw3 = y * w * 3;

    for (unsigned int vx = 0; vx < vw; ++vx) {
      unsigned int x = x0 + vx;
      unsigned int x3 = x * 3;

      labtorgb(
        vec[i+0], vec[i+1], vec[i+2], 
        data+yw3+x3+0, data+yw3+x3+1, data+yw3+x3+2
      );

      i += 3;
    }
  }
}

void PPM::pastelab(const uint8_t *vec, unsigned int vw, unsigned int vh, unsigned int x0, unsigned int y0) {
  assert(y0 + vh <= h);
  assert(x0 + vw <= w);

  unsigned int i = 0;
  for (unsigned int vy = 0; vy < vh; ++vy) {
    unsigned int y = y0 + vy;
    unsigned int yw3 = y * w * 3;

    for (unsigned int vx = 0; vx < vw; ++vx) {
      unsigned int x = x0 + vx;
      unsigned int x3 = x * 3;

      labtorgb(
       (double)(vec[i+0] * 100.0) / 255.0, (double)(vec[i+1] - 128.0), (double)(vec[i+2] - 128.0),
        data+yw3+x3+0, data+yw3+x3+1, data+yw3+x3+2
      );

      i += 3;
    }
  }
}

void PPM::paste(const uint8_t *vec, unsigned int vw, unsigned int vh, unsigned int x0, unsigned int y0) {
//  assert(y0 + vh <= h);
//  assert(x0 + vw <= w);

  unsigned int i = 0;
  for (unsigned int vy = 0; vy < vh; ++vy) {
    unsigned int y = y0 + vy;
    unsigned int yw3 = y * w * 3;

    for (unsigned int vx = 0; vx < vw; ++vx) {
      unsigned int x = x0 + vx;
      unsigned int x3 = x * 3;

if (x < w && y < h) {
      data[yw3 + x3 + 0] = vec[i + 0];
      data[yw3 + x3 + 1] = vec[i + 1];
      data[yw3 + x3 + 2] = vec[i + 2];
}

      i += 3;
    }
  }
}

void PPM::cutlab(double *vec, unsigned int vw, unsigned int vh, unsigned int x0, unsigned int y0) {
  assert(y0 + vh <= h);
  assert(x0 + vw <= w);

  unsigned int i = 0;
  for (unsigned int vy = 0; vy < vh; ++vy) {
    unsigned int y = y0 + vy;
    unsigned int yw3 = y * w * 3;

    for (unsigned int vx = 0; vx < vw; ++vx) {
      unsigned int x = x0 + vx;
      unsigned int x3 = x * 3;

      rgbtolab(
        data[yw3+x3+0], data[yw3+x3+1], data[yw3+x3+2],
        vec+i+0, vec+i+1, vec+i+2
      );

      i += 3;
    }
  }
}

void PPM::unvectorize(const uint8_t *vec, unsigned int _w, unsigned int _h) {
  unsigned int n = _w * _h * 3;

  w = _w;
  h = _h;
  if (data)
    delete[] data;
  data = new uint8_t[n];

  for (unsigned int i = 0; i < n; i += 3) {
    uint8_t r, g, b;
    labtorgb((double)(vec[i+0] * 100.0) / 255.0, (double)(vec[i+1] - 128.0), (double)(vec[i+2] - 128.0), &r, &g, &b);
    data[i+0] = r;
    data[i+1] = g;
    data[i+2] = b;
  }
}
void PPM::unvectorize(const double *vec, unsigned int _w, unsigned int _h) {
  unsigned int n = _w * _h * 3;

  w = _w;
  h = _h;
  if (data)
    delete[] data;
  data = new uint8_t[n];

  for (unsigned int i = 0; i < n; i += 3) {
    uint8_t r, g, b;
    labtorgb(vec[i+0], vec[i+1], vec[i+2], &r, &g, &b);
    data[i+0] = r;
    data[i+1] = g;
    data[i+2] = b;
  }
}

void PPM::unvectorize(const std::vector<double> &vec, unsigned int _w, unsigned int _h) {
  unsigned int n = _w * _h * 3;
  assert(n == vec.size());

  w = _w;
  h = _h;
  if (data)
    delete[] data;
  data = new uint8_t[n];

  for (unsigned int i = 0; i < n; i += 3) {
    uint8_t r, g, b;
    labtorgb(vec[i+0], vec[i+1], vec[i+2], &r, &g, &b);
    data[i+0] = r;
    data[i+1] = g;
    data[i+2] = b;
  }
}

void PPM::pad() {
  unsigned int nw = w + 2;
  unsigned int nh = h + 2;
  uint8_t *ndata = new uint8_t[nw * nh * 3];

  for (unsigned int c = 0; c < 3; ++c) {

  for (unsigned int y = 0; y < nh; ++y) {
    for (unsigned int x = 0; x < nw; ++x) {
      unsigned int zx = x;
      if (zx > 0) zx--;
      if (zx >= w) zx--;

      unsigned int zy = y;
      if (zy > 0) zy--;
      if (zy >= h) zy--;

      ndata[y*3*nw + x*3 + c] = data[zy*3*w + zx*3 + c];
#if 0
      continue;

      if (x > 0 && x < nw - 1 && y > 0 && y < nh - 1) {
        ndata[y*3*nw + x*3 + c] = data[(y-1)*3*w + (x-1)*3 + c];
        continue;
      }
       
      ndata[y*3*nw + x*3 + c] = 0;
#endif
    }
  }

  }

  delete[] data;
  data = ndata;
  w = nw;
  h = nh;
}
        

void PPM::zoom() {
  unsigned int nw = w * 2;
  unsigned int nh = h * 2;

  assert(nw > 0);
  assert(nh > 0);
  assert(data);

  double *vec = new double[nw * nh * 3]();

  for (unsigned int y = 0; y < h; ++y) {
    for (unsigned int x = 0; x < w; ++x) {
      for (unsigned int c = 0; c < 3; ++c) {
        uint8_t z = data[y * w * 3 + x * 3 + c];

        vec[(2*y+0) * nw * 3 + (2*x+0) * 3 + c] = z;

        vec[(2*y+1) * nw * 3 + (2*x+0) * 3 + c] += z/2.0;
        if (y > 0)
          vec[(2*y-1) * nw * 3 + (2*x+0) * 3 + c] += z/2.0;

        vec[(2*y+0) * nw * 3 + (2*x+1) * 3 + c] += z/2.0;
        if (x > 0)
          vec[(2*y+0) * nw * 3 + (2*x-1) * 3 + c] += z/2.0;

        vec[(2*y+1) * nw * 3 + (2*x+1) * 3 + c] += z/4.0;
        if (x > 0)
          vec[(2*y+1) * nw * 3 + (2*x-1) * 3 + c] += z/4.0;
        if (y > 0)
          vec[(2*y-1) * nw * 3 + (2*x+1) * 3 + c] += z/4.0;
        if (x > 0 && y > 0)
          vec[(2*y-1) * nw * 3 + (2*x-1) * 3 + c] += z/4.0;

        if (x == w-1) {
          vec[(2*y+0) * nw * 3 + (2*x+1) * 3 + c] += z/2.0;
          vec[(2*y+1) * nw * 3 + (2*x+1) * 3 + c] += z/4.0;
          if (y > 0)
            vec[(2*y-1) * nw * 3 + (2*x+1) * 3 + c] += z/4.0;
        }

        if (y == h-1) {
          vec[(2*y+1) * nw * 3 + (2*x+0) * 3 + c] += z/2.0;
          vec[(2*y+1) * nw * 3 + (2*x+1) * 3 + c] += z/4.0;
          if (x > 0)
            vec[(2*y+1) * nw * 3 + (2*x-1) * 3 + c] += z/4.0;
        }

        if (x == w-1 && y == h-1) {
          vec[(2*y+1) * nw * 3 + (2*x+1) * 3 + c] += z/4.0;
        }
      }
    }
  }

  uint8_t *ndata = new uint8_t[nw * nh * 3];
  for (int i = nw * nh * 3; i >= 0; --i)
    ndata[i] = vec[i];
  delete[] vec;

  delete[] data;
  data = ndata;
  w = nw;
  h = nh;
}

void PPM::rawzoom() {
  unsigned int nw = w * 2;
  unsigned int nh = h * 2;

  assert(nw > 0);
  assert(nh > 0);
  assert(data);

  double *vec = new double[nw * nh * 3]();

  for (unsigned int y = 0; y < h; ++y) {
    for (unsigned int x = 0; x < w; ++x) {
      for (unsigned int c = 0; c < 3; ++c) {
        double z = data[y * w * 3 + x * 3 + c];

        vec[(2*y+0) * nw * 3 + (2*x+0) * 3 + c] = z;
        vec[(2*y+1) * nw * 3 + (2*x+0) * 3 + c] = z;
        vec[(2*y+0) * nw * 3 + (2*x+1) * 3 + c] = z;
        vec[(2*y+1) * nw * 3 + (2*x+1) * 3 + c] = z;
      }
    }
  }

  uint8_t *ndata = new uint8_t[nw * nh * 3];
  for (int i = nw * nh * 3; i >= 0; --i)
    ndata[i] = vec[i];
  delete[] vec;

  delete[] data;
  data = ndata;
  w = nw;
  h = nh;
}

void PPM::shrink() {
  unsigned int nw = w / 2;
  unsigned int nh = h / 2;

  assert(nw > 0);
  assert(nh > 0);
  assert(data);

  uint8_t *ndata = new uint8_t[nw * nh * 3];

  for (unsigned int y = 0; y < nh; ++y) {
    for (unsigned int x = 0; x < nw; ++x) {
      for (unsigned int c = 0; c < 3; ++c) {

        double z = 0;
        z += data[y * w * 6 + x * 6 + c + 0];
        z += data[y * w * 6 + x * 6 + c + 3];
        z += data[y * w * 6 + x * 6 + c + w * 3 + 0];
        z += data[y * w * 6 + x * 6 + c + w * 3 + 3];
        z /= 4.0;

        ndata[y * nw * 3 + x * 3 + c] = round(z);
      }
    }
  }

  delete[] data;
  data = ndata;
  w = nw;
  h = nh;
}

static inline double pd(double x, double y, double w, double h) {
  double dx = (x - w/2.0) / (w/2.0);
  double dy = (y - h/2.0) / (h/2.0);
  double dx2 = dx * dx;
  double dy2 = dy * dy;
  double d = sqrt(dx2 + dy2);
  d -= 0.5;
  d *= 1.8;
  if (d < 0)
    d = 0;
  if (d > 1)
    d = 1;
  return d;
}

void PPM::blurge() {
  assert(data);

  assert(h % 2 == 0);
  assert(w % 2 == 0);
  uint8_t *ndata = new uint8_t[w * h * 3];

  for (unsigned int y = 0; y < h; y += 2) {
    for (unsigned int x = 0; x < w; x += 2) {
      for (unsigned int c = 0; c < 3; ++c) {
        double z = 0;
        z += data[y * w * 3 + x * 3 + c + 0];
        z += data[y * w * 3 + x * 3 + c + 3];
        z += data[y * w * 3 + x * 3 + c + w * 3 + 0];
        z += data[y * w * 3 + x * 3 + c + w * 3 + 3];
        z /= 4.0;

        double q, d;
        q = data[y * w * 3 + x * 3 + c + 0];
        d = pd(x, y, w, h);
        ndata[y * w * 3 + x * 3 + c + 0] = (1-d)*q + d*z;

        q = data[y * w * 3 + x * 3 + c + 3];
        d = pd(x+1, y, w, h);
        ndata[y * w * 3 + x * 3 + c + 3] = (1-d)*q + d*z;

        q = data[y * w * 3 + x * 3 + c + w * 3 + 0];
        d = pd(x, y+1, w, h);
        ndata[y * w * 3 + x * 3 + c + w * 3 + 0] = (1-d)*q + d*z;

        q = data[y * w * 3 + x * 3 + c + w * 3 + 3];
        d = pd(x+1, y+1, w, h);
        ndata[y * w * 3 + x * 3 + c + w * 3 + 3] = (1-d)*q + d*z;
      }
    }
  }

  delete[] data;
  data = ndata;
}







#ifdef PPMBLURGE_MAIN
int main() {
  PPM p;
  p.read(stdin);
  p.blurge();
  p.write(stdout);
  return 0;
}
#endif


} //namespace





#if PPMTEST_MAIN
int main() {
  PPM p;
  p.read(stdin);
  for (unsigned int y = 0; y < p.h; ++y) {
    for (unsigned int x = 0; x < p.w; ++x) {
      uint8_t r = p.data[y * p.w * 3 + x * 3 + 0];
      uint8_t g =   p.data[y * p.w * 3 + x * 3 + 1];
       uint8_t b =  p.data[y * p.w * 3 + x * 3 + 2];
      double l, a, B;
      rgbtolab(r,g, b,
        &l, &a, &B
      );
      uint8_t r2, g2, b2;
      labtorgb(l, a, B, &r2, &g2, &b2);
      printf("%lf\t%lf\t%lf (%d %d %d)\n", l, a, B, r-r2, g-g2, b-b2);
    }
  }
  return 0;
}
#endif


#if PPMSHRINK_MAIN
int main() {
  PPM p;
  p.read(stdin);
  p.shrink();
  p.write(stdout);
  return 0;
}
#endif

#if PPMZOOM_MAIN
int main() {
  PPM p;
  p.read(stdin);
  p.zoom();
  p.write(stdout);
  return 0;
}
#endif


#if PPMPAD_MAIN
int main(int argc, char **argv) {
  PPM p;
  unsigned int n = 1;
  if (argc > 1)
    n = atoi(argv[1]);

  p.read(stdin);
  for (unsigned int i = 0; i < n; ++i)
    p.pad();
  p.write(stdout);
  return 0;
}
#endif


#if PPMTOLAB_MAIN
int main(int argc, char **argv) {
  PPM p;
  p.read(stdin);
  std::vector<double> v;
  p.vectorize(&v);
  assert(v.size() == p.w * p.h * 3);
  fwrite(v.data(), sizeof(double), v.size(), stdout);
  return 0;
}
#endif

#if PPMTOL_MAIN
int main(int argc, char **argv) {
  PPM p;
  p.read(stdin);
  std::vector<double> v;
  p.vectorize(&v);
  assert(v.size() == p.w * p.h * 3);

  std::vector<double> l;
  l.resize(p.w * p.h);
  for (unsigned int i = 0; i < v.size(); i += 3)
    l[i/3] = v[i];

  fwrite(l.data(), sizeof(double), l.size(), stdout);
  return 0;
}
#endif



#if PPMTOLABTWID_MAIN

typedef std::vector<double> Vec;
void twiddle(const Vec &z, unsigned int w, unsigned int h, Vec *lo, Vec *hi) {
  assert(w % 2 == 0 && h % 2 == 0);
  assert(z.size() == 3 * w * h);

  unsigned int nw = w/2;
  unsigned int nh = h/2;
  lo->resize(nw * nh * 3);
  hi->resize(nw * nh * 9);

  unsigned int ilo = 0, ihi = 0;

  unsigned int w3 = w * 3;
  for (unsigned int y = 0; y < h; y += 2) {
    for (unsigned int x = 0; x < w; x += 2) {
      for (unsigned int c = 0; c < 3; ++c) {
        unsigned int p = y * w3 + x * 3 + c;

        double m = (z[p] + z[p+3] + z[p+w3] + z[p+w3+3]) / 4.0;
        double l = (z[p] + z[p+w3]) / 2.0 - m;
        double t = (z[p] + z[p+3]) / 2.0 - m;
        double s = (z[p] + z[p+w3+3]) / 2.0 - m;

        (*lo)[ilo++] = m;
        (*hi)[ihi++] = 0.5 + l/2.0;
        (*hi)[ihi++] = 0.5 + t/2.0;
        (*hi)[ihi++] = 0.5 + s/2.0;
      }
    }
  }
}

int main(int argc, char **argv) {
  PPM p;
  p.read(stdin);
  std::vector<double> v;
  p.vectorize(&v);
  assert(v.size() == p.w * p.h * 3);
  std::vector<double> lo, hi;
  twiddle(v, p.w, p.h, &lo, &hi);
  fwrite(hi.data(), sizeof(double), hi.size(), stdout);
  fwrite(lo.data(), sizeof(double), lo.size(), stdout);
  return 0;
}
#endif

#if PPMTOLABHI_MAIN
typedef std::vector<double> Vec;
void twiddle(const Vec &z, unsigned int w, unsigned int h, Vec *lo, Vec *hi) {
  assert(w % 2 == 0 && h % 2 == 0);
  assert(z.size() == 3 * w * h);

  unsigned int nw = w/2;
  unsigned int nh = h/2;
  lo->resize(nw * nh * 3);
  hi->resize(nw * nh * 9);

  unsigned int ilo = 0, ihi = 0;

  unsigned int w3 = w * 3;
  for (unsigned int y = 0; y < h; y += 2) {
    for (unsigned int x = 0; x < w; x += 2) {
      for (unsigned int c = 0; c < 3; ++c) {
        unsigned int p = y * w3 + x * 3 + c;

        double m = (z[p] + z[p+3] + z[p+w3] + z[p+w3+3]) / 4.0;
        double l = (z[p] + z[p+w3]) / 2.0 - m;
        double t = (z[p] + z[p+3]) / 2.0 - m;
        double s = (z[p] + z[p+w3+3]) / 2.0 - m;

        (*lo)[ilo++] = m;
        (*hi)[ihi++] = 0.5 + l/2.0;
        (*hi)[ihi++] = 0.5 + t/2.0;
        (*hi)[ihi++] = 0.5 + s/2.0;
      }
    }
  }
}

int main(int argc, char **argv) {
  PPM p;
  p.read(stdin);
  std::vector<double> v;
  p.vectorize(&v);
  assert(v.size() == p.w * p.h * 3);
  std::vector<double> lo, hi;
  twiddle(v, p.w, p.h, &lo, &hi);
  fwrite(hi.data(), sizeof(double), hi.size(), stdout);
  return 0;
}
#endif

#if PPMTOLTWID_MAIN
typedef std::vector<double> Vec;

void twiddle(const Vec &z, unsigned int w, unsigned int h, Vec *lo, Vec *hi) {
  assert(w % 2 == 0 && h % 2 == 0);
  assert(z.size() == 3 * w * h);

  unsigned int nw = w/2;
  unsigned int nh = h/2;
  lo->resize(nw * nh * 3);
  hi->resize(nw * nh * 9);

  unsigned int ilo = 0, ihi = 0;

  unsigned int w3 = w * 3;
  for (unsigned int y = 0; y < h; y += 2) {
    for (unsigned int x = 0; x < w; x += 2) {
      for (unsigned int c = 0; c < 3; ++c) {
        unsigned int p = y * w3 + x * 3 + c;

        double m = (z[p] + z[p+3] + z[p+w3] + z[p+w3+3]) / 4.0;
        double l = (z[p] + z[p+w3]) / 2.0 - m;
        double t = (z[p] + z[p+3]) / 2.0 - m;
        double s = (z[p] + z[p+w3+3]) / 2.0 - m;

        (*lo)[ilo++] = m;
        (*hi)[ihi++] = 0.5 + l/2.0;
        (*hi)[ihi++] = 0.5 + t/2.0;
        (*hi)[ihi++] = 0.5 + s/2.0;
      }
    }
  }
}

int main(int argc, char **argv) {
  PPM p;
  p.read(stdin);
  std::vector<double> v;
  p.vectorize(&v);
  assert(v.size() == p.w * p.h * 3);
  assert(p.w % 2 == 0 && p.h % 2 == 0);

  std::vector<double> lo, hi;
  twiddle(v, p.w, p.h, &lo, &hi);
  assert(hi.size() % 3 == 0);
  assert(lo.size() % 3 == 0);

  std::vector<double> h;
  h.resize(hi.size() / 3);
  for (unsigned int i = 0; i < hi.size(); i += 3)
    h[i/3] = hi[i];

  fwrite(h.data(), sizeof(double), h.size(), stdout);

  std::vector<double> l;
  l.resize(lo.size() / 3);
  for (unsigned int i = 0; i < lo.size(); i += 3)
    l[i/3] = lo[i];

  fwrite(l.data(), sizeof(double), l.size(), stdout);
  return 0;
}
#endif


#if LABTOPPM_MAIN
int main(int argc, char **argv) {
  assert(argc == 3);
  unsigned int w = atoi(argv[1]);
  unsigned int h = atoi(argv[2]);

  std::vector<double> v;
  v.resize(w * h * 3);
  int ret = fread(v.data(), sizeof(double), w * h * 3, stdin);
  assert(ret == w * h * 3);

  PPM p;
  p.unvectorize(v, w, h);
  p.write(stdout);
  return 0;
}
#endif


#if LTOPPM_MAIN
int main(int argc, char **argv) {
  assert(argc == 3);
  unsigned int w = atoi(argv[1]);
  unsigned int h = atoi(argv[2]);

  std::vector<double> v;
  v.resize(w * h);
  int ret = fread(v.data(), sizeof(double), w * h, stdin);
  assert(ret == w * h);

  PPM p;
  p.unvectorizegray(v.data(), w, h);
  p.write(stdout);
  return 0;
}
#endif

#if LABTWIDTOPPM_MAIN
typedef std::vector<double> Vec;

void untwiddle(const Vec &lo, const Vec &hi, unsigned int w, unsigned int h, Vec *z) {
  assert(w % 2 == 0 && h % 2 == 0);
  z->resize(3 * w * h);

  unsigned int nw = w/2;
  unsigned int nh = h/2;
  assert(lo.size() == nw * nh * 3);
  assert(hi.size() == nw * nh * 9);

  unsigned int ilo = 0, ihi = 0;

  unsigned int w3 = w * 3;
  for (unsigned int y = 0; y < h; y += 2) {
    for (unsigned int x = 0; x < w; x += 2) {
      for (unsigned int c = 0; c < 3; ++c) {
        unsigned int p = y * w3 + x * 3 + c;

        double m = lo[ilo++];
        double l = (hi[ihi++] - 0.5) * 2.0;
        double t = (hi[ihi++] - 0.5) * 2.0;
        double s = (hi[ihi++] - 0.5) * 2.0;

        (*z)[p] = m + l + t + s;
        (*z)[p+3] = m - l + t - s;
        (*z)[p+w3] = m + l - t - s;
        (*z)[p+w3+3] = m - l - t + s;
      }
    }
  }
}


int main(int argc, char **argv) {
  assert(argc == 3);
  unsigned int w = atoi(argv[1]);
  unsigned int h = atoi(argv[2]);

  assert(w % 2 == 0 && h % 2 == 0);

  std::vector<double> hi;
  hi.resize((w * h * 3 * 3)/4);
  int ret = fread(hi.data(), sizeof(double), hi.size(), stdin);
  assert(ret == hi.size());

  std::vector<double> lo;
  lo.resize((w * h * 3) / 4);
  ret = fread(lo.data(), sizeof(double), lo.size(), stdin);
  assert(ret == lo.size());

  PPM p;
  std::vector<double> v;
  untwiddle(lo, hi, w, h, &v);
  
  p.unvectorize(v, w, h);
  p.write(stdout);
  return 0;
}
#endif



