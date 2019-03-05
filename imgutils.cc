#define __MAKEMORE_IMGUTILS_CC
#include "imgutils.hh"

#include <string.h>
#include <assert.h>

#include <math.h>

#include <map>
#include <set>
#include <vector>
#include <string>

#include <Magick++.h>

#include <png.h>

#include "strutils.hh"
#include "parson.hh"

namespace makemore {
using namespace std;

static bool _magick_initialized = false;

static void _ensure_init_magick() {
  if (!_magick_initialized) {
    fprintf(stderr, "initializing Magick++\n");
    Magick::InitializeMagick("libmakemore");
    _magick_initialized = true;
  }
}

static inline double f(double n) {
   return (n > 0.04045 ? pow((n + 0.055) / 1.055, 2.4) : n / 12.92) * 100.0;
}

void rgbtoxyz(uint8_t r, uint8_t g, uint8_t b, double *xp, double *yp, double *zp) {
  double dr = f((double)r / 255.0);
  double dg = f((double)g / 255.0);
  double db = f((double)b / 255.0);

  *xp = dr * 0.4124 + dg * 0.3576 + db * 0.1805;
  *yp = dr * 0.2126 + dg * 0.7152 + db * 0.0722;
  *zp = dr * 0.0193 + dg * 0.1192 + db * 0.9505;
}

void xyztorgb(double x, double y, double z, uint8_t *rp, uint8_t *gp, uint8_t *bp) {
  double dx = x / 100.0;
  double dy = y / 100.0;
  double dz = z / 100.0;

  double pr = dx * 3.2404542 + dy * -1.5371385 + dz * -0.4985314;
  double pg = dx * -0.9692660 + dy * 1.8760108 + dz * 0.0415560;
  double pb = dx * 0.0556434 + dy * -0.2040259 + dz * 1.0572252;

  double qr = (((pr > 0.0031308) ? (1.055*pow(pr, 1 / 2.4) - 0.055) : (12.92*pr)) * 255.0);
  double qg = (((pg > 0.0031308) ? (1.055*pow(pg, 1 / 2.4) - 0.055) : (12.92*pg)) * 255.0);
  double qb = (((pb > 0.0031308) ? (1.055*pow(pb, 1 / 2.4) - 0.055) : (12.92*pb)) * 255.0);

  if (qr < 0) qr = 0; if (qr > 255.0) qr = 255.0;
  if (qg < 0) qg = 0; if (qg > 255.0) qg = 255.0;
  if (qb < 0) qb = 0; if (qb > 255.0) qb = 255.0;

  *rp = qr;
  *gp = qg;
  *bp = qb;
}

		

static inline double g(double n) {
  const double EPS = 0.008856;
  const double KAPPA = 903.3;
  return n > EPS ? pow(n, 1.0/3.0) : (KAPPA * n + 16.0) / 116.0;
}

//double tl = 0.25;
//double tab = 0.5;

void xyztolab(double x, double y, double z, double *lp, double *ap, double *bp) {
  double dx = g(x / 95.047);
  double dy = g(y / 100.0);
  double dz = g(z / 108.883);

  *lp = 116.0 * dy - 16.0;
  if (*lp < 0)
    *lp = 0;
  *ap = 500.0 * (dx - dy);
  *bp = 200.0 * (dy - dz);

  *lp -= 50.0;
  *lp /= 100.0;
  *ap /= 100.0;
  *bp /= 100.0;

//  *lp *= tl;
//  *ap *= tab;
//  *bp *= tab;

  *lp += 0.5;
  *ap += 0.5;
  *bp += 0.5;
}

void labtoxyz(double l, double a, double b, double *xp, double *yp, double *zp) {
  if (l <  0.0) l = 0.0;
  if (l >= 1.0) l = 1.0;
  l -= 0.5;
//  l /= tl;
  l *= 100.0;
  l += 50.0;

  if (a <  0.0) a = 0.0;
  if (a >= 1.0) a = 1.0;
  a -= 0.5;
//  a /= tab;
  a *= 100.0;

  if (b <  0.0) b = 0.0;
  if (b >= 1.0) b = 1.0;
  b -= 0.5;
//  b /= tab;
  b *= 100.0;
  
  double py = (l + 16.0) / 116.0;
  double px = a / 500.0 + py;
  double pz = py - b / 200.0;

  double x3 = px*px*px;
  double y3 = py*py*py;
  double z3 = pz*pz*pz;

  *xp = ((x3 > 0.008856) ? x3 : ((px - 16.0 / 116.0) / 7.787)) * 95.047;
  *yp = ((y3 > 0.008856) ? y3 : ((py - 16.0 / 116.0) / 7.787)) * 100.0;
  *zp = ((z3 > 0.008856) ? z3 : ((pz - 16.0 / 116.0) / 7.787)) * 108.883;
}


bool imglab(
  const std::string &fmt, 
  const std::string &data,
  unsigned int w,
  unsigned int h,
  uint8_t *lab,
  vector<string> *tags
) {
  _ensure_init_magick();

  if (fmt != "jpg" && fmt != "ppm" && fmt != "png")
    return false;

  // shouldn't be bigger than raw
  unsigned long max_size = 32768 + (w * h * 3);
  if (data.size() > max_size)
    return false;

  Magick::Image image;
  image.magick(fmt);
  Magick::Blob blobin(data.data(), data.length());
  image.read(blobin);

  if (image.size().width() != w)
    return false;
  if (image.size().height() != h)
    return false;

  if (tags) {
    vector<string> words;
    splitwords(image.comment(), &words);
    for (auto word : words)
      if (*word.c_str() == '#')
        tags->push_back(word.c_str() + 1);
  }

  image.magick("rgb");
  Magick::Blob blobout(lab, w * h * 3);
  image.write(&blobout);

  for (unsigned int i = 0, n = 3 * w * h; i < n; i += 3) {
    rgbtolab(
       lab[i + 0],  lab[i + 1],  lab[i + 2],
      &lab[i + 0], &lab[i + 1], &lab[i + 2]
    );
  }

  return true;
}

bool labimg(
  const uint8_t *lab,
  unsigned int w,
  unsigned int h,
  const std::string &fmt,
  std::string *img,
  const vector<string> *tags
) {
  _ensure_init_magick();

  std::string rgbstr;
  rgbstr.resize(w * h * 3);
  uint8_t *rgb = (uint8_t *)rgbstr.data();

  for (unsigned int i = 0, n = 3 * w * h; i < n; i += 3) {
    labtorgb(
       lab[i + 0],  lab[i + 1],  lab[i + 2],
      &rgb[i + 0], &rgb[i + 1], &rgb[i + 2]
    );
  }


  Magick::Image image;

  image.magick("rgb");
  char sizebuf[64];
  sprintf(sizebuf, "%ux%u", w, h);
  image.size(sizebuf);

  Magick::Blob blobin(rgb, w * h * 3);
  image.read(blobin);

  assert(image.size().width() == w);
  assert(image.size().height() == h);

  if (tags) {
    string comment;
    for (auto tag : *tags)
      comment += string(comment.length() ? " " : "") + "#" + tag;
    image.comment(comment);
  }

  image.magick(fmt);
  Magick::Blob blobout;
  image.write(&blobout);

  img->assign((const char *)blobout.data(), blobout.length());

  return true;
}

bool pnglab(
  const std::string &png,
  unsigned int w,
  unsigned int h,
  uint8_t *lab,
  vector<string> *tags
) {
  FILE *fp = fmemopen((void *)png.data(), png.length(), "r");
  assert(fp);

  uint8_t header[8];
  size_t ret;
  png_structp png_ptr;
  png_infop info_ptr;
  png_infop end_ptr;
  png_textp text = NULL;
  png_bytep *row_pointers = NULL;

  ret = fread(header, 1, 8, fp);
  if (ret != 8)
    goto fail;
  if (png_sig_cmp(header, 0, 8))
    goto fail;

  png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png_ptr)
    goto fail;

  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
    goto fail;
  end_ptr = png_create_info_struct(png_ptr);
  if (!end_ptr)
    goto fail;

  if (setjmp(png_jmpbuf(png_ptr)))
    goto fail;

  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);

  png_read_info(png_ptr, info_ptr);

  if (w != png_get_image_width(png_ptr, info_ptr)) 
    goto fail;
  if (h != png_get_image_height(png_ptr, info_ptr))
    goto fail;
  if (PNG_COLOR_TYPE_RGB != png_get_color_type(png_ptr, info_ptr))
    goto fail;
  if (8 != png_get_bit_depth(png_ptr, info_ptr))
    goto fail;

  png_set_interlace_handling(png_ptr);
  png_read_update_info(png_ptr, info_ptr);

  /* read file */
  assert(!setjmp(png_jmpbuf(png_ptr)));

  row_pointers = new png_bytep[h];
  for (unsigned int y = 0; y < h; y++)
    row_pointers[y] = lab + y * w * 3;

  png_read_image(png_ptr, row_pointers);

  png_read_end(png_ptr, end_ptr);

  if (tags) {
    for (int j = 0; j < 2; ++j) {
      int numtext = png_get_text(png_ptr, j ? info_ptr : end_ptr, &text, NULL);
      set<string> seen_tag;

      for (int i = 0; i < numtext; ++i) {
        std::string comment(text[i].text, text[i].text_length);
        vector<string> words;
        splitwords(comment, &words);
        for (auto word : words) {
          if (*word.c_str() != '#')
            continue;
          std::string tag = word.c_str() + 1;
          if (!tag.length())
            continue;
          tag = lowercase(tag);
          if (!Parson::valid_tag(tag))
            continue;
          if (seen_tag.count(tag))
            continue;
          tags->push_back(tag);
          seen_tag.insert(tag);
        }
      }
    }
  }

  for (unsigned int i = 0, n = 3 * w * h; i < n; i += 3) {
    rgbtolab(
       lab[i + 0],  lab[i + 1],  lab[i + 2],
      &lab[i + 0], &lab[i + 1], &lab[i + 2]
    );
  }

  fclose(fp);
  if (row_pointers)
    delete[] row_pointers;

  return true;

fail:
  fclose(fp);
  if (row_pointers)
    delete[] row_pointers;

  return false;
}

bool labpng(
  const uint8_t *lab,
  unsigned int w,
  unsigned int h,
  std::string *png,
  const vector<string> *tags
) {
  uint8_t *rgb = new uint8_t[w * h * 3];

  for (unsigned int i = 0, n = 3 * w * h; i < n; i += 3) {
    labtorgb(
       lab[i + 0],  lab[i + 1],  lab[i + 2],
      &rgb[i + 0], &rgb[i + 1], &rgb[i + 2]
    );
  }

  uint8_t *pngbuf = new uint8_t[1 << 20];
  FILE *fp = fmemopen(pngbuf, 1 << 20, "wb");
  assert(fp);

  uint8_t header[8];
  size_t ret;
  png_structp png_ptr;
  png_infop info_ptr;
  png_infop end_ptr;
  png_textp text = NULL;
  png_bytep *row_pointers = NULL;

  png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png_ptr)
    goto fail;
  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
    goto fail;
  end_ptr = png_create_info_struct(png_ptr);
  if (!end_ptr)
    goto fail;

  assert(!setjmp(png_jmpbuf(png_ptr)));

  png_init_io(png_ptr, fp);

  assert(!setjmp(png_jmpbuf(png_ptr)));

  png_set_IHDR(
    png_ptr, info_ptr, w, h,
    8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
    PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE
  );

  png_write_info(png_ptr, info_ptr);

  assert(!setjmp(png_jmpbuf(png_ptr)));
  row_pointers = new png_bytep[h];
  for (unsigned int y = 0; y < h; y++)
    row_pointers[y] = rgb + y * w * 3;
  png_write_image(png_ptr, row_pointers);

  if (tags) {
    string comment;
    for (auto tag : *tags)
      comment += string(comment.length() ? " " : "") + "#" + tag;
    png_text text;
    text.compression = PNG_TEXT_COMPRESSION_NONE;
    text.key = (char *)"comment";
    text.text = (char *)comment.c_str();
    text.text_length = comment.length();
    png_set_text(png_ptr, end_ptr, &text, 1);
  }

  assert(!setjmp(png_jmpbuf(png_ptr)));
  png_write_end(png_ptr, end_ptr);

  {
    unsigned long pngbufn = ftell(fp);
    fclose(fp);
    png->assign((char *)pngbuf, pngbufn);
  }

  delete[] pngbuf;
  delete[] rgb;
  delete[] row_pointers;

  return true;

fail:
  fclose(fp);
  if (rgb)
    delete[] rgb;
  if (row_pointers)
    delete[] row_pointers;
  if (pngbuf)
    delete[] pngbuf;
  return false;
}

}