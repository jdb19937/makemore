#define __MAKEMORE_PARTRAIT_CC__ 1

#include <stdlib.h>
#include <string.h>

#include <string>

#include "partrait.hh"
#include "strutils.hh"
#include "imgutils.hh"
#include "warp.hh"
#include "pose.hh"
#include "triangle.hh"
#include "cudamem.hh"
#include "numutils.hh"
#include "strutils.hh"

namespace makemore {

Partrait::Partrait() {
  w = 0;
  h = 0;
  rgb = NULL;
}

Partrait::Partrait(unsigned int _w, unsigned int _h) : w(_w), h(_h) {
  rgb = new uint8_t[w * h * 3];
}

Partrait::~Partrait() {
  clear();
}

bool Partrait::empty() const {
  return (w == 0 || h == 0 || rgb == NULL);
}


void Partrait::clear() {
  tags.clear();

  if (rgb) {
    delete[] rgb;
    rgb = NULL;
  }
  w = h = 0;
}

void Partrait::load(const std::string &fn) {
  clear();

  std::string png = makemore::slurp(fn);
  bool ret = pngrgb(png, &w, &h, &rgb, &tags);
  assert(ret);
  assert(!empty());
}

void Partrait::save(const std::string &fn) const {
  std::string png;
  to_png(&png);
  makemore::spit(png, fn);
}

void Partrait::save(FILE *fp) const {
  std::string png;
  to_png(&png);
  makemore::spit(png, fp);
}
  

void Partrait::to_png(std::string *png) const {
  assert(!empty());


  bool ret = rgbpng(rgb, w, h, png, &tags);
  assert(ret);
}

Pose Partrait::get_pose() const {
  return Pose(get_mark());
}

void Partrait::set_pose(const Pose &pose) {
  set_mark(pose.mark());
}

bool Partrait::has_pose() const {
  return has_mark();
}

bool Partrait::has_mark() const {
  bool got_left = 0, got_right = 0, got_mouth = 0;

  for (auto tag : tags) {
    if (strbegins(tag, "left_eye:"))
      got_left = true;
    else if (strbegins(tag, "right_eye:"))
      got_right = true;
    else if (strbegins(tag, "mouth:"))
      got_mouth = true;
  }

  return (got_left && got_right && got_mouth);
}

bool Partrait::has_auto_mark() const {
  bool got_left = 0, got_right = 0, got_mouth = 0;

  for (auto tag : tags) {
    if (strbegins(tag, "auto_left_eye:"))
      got_left = true;
    else if (strbegins(tag, "auto_right_eye:"))
      got_right = true;
    else if (strbegins(tag, "auto_mouth:"))
      got_mouth = true;
  }

  return (got_left && got_right && got_mouth);
}

Triangle Partrait::get_mark() const {
  Triangle mark;
  bool got_left = 0, got_right = 0, got_mouth = 0;

  for (auto tag : tags) {
    const char *t = tag.c_str();
    if (!strncmp(t, "left_eye:", 9)) {
      t += 9;
      mark.p.x = atoi(t);
      assert(t = strchr(t, ','));
      t += 1;
      mark.p.y = atoi(t);
      got_left = true;
    } else if (!strncmp(t, "right_eye:", 10)) {
      t += 10;
      mark.q.x = atoi(t);
      assert(t = strchr(t, ','));
      t += 1;
      mark.q.y = atoi(t);
      got_right = true;
    } else if (!strncmp(t, "mouth:", 6)) {
      t += 6;
      mark.r.x = atoi(t);
      assert(t = strchr(t, ','));
      t += 1;
      mark.r.y = atoi(t);
      got_mouth = true;
    } 
  }

  assert(got_left && got_right && got_mouth);
  return mark;
}

Triangle Partrait::get_auto_mark() const {
  Triangle mark;
  bool got_left = 0, got_right = 0, got_mouth = 0;

  for (auto tag : tags) {
    const char *t = tag.c_str();
    if (!strncmp(t, "auto_left_eye:", 14)) {
      t += 14;
      mark.p.x = atoi(t);
      assert(t = strchr(t, ','));
      t += 1;
      mark.p.y = atoi(t);
      got_left = true;
    } else if (!strncmp(t, "auto_right_eye:", 15)) {
      t += 15;
      mark.q.x = atoi(t);
      assert(t = strchr(t, ','));
      t += 1;
      mark.q.y = atoi(t);
      got_right = true;
    } else if (!strncmp(t, "auto_mouth:", 11)) {
      t += 11;
      mark.r.x = atoi(t);
      assert(t = strchr(t, ','));
      t += 1;
      mark.r.y = atoi(t);
      got_mouth = true;
    } 
  }

  assert(got_left && got_right && got_mouth);
  return mark;
}

void Partrait::set_mark(const Triangle &tri) {
  std::vector<std::string> vtags;

  for (auto tag : tags) {
    if (strbegins(tag, "left_eye:") || strbegins(tag, "right_eye:") || strbegins(tag, "mouth:"))
      continue;
    vtags.push_back(tag);
  }
  
  {
    char posebuf[256];
    sprintf(posebuf, "left_eye:%ld,%ld", lround(tri.p.x), lround(tri.p.y));
    vtags.push_back(posebuf);
    sprintf(posebuf, "right_eye:%ld,%ld", lround(tri.q.x), lround(tri.q.y));
    vtags.push_back(posebuf);
    sprintf(posebuf, "mouth:%ld,%ld", lround(tri.r.x), lround(tri.r.y));
    vtags.push_back(posebuf);
  }

  tags = vtags;
}

void Partrait::set_auto_mark(const Triangle &tri) {
  std::vector<std::string> vtags;

  for (auto tag : tags) {
    if (strbegins(tag, "auto_left_eye:") || strbegins(tag, "auto_right_eye:") || strbegins(tag, "auto_mouth:"))
      continue;
    vtags.push_back(tag);
  }
  
  {
    char posebuf[256];
    sprintf(posebuf, "auto_left_eye:%ld,%ld", lround(tri.p.x), lround(tri.p.y));
    vtags.push_back(posebuf);
    sprintf(posebuf, "auto_right_eye:%ld,%ld", lround(tri.q.x), lround(tri.q.y));
    vtags.push_back(posebuf);
    sprintf(posebuf, "auto_mouth:%ld,%ld", lround(tri.r.x), lround(tri.r.y));
    vtags.push_back(posebuf);
  }

  tags = vtags;
}

void Partrait::warp(Partrait *to) const {
  assert(!to->empty());
  assert(!empty());

  Triangle fmark = get_mark();
  assert(to->has_mark());
  Triangle tmark = to->get_mark();

  to->tags = tags;
  to->set_mark(tmark);

  Point p = makemore::trimap(Point(0, 0), tmark, fmark);
  Point q = makemore::trimap(Point(to->w, 0), tmark, fmark);
  Point r = makemore::trimap(Point(0, to->h), tmark, fmark);

  kwarp(
    rgb, w, h,
    p.x, p.y, q.x, q.y, r.x, r.y,
    NULL, NULL, NULL, NULL, NULL, NULL,
    to->w, to->h, to->rgb
  );
}

void Partrait::warpover(Partrait *to) const {
  assert(!to->empty());
  assert(!empty());

  Triangle fmark = get_mark();
  assert(to->has_mark());
  Triangle tmark = to->get_mark();

  to->tags = tags;
  to->set_mark(tmark);

  Point p = makemore::trimap(Point(0, 0), tmark, fmark);
  Point q = makemore::trimap(Point(to->w, 0), tmark, fmark);
  Point r = makemore::trimap(Point(0, to->h), tmark, fmark);

  kwarpover(
    rgb, w, h,
    p.x, p.y, q.x, q.y, r.x, r.y,
    NULL, NULL, NULL, NULL, NULL, NULL,
    to->w, to->h, to->rgb
  );
}

void Partrait::encudub(double *cubuf) const {
  assert(rgb && w > 0 && h > 0);
  double *tmp = new double[w * h * 3];
  btodv(rgb, tmp, w * h * 3);
  makemore::encude(tmp, w * h * 3, cubuf);
  delete[] tmp;
}

void Partrait::reflect() {
  unsigned int w2 = w / 2;
  for (unsigned int y = 0; y < h; ++y) {
    for (unsigned int x = 0; x < w2; ++x) { 
      for (unsigned int c = 0; c < 3; ++c) {
        std::swap(rgb[y * w * 3 + x * 3 + c], rgb[y * w * 3 + (w - 1 - x) * 3 + c]);
      }
    }
  }

  if (has_mark()) {
    Triangle mark = get_mark();
    std::swap(mark.p, mark.q);
    mark.p.x = w - 1 - mark.p.x;
    mark.q.x = w - 1 - mark.q.x;
    mark.r.x = w - 1 - mark.r.x;
    set_mark(mark);
  }
}

bool Partrait::read_ppm(FILE *fp) {
  int ret = getc(fp);
  if (ret == EOF)
    return false;
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

  if (rgb)
    delete[] rgb;
  rgb = new uint8_t[3 * w * h];

  assert(3 * w * h == fread(rgb, 1, 3 * w * h, fp));
  return true;
}

void Partrait::write_ppm(FILE *fp) {
  fprintf(fp, "P6\n%u %u\n255\n", w, h);
  assert(3 * w * h == fwrite(rgb, 1, 3 * w * h, fp));
  fflush(fp);
}

void Partrait::make_sketch(double *sketch) {
  assert(w % 8 == 0 && h % 8 == 0);

  const unsigned int w8 = w / 8;
  const unsigned int h8 = h / 8;
  const unsigned int wh8 = w8 * h8;

  for (unsigned int b = 0; b < 8; ++b) {
    for (unsigned int a = 0; a < 8; ++a) {
      uint8_t col[3];

      for (unsigned int c = 0; c < 3; ++c) {
        unsigned long v = 0;
        unsigned int x0 = a * w8, x1 = x0 + w8;
        unsigned int y0 = b * h8, y1 = y0 + h8;
        for (unsigned int y = y0; y < y1; ++y) {
          for (unsigned int x = x0; x < x1; ++x) {
            v += rgb[y * w * 3 + x * 3 + c];
          }
        }
        col[c] = (uint8_t)(v / wh8);
      }

      rgbtolab(col[0], col[1], col[2], sketch + 0, sketch + 1, sketch + 2);
      sketch += 3;
    }
  }
}

}
