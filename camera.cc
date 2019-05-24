#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include <linux/videodev2.h>

#include <string>
#include <algorithm>

#include "camera.hh"
#include "warp.hh"
#include "partrait.hh"

namespace makemore {

Camera::Camera(const std::string &_devfn) {
  w = 1280;
  h = 720;
  tmp = new uint8_t[w * h * 3];
  devfn = _devfn;
  fd = -1;
  buffers = NULL;
  n_buffers = 0;
}

Camera::~Camera() {
  close();
  delete[] tmp;
}

void Camera::open() {
  struct stat st;

  assert(fd == -1);
  assert(0 == stat(devfn.c_str(), &st));
  assert(S_ISCHR(st.st_mode));

//  fd = ::open(devfn.c_str(), O_RDWR | O_NONBLOCK, 0);
  fd = ::open(devfn.c_str(), O_RDWR, 0);

  assert(fd != -1);


  struct v4l2_capability cap;
  struct v4l2_cropcap cropcap;
  struct v4l2_crop crop;
  struct v4l2_format fmt;
  struct v4l2_requestbuffers req;
  unsigned int min;

  assert(0 == ioctl(fd, VIDIOC_QUERYCAP, &cap));
  assert(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE);
  assert(cap.capabilities & V4L2_CAP_STREAMING);

#if 0
  memset(&cropcap, 0, sizeof(cropcap));
  cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

  if (0 == ioctl(fd, VIDIOC_CROPCAP, &cropcap)) {
    crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    crop.c = cropcap.defrect;
    (void)ioctl(fd, VIDIOC_S_CROP, &crop);
  }
#endif

  memset(&fmt, 0, sizeof(fmt));

  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = w;
  fmt.fmt.pix.height = h;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
  fmt.fmt.pix.field       = V4L2_FIELD_INTERLACED;
  assert(0 == ioctl(fd, VIDIOC_S_FMT, &fmt));

  memset(&req, 0, sizeof(req));
  req.count = 4;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;

  assert(0 == ioctl(fd, VIDIOC_REQBUFS, &req));
  assert(req.count >= 2);

  buffers = new Buffer[req.count];
  assert(buffers);

  for (n_buffers = 0; n_buffers < req.count; ++n_buffers) {
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));

    buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory      = V4L2_MEMORY_MMAP;
    buf.index       = n_buffers;

    assert(0 == ioctl(fd, VIDIOC_QUERYBUF, &buf));

    buffers[n_buffers].length = buf.length;
    buffers[n_buffers].start = mmap(
      NULL, buf.length, PROT_READ | PROT_WRITE,
      MAP_SHARED, fd, buf.m.offset);
    assert(buffers[n_buffers].start != MAP_FAILED);
  }

  for (unsigned int i = 0; i < n_buffers; ++i) {
    struct v4l2_buffer buf;

    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;

    assert (0 == ioctl(fd, VIDIOC_QBUF, &buf));
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    assert (0 == ioctl(fd, VIDIOC_STREAMON, &type));
  }
}

void Camera::close() {
  if (fd == -1)
    return;

  enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  (void)ioctl(fd, VIDIOC_STREAMOFF, &type);
  ::close(fd);
  fd = -1;

  for (unsigned int i = 0; i < n_buffers; ++i) {
    (void)munmap(buffers[i].start, buffers[i].length);
  }
  delete[] buffers;
  buffers = 0;
}

static inline uint8_t clip(double x) {
  return (x >= 0xFF ? 0xFF : (x <= 0x00 ? 0x00 : x));
}

static void yuv422toRGB(const uint8_t *src, int w, int h, uint8_t *dst) {
  int row, col;
  const uint8_t *py, *pu, *pv;
  uint8_t *tmp = dst;

  py = src;
  pu = src + 1;
  pv = src + 3;

  for (row = 0; row < h; ++row) {
    for (col = 0; col < w; ++col) {
      *tmp++ = clip((double)*py + 1.402 * ((double)*pv - 128.0));
      *tmp++ = clip((double)*py - 0.344 * ((double)*pu - 128.0) - 0.714 * ((double)*pv - 128.0));      
      *tmp++ = clip((double)*py + 1.772 * ((double)*pu - 128.0));

      py += 2;
      if (col & 1) {
        pu += 4;
        pv += 4;
      }
    }
  }
}


void Camera::read(Partrait *par, bool reflect) {
  struct v4l2_buffer buf;
  unsigned int i;

  memset(&buf, 0, sizeof(buf));

  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_MMAP;

  int ret = ioctl(fd, VIDIOC_DQBUF, &buf);
  assert(ret == 0);
  assert(buf.index < n_buffers);

  fprintf(stderr, "i=%u bs=%u [%lu]\n", buf.index, buf.bytesused, buffers[buf.index].length);
  assert(buffers[buf.index].length == buf.bytesused);
  assert(buffers[buf.index].length == w * h * 2);

assert(par->w == w && par->h == h);
  if (par->w == w && par->h == h) {
    yuv422toRGB((const uint8_t *)buffers[buf.index].start, w, h, par->rgb);
  } else {
    yuv422toRGB((const uint8_t *)buffers[buf.index].start, w, h, tmp);

    int d = w / 2;
    if (h/2 < d)
      d = h/2;

    int x0 = w/2 - d;
    int y0 = h/2 - d;
    int x1 = x0 + d * 2;
    int y1 = y0;
    int x2 = x0;
    int y2 = y0 + d * 2;

    kwarp(tmp, w, h, x0, y0, x1, y1, x2, y2, 
      NULL, NULL, NULL, NULL, NULL, NULL,
      par->w, par->h, par->rgb);
  }

  if (reflect) {
    unsigned int fw2 = par->w / 2;
    for (unsigned int y = 0; y < par->h; ++y) {
      for (unsigned int x = 0; x < fw2; ++x) { 
        for (unsigned int c = 0; c < 3; ++c) {
          std::swap(par->rgb[y * par->w * 3 + x * 3 + c], par->rgb[y * par->w * 3 + (par->w - 1 - x) * 3 + c]);
        }
      }
    }
  }

  assert(0 == ioctl(fd, VIDIOC_QBUF, &buf));
}

void Camera::read(uint8_t *rgb, unsigned int fw, unsigned int fh, bool reflect) {
  struct v4l2_buffer buf;
  unsigned int i;

  memset(&buf, 0, sizeof(buf));

  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_MMAP;

  int ret = ioctl(fd, VIDIOC_DQBUF, &buf);
  assert(ret == 0);
  assert(buf.index < n_buffers);

  fprintf(stderr, "i=%u bs=%u [%lu]\n", buf.index, buf.bytesused, buffers[buf.index].length);
  assert(buffers[buf.index].length == buf.bytesused);
  assert(buffers[buf.index].length == w * h * 2);

  if (fw == w && fh == h) {
    yuv422toRGB((const uint8_t *)buffers[buf.index].start, w, h, rgb);
  } else {
    yuv422toRGB((const uint8_t *)buffers[buf.index].start, w, h, tmp);

#if 1
    kwarp(tmp, w, h, 0, 0, w - 1, 0, 0, h - 1, 
      NULL, NULL, NULL, NULL, NULL, NULL,
      fw, fh, rgb);
#else

//assert(fw == 400 && fh == 400);
int d = w / 2;
if (h/2 < d)
  d = h/2;

int x0 = w/2 - d;
int y0 = h/2 - d;
int x1 = x0 + d * 2;
int y1 = y0;
int x2 = x0;
int y2 = y0 + d * 2;

    kwarp(tmp, w, h, x0, y0, x1, y1, x2, y2, 
      NULL, NULL, NULL, NULL, NULL, NULL,
      fw, fh, rgb);
#endif
  }

  if (reflect) {
    unsigned int fw2 = fw / 2;
    for (unsigned int y = 0; y < fh; ++y) {
      for (unsigned int x = 0; x < fw2; ++x) { 
        for (unsigned int c = 0; c < 3; ++c) {
          std::swap(rgb[y * fw * 3 + x * 3 + c], rgb[y * fw * 3 + (fw - x) * 3 + c]);
        }
      }
    }
  }

  assert(0 == ioctl(fd, VIDIOC_QBUF, &buf));
}

}

#if MAIN
int main() {
  using namespace makemore;

  Camera cam;
  cam.open();

  uint8_t *rgb = new uint8_t[cam.w * cam.h * 3];
  printf("P6\n%u %u 255\n", cam.w, cam.h);
  cam.read(rgb);

  (void)fwrite(rgb, 1, cam.w * cam.h * 3, stdout);
  return 0;
}
#endif
