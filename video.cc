#define __MAKEMORE_VIDEO_CC__ 1

#include "video.hh"

#include <sys/types.h>
#include <sys/wait.h>

namespace makemore {

Video::Video() {
  fp = NULL;
  mode = 0;
  pid = 0;
}

Video::~Video() {
  close();
}

void Video::open_read(const std::string &_fn) {
  open(_fn, O_RDONLY);
}

void Video::open_write(const std::string &_fn) {
  open(_fn, O_RDWR);
}

void Video::open(const std::string &_fn, int _mode) {
  assert:(!fp);

  fn = _fn;
  mode = _mode;

  if (mode == O_RDONLY) {
    int fd[2];
    int ret = ::pipe(fd);
    assert(ret == 0);

    pid = ::fork();
    if (!pid) {
      ::close(fd[0]);
      if (fd[1] != 1) {
        ::dup2(fd[1], 1);
        ::close(fd[1]);
      }
      ::execlp("ffmpeg", "ffmpeg", "-i", fn.c_str(), "-f", "rawvideo", "-vcodec", "ppm", "-y", "/dev/stdout", NULL);
      assert(0);
    }

    ::close(fd[1]);
    fp = fdopen(fd[0], "r");
    assert(fp);
  } else if (mode == O_RDWR) {
    int fd[2];
    int ret = ::pipe(fd);
    assert(ret == 0);

    pid = ::fork();
    if (!pid) {
      ::close(fd[1]);
      if (fd[0] != 0) {
        ::dup2(fd[0], 0);
        ::close(fd[0]);
      }

//      ::execlp("ffmpeg", "ffmpeg", "-f", "rawvideo", "-vcodec", "ppm", "-i", "/dev/stdin", "-f", "mp4", "-y", fn.c_str(), NULL);
      ::execlp("ffmpeg", "ffmpeg", "-r", "32", "-i", "/dev/stdin", "-r", "32", "-f", "mp4", "-y", fn.c_str(), NULL);
      assert(0);
    }

    ::close(fd[0]);
    fp = fdopen(fd[1], "w");
    assert(fp);
  } else {
    assert(0);
  }
}

void Video::close() {
  if (fp) {
    fclose(fp);
    waitpid(pid, NULL, 0);
    fp = NULL;
    pid = 0;
    mode = 0;
  }
}

bool Video::read(Partrait *prtp) {
  return prtp->read_ppm(fp);
}

void Video::write(const Partrait &prt) {
  prt.write_ppm(fp);
}

}
