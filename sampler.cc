#define __MAKEMORE_SAMPLER_CC__ 1

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <sys/fcntl.h>
#include <sys/types.h>
#include <sys/select.h>
#include <sys/stat.h>
#include <sys/wait.h>

#include "sampler.hh"
#include "random.hh"

Sampler::Sampler(const char *_fn, unsigned int _k, unsigned long _inbuflen, unsigned long _membuflen, unsigned int _batch) {
  feeder_pid = 0;
  buffer_pid = 0;

  k = _k;
  inbuflen = _inbuflen;
  membuflen = _membuflen;
  batch = _batch;
  fn = _fn;
  fp = NULL;
}

Sampler::~Sampler() {
  if (fp)
    fclose(fp);
  stop();
  wait();
}

void Sampler::wait() {
  int ret;
  if (feeder_pid > 0) {
    ret = ::waitpid(feeder_pid, NULL, 0);
    assert(ret == feeder_pid);
    feeder_pid = 0;
  }
  if (buffer_pid > 0) {
    ret = ::waitpid(buffer_pid, NULL, 0);
    assert(ret == buffer_pid);
    buffer_pid = 0;
  }
}

void Sampler::stop() {
  if (feeder_pid > 0)
    ::kill(feeder_pid, 9);
  if (buffer_pid > 0)
    ::kill(buffer_pid, 9);
}

static void _run_feeder(int infd, int outfd, unsigned int k, unsigned int batch) {
  assert(batch > 0);

  struct stat stbuf;
  int ret;

  ret = fstat(infd, &stbuf);
  assert(ret == 0);

  size_t size = stbuf.st_size;
  assert(size % k == 0);
  uint64_t n = size / k;

  uint8_t *buf = new uint8_t[k];

  unsigned int i = 0;
  off_t off = (randuint() % n) * k;
  off_t offret = ::lseek(infd, off, SEEK_SET);
  assert(offret == off);

  while (1) {
    if (off == size) {
      offret = ::lseek(infd, 0, SEEK_SET);
      assert(offret == 0);
      off = 0;
    }
    assert(off < size);

    ret = ::read(infd, buf, k);
    assert(ret == k);
    off += k;

    ret = ::write(outfd, buf, k);
    assert(ret == k);

    if (i % batch == 0) {
      off = (randuint() % n) * k;
      offret = lseek(infd, off, SEEK_SET);
      assert(offret == off);
    }

    usleep(1000);
    ++i;
  }
}


static void _run_buffer(
  int infd, int outfd, unsigned int k,
  uint64_t inbuflen, uint64_t membuflen
) {
  int ret;

  int topfd = infd;
  if (outfd > topfd)
    topfd = outfd;
  ++topfd;

  assert(k > 0);

  assert(membuflen % k == 0);
  uint8_t *membuf = new uint8_t[membuflen];
  uint64_t membufn = membuflen / k;

  bool *used = new bool[membufn];
  memset(used, 0, membufn * sizeof(bool));

  assert(inbuflen % k == 0);
  uint8_t *inbuf = new uint8_t[inbuflen];
  unsigned int inbufoff = 0;
  assert(inbufoff < inbuflen);

  unsigned int outbufoff = 0;
  uint8_t *outbuf = new uint8_t[k];

  while (1) {
    fd_set rfds, wfds, efds;
    FD_ZERO(&rfds);
    FD_ZERO(&wfds);
    FD_ZERO(&efds);
    FD_SET(infd, &rfds);
    FD_SET(outfd, &wfds);

    ret = ::select(topfd, &rfds, &wfds, &efds, NULL);
    assert(ret != -1);
    assert(ret > 0);
    assert(ret == 1 || ret == 2);

    if (FD_ISSET(infd, &rfds)) {
      assert(inbufoff < inbuflen);
      ret = ::read(infd, inbuf + inbufoff, inbuflen - inbufoff);
      if (ret > 0) {
        inbufoff += ret;
      } else {
        assert(ret == -1);
        assert(errno == EAGAIN || errno == EINTR);
      }
    }

    if (inbufoff >= k) {
      for (unsigned int i = 0; i < inbufoff; i += k) {
        unsigned int membufi = randuint() % membufn;
        memcpy(membuf + k * membufi, inbuf + i, k);
        used[membufi] = true;
      }

      if (unsigned int rem = inbufoff % k) {
        memcpy(inbuf, inbuf + inbufoff - rem, rem);
        inbufoff = rem;
      } else { 
        inbufoff = 0;
      }

      assert(inbufoff < k);
      assert(inbufoff < inbuflen);
    }

    if (FD_ISSET(outfd, &wfds)) {
      if (outbufoff > 0) {
        ret = ::write(outfd, outbuf, outbufoff);
        if (ret > 0) {
          if (ret < outbufoff) {
            memmove(outbuf, outbuf + ret, outbufoff - ret);
            outbufoff -= ret;
          } else {
            assert(ret == outbufoff);
            outbufoff = 0;
          }
        } else {
          assert(ret == -1);
          assert(errno == EAGAIN || errno == EINTR);
        }
      }

      if (outbufoff == 0) {
        for (
          unsigned int membufi = randuint() % membufn;
          used[membufi];
          membufi = randuint() % membufn
        ) {
          uint8_t *row = membuf + membufi * k;
          ret = ::write(outfd, row, k);

          if (ret > 0) {
            if (ret < k) {
              outbufoff = k - ret;
              memcpy(outbuf, row + ret, outbufoff);
              break;
            }
            assert(ret == k);
          } else {
            assert(ret == -1);
            assert(errno == EAGAIN || errno == EINTR);
            break;
          }
        }
      }
    }
  }
}
  
  



void Sampler::start() {
  int ret;
  int feeder_fd[2], buffer_fd[2];

  assert(!fp);
  ret = ::pipe(feeder_fd);
  assert(ret == 0);

  feeder_pid = fork();
  if (!feeder_pid) {
    FILE *infp = fopen(fn.c_str(), "r");
    assert(infp);
    ::close(feeder_fd[0]);
    _run_feeder(fileno(infp), feeder_fd[1], k, batch);
    exit(1);
  }
  ::close(feeder_fd[1]);

  ret = ::pipe(buffer_fd);
  buffer_pid = fork();
  if (!buffer_pid) {
    ::close(buffer_fd[0]);

    int flags = ::fcntl(feeder_fd[0], F_GETFL, 0);
    assert(flags != -1);
    ret = ::fcntl(feeder_fd[0], F_SETFL, flags | O_NONBLOCK);
    assert(ret == 0);

    flags = ::fcntl(buffer_fd[1], F_GETFL, 0);
    assert(flags != -1);
    ret = ::fcntl(buffer_fd[1], F_SETFL, flags | O_NONBLOCK);
    assert(ret == 0);

    _run_buffer(feeder_fd[0], buffer_fd[1], k, inbuflen, membuflen);
    exit(1);
  }
  ::close(feeder_fd[0]);
  ::close(buffer_fd[1]);

  fp = fdopen(buffer_fd[0], "r");
  assert(fp);
}

 

 
#if SAMPLER_MAIN
int main(int argc, char **argv) {
  assert(argc >= 3);
  const char *fn = argv[1];
  unsigned int k = atoi(argv[2]);
  unsigned int reps = 1;
  if (argc >= 4)
    reps = (unsigned)atoi(argv[3]);
  assert(reps > 0);
  unsigned long inbuflen = 1ULL<<20;
  unsigned long membuflen = 1ULL<<30;
  unsigned int batch = 4096;

  inbuflen -= (inbuflen % k);
  membuflen -= (membuflen % k);

  Sampler sampler(fn, k, inbuflen, membuflen, batch);
  sampler.start();

  int ret;
  uint8_t *buf = new uint8_t[k];
  unsigned int bufoff;
  while (1) {
    assert(k == fread(buf, 1, k, sampler.file()));
    for (int i = 0; i < reps; ++i)
      assert(k == fwrite(buf, 1, k, stdout));
  }

  return 0;
}
#endif
