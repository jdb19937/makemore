#ifndef __MAKEMORE_SAMPLER_HH__
#define __MAKEMORE_SAMPLER_HH__ 1

#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>

#include <string>

struct Sampler {
  pid_t feeder_pid;
  pid_t buffer_pid;

  FILE *fp;
 
  std::string fn;
  unsigned int k;

  unsigned long inbuflen;
  unsigned long membuflen;
  unsigned int batch;

  Sampler(const char *_fn, unsigned int _k, unsigned long _inbuflen, unsigned long _membuflen, unsigned int _batch);
  ~Sampler();

  void stop();
  void wait();
  void start();

  FILE *file() const { return fp; }
};

#endif
