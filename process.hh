#ifndef __MAKEMORE_PROCESS_HH__
#define __MAKEMORE_PROCESS_HH__ 1

#include <assert.h>
#include <stdlib.h>
#include <ucontext.h>

#include <string>
#include <vector>
#include <list>

#include "strutils.hh"
#include "tmutils.hh"
#include "command.hh"
#include "io.hh"
#include "word.hh"

namespace makemore {

struct Process {
  const static unsigned long stack_size = (1 << 20);

  class System *system;
  class Session *session;

  typedef enum {
    MODE_BEGIN,
    MODE_RESUME,
    MODE_THINKING,
    MODE_READING,
    MODE_WRITING,
    MODE_WAITING,
    MODE_DONE
  } Mode;

  volatile Mode mode;

  typedef enum { 
    FLAG_CUDA_IN = 0x01,
    FLAG_CUDA_OUT = 0x02,
    FLAG_CUDA = 0x03
  } Flags;

  Flags flags;

  uint8_t stack[stack_size];
  ucontext_t me;
  ucontext_t uc;
  volatile bool inuc;

  Command func;
  std::string cmd;
  strvec args;

  std::vector<IO*> itab;
  std::vector<IO*> otab;
  int waitfd;

  bool woke;
  double scheduled;

  struct Process *prev_sproc, *next_sproc;
  struct Process *prev_proc, *next_proc;
  struct Process *prev_woke, *next_woke;
  struct Process *prev_done, *next_done;
  void wake();

  Process(
    System *_system,
    Session *_session,
    const std::string &cmd,
    const strvec &_args,
    IO *_in, IO *_out
  );

  ~Process();

  bool read(strvec *x, int ifd = 0, bool ignore_eof = false);
  Line *read(int ifd = 0, bool ignore_eof = false);
  Line *peek(int ifd = 0);
  bool write(Line *, int ofd = 0);
  bool write(const strvec &s, int ofd = 0);

  void run();

  void sleep(double dt);

  void yield(Mode newmode = MODE_THINKING, int _waitfd = -1) {
    assert(newmode != MODE_BEGIN);
    assert(mode != MODE_DONE);
    assert(inuc);
    mode = newmode;
    waitfd = _waitfd;
    inuc = false;
    ::swapcontext(&uc, &me);
    assert(mode != MODE_DONE);
    assert(inuc);
  }

  void finish() {
    this->yield(MODE_DONE);
  }
};

}

#endif
