#define __MAKEMORE_PROCESS_CC__ 1
#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "process.hh"
#include "strutils.hh"
#include "system.hh"
#include "session.hh"

namespace makemore {

using namespace std;

static void _call_cmd(Process *p, ...) {
fprintf(stderr, "calling cmd args=[%s]\n", joinwords(p->args).c_str());
  assert(p->inuc);
  Command f = p->func;
  f(p);
  p->finish();
  assert(0);
}

Process::Process(System *_system, Session *_session, const string &_cmd, const strvec &_args, IO *_in, IO *_out) {

  system = _system;
  system->link_proc(this);

  session = _session;
  session->link_sproc(this);

  itab.resize(1);
  itab[0] = _in ? _in : new IO;
  itab[0]->link_reader(this);

  otab.resize(1);
  otab[0] = _out ? _out : new IO;
  otab[0]->link_writer(this);

  cmd = _cmd;
  func = find_command(cmd);
  args = _args;

  ::getcontext(&uc);
  uc.uc_stack.ss_sp = stack;
  uc.uc_stack.ss_size = sizeof(stack);
  uc.uc_stack.ss_flags = 0;
  uc.uc_link = 0;
  ::makecontext(&uc, (void (*)())_call_cmd, 2, this, NULL);
  inuc = false;

  this->mode = MODE_BEGIN;
  this->waitfd = -1;

  this->flags = (Flags)0;

  scheduled = 0.0;
  woke = false;
  wake();
}

Process::~Process() {
  for (auto in : itab) {
    if (in)
      in->unlink_reader(this);
  }
  for (auto out : otab) {
    if (out)
      out->unlink_writer(this);
  }

  session->unlink_sproc(this);

  if (woke)
    system->unlink_woke(this);
  if (mode == MODE_DONE)
    system->unlink_done(this);

  system->unlink_proc(this);

  if (mode == MODE_WAITING) {
    auto i = system->schedule.find(std::make_pair(scheduled, this));
    assert(i != system->schedule.end());
    system->schedule.erase(i);
  }
}

bool Process::write(const strvec &sv, int ofd) {
  Line *wp = new Line;
  strvec_to_line(sv, wp);
  return this->write(wp, ofd);
}

bool Process::write(Line *vec, int ofd) {
  assert(inuc);

  IO *out = NULL;
  if (ofd >= 0 && ofd < otab.size())
    out = otab[ofd];
  if (!out)
    return false;

  while (!out->can_put()) {
    if (out->done_put())
      return false;
    this->yield(MODE_WRITING, ofd);
  }

  bool ret = out->put(vec);
  assert(ret);
  return true;
}


void Process::run() {
  assert(woke || mode == MODE_WAITING);
  assert(mode != MODE_DONE);
  assert(!inuc);
  inuc = true;
  ::swapcontext(&me, &uc);
  assert(!inuc);

  switch (mode) {
  default:
    assert(0);

  case MODE_WAITING:
    if (woke) {
      woke = false;
      system->unlink_woke(this);
    }
    system->schedule.insert(std::make_pair(scheduled, this));
    break;

  case MODE_DONE:
    system->link_done(this);
    // fall
  case MODE_READING:
  case MODE_WRITING:
    woke = false;
    system->unlink_woke(this);
    break;
  case MODE_THINKING:
    ;
  }
}

bool Process::read(strvec *xp, int ifd, bool ignore_eof) {
  Line *w = this->read(ifd);
  if (!w)
    return false;
  line_to_strvec(*w, xp);
  return true;
}

Line *Process::read(int ifd, bool ignore_eof) {
  assert(inuc);

  IO *in = NULL;
  if (ifd >= 0 && ifd < itab.size())
    in = itab[ifd];
  if (!in)
    return NULL;

  if (ignore_eof) {
    while (!in->can_get()) {
      this->yield(MODE_READING, ifd);
    }
  } else {
    while (!in->can_get()) {
      if (in->done_get())
        return NULL;
      this->yield(MODE_READING, ifd);
    }
  }

  Line *ret = in->get();
  assert(ret);
  return ret;
}

Line *Process::peek(int ifd) {
  assert(inuc);

  IO *in = NULL;
  if (ifd >= 0 && ifd < itab.size())
    in = itab[ifd];
  if (!in)
    return NULL;

  while (!in->can_get()) {
    if (in->done_get())
      return NULL;
    this->yield(MODE_READING, ifd);
  }

  Line *ret = in->peek();
  assert(ret);
  return ret;
}

void Process::wake() {
  if (woke)
    return;

  system->link_woke(this);

  woke = true;
}


void Process::sleep(double dt) {
  assert(inuc);
  scheduled = now() + dt;
  yield(MODE_WAITING);
}

}


