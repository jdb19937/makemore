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
  Command cmd = p->cmd;
  cmd(p);
  p->finish();
  assert(0);
}

Process::Process(System *_system, Session *_session, Command _cmd, const strvec &_args, IO *_in, IO *_out) {

  system = _system;

  session = _session;
  session->link_sproc(this);

  in = _in ? _in : new IO;
  in->link_reader(this);
  out = _out ? _out : new IO;
  out->link_writer(this);

  cmd = _cmd;
  args = _args;


  ::getcontext(&uc);
  uc.uc_stack.ss_sp = stack;
  uc.uc_stack.ss_size = sizeof(stack);
  uc.uc_stack.ss_flags = 0;
  uc.uc_link = 0;
  ::makecontext(&uc, (void (*)())_call_cmd, 2, this, NULL);
  inuc = false;

  this->mode = MODE_BEGIN;

  system->link_proc(this);

  this->flags = (Flags)0;

  woke = false;
  wake();
}

Process::~Process() {
  in->unlink_reader(this);
  in = NULL;

  out->unlink_writer(this);
  out = NULL;

  session->unlink_sproc(this);
  session = NULL;

  if (woke)
    system->unlink_woke(this);
  if (mode == MODE_DONE)
    system->unlink_done(this);

  system->unlink_proc(this);

}

bool Process::write(const strvec &vec) {
  assert(inuc);

  while (!out->can_put()) {
    if (out->done_put())
      return false;
    this->yield(MODE_WRITING);
  }

  bool ret = out->put(vec);
  assert(ret);
  return true;
}


void Process::run() {
  assert(woke);
  assert(mode != MODE_DONE);
  assert(!inuc);
  inuc = true;
  ::swapcontext(&me, &uc);
  assert(!inuc);

  switch (mode) {
  default:
    assert(0);
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


strvec *Process::read() {
  assert(inuc);

  while (!in->can_get()) {
    if (in->done_get())
      return NULL;
    this->yield(MODE_READING);
  }

  strvec *ret = in->get();
  assert(ret);
  return ret;
}

strvec *Process::peek() {
  assert(inuc);

  while (!in->can_get()) {
    if (in->done_get())
      return NULL;
    this->yield(MODE_READING);
  }

  strvec *ret = in->peek();
  assert(ret);
  return ret;
}

void Process::wake() {
  if (woke)
    return;

  system->link_woke(this);

  woke = true;
}


void Process::sleep() {
  if (!woke)
    return;

  system->unlink_woke(this);

  woke = false;
}

}


