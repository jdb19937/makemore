#define __MAKEMORE_PROCESS_CC__ 1
#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "process.hh"
#include "agent.hh"
#include "strutils.hh"
#include "system.hh"

namespace makemore {

using namespace std;


Process::Process(System *_system, const Urbite *_who, Command _cmd, const strvec &_args, Process *_inproc, Process *_outproc, Agent *_inagent, Agent *_outagent) {
  system = _system;
  who = _who ? new Urbite(*_who) : NULL;

  inqm = 32;
  inqn = 0;
  cmd = _cmd;
  args = _args;

  inproc = _inproc;
  outproc = _outproc;
  inagent = _inagent;
  outagent = _outagent;

  if (outagent)
    outagent->add_writer(this);

  this->coro = new Coro<Mode>(
    (corofunc_t)cmd,
    this
  );
  this->mode = MODE_THINKING;


  Process **head_procp = &system->head_proc;
  Process *head_proc = *head_procp;
  prev_proc = NULL;
  next_proc = head_proc;
  if (head_proc) {
    assert(!head_proc->prev_proc);
    head_proc->prev_proc = this;
  }
  *head_procp = this;

  woke = false;
  wake();
}


Process::~Process() {
  delete coro;

  if (inproc) {
    assert(inproc->outproc == this);
    inproc->outproc = NULL;

    if (inproc->mode == MODE_WRITING)
      inproc->wake();
  }
  if (outproc) {
    assert(outproc->inproc == this);
    outproc->inproc = NULL;

    if (outproc->mode == MODE_READING)
      outproc->wake();
  }
  if (inagent) {
    assert(inagent->shell == this);
    inagent->shell = NULL;
  }
  if (outagent && outagent->shell == this) {
    outagent->shell = NULL;
  }
  if (who) 
    delete who;

  if (outagent) {
    outagent->remove_writer(this);
  }

  Process **head_wokep = &system->head_woke;
  Process *head_woke = *head_wokep;
  if (next_woke)
    next_woke->prev_woke = prev_woke;
  if (prev_woke)
    prev_woke->next_woke = next_woke;
  if (head_woke == this)
    *head_wokep = next_woke;

  Process **head_donep = &system->head_done;
  Process *head_done = *head_donep;
  if (next_done)
    next_done->prev_done = prev_done;
  if (prev_done)
    prev_done->next_done = next_done;
  if (head_done == this)
    *head_donep = next_done;

  Process **head_procp = &system->head_proc;
  Process *head_proc = *head_procp;
  if (next_proc)
    next_proc->prev_proc = prev_proc;
  if (prev_proc)
    prev_proc->next_proc = next_proc;
  if (head_proc == this)
    *head_procp = next_proc;
}

bool Process::write(const strvec &outvec) {
fprintf(stderr, "got write: [%s]\n", joinwords(outvec).c_str());

  assert(!outproc || !outagent);

  if (!outproc && !outagent) {
    return false;
  }

  if (outproc) {
    while (outproc && !outproc->can_put()) {
      coro->yield(MODE_WRITING);
    }

    if (!outproc)
      return false;

    assert(outproc->can_put());
    outproc->put(outvec);
    return true;
  } else {
    assert(outagent);

    while (outagent && outagent->outbufn > 0) {
      coro->yield(MODE_WRITING);
    }

    if (!outagent)
      return false;

    assert(outagent->outbufn == 0);
    outagent->write(outvec);
    return true;
  }

  assert(0);
}


void Process::put(const strvec &in) {
fprintf(stderr, "got put: [%s]\n", joinwords(in).c_str());
  assert(inqn < inqm);
  ++inqn;
  inq.push_back(in);

  if (mode == MODE_READING)
    wake();
}


strvec *Process::read() {
  assert(!inproc || !inagent);

fprintf(stderr, "got read inqn=%u inproc=%lu inagent=%lu\n", inqn, (unsigned long)inproc, (unsigned long)inagent);
  while (inqn == 0 && (inproc || inagent)) {
fprintf(stderr, "read yielding\n");
    coro->yield(MODE_READING);
  }
fprintf(stderr, "read inqn=%u\n", (unsigned int)inqn);

  if (inqn == 0) {
    assert(!inproc && !inagent);
    return NULL;
  }

  auto inqi = inq.begin();
  assert(inqi != inq.end());
  inx = *inqi;
  inq.erase(inqi);
  --inqn;

  assert(inqn < inqm);
  if (inproc && inproc->mode == MODE_WRITING)
    inproc->wake();

  return &inx;
}

void Process::wake() {
  if (woke)
    return;

  Process **head_wokep = &system->head_woke;
  Process *head_woke = *head_wokep;

  prev_woke = NULL;
  next_woke = head_woke;
  if (head_woke) {
    assert(!head_woke->prev_woke);
    head_woke->prev_woke = this;
  }
  *head_wokep = this;

  woke = true;
}

void Process::finish() {
  assert(mode == MODE_DONE);

}


bool Process::run() {
  assert(woke);

  Mode old_mode = mode;
  assert(old_mode != MODE_DONE);

  Mode *modep = coro->get();
  if (!modep) {
    Process **head_donep = &system->head_done;
    Process *head_done = *head_donep;

    prev_done = NULL;
    next_done = head_done;
    if (head_done) {
      assert(!head_done->prev_done);
      head_done->prev_done = this;
    }
    *head_donep = this;

    mode = MODE_DONE;
    return true;
  }

  mode = *modep;

  switch (mode) {
  case MODE_THINKING:
    break;
  case MODE_READING:
  case MODE_WRITING:
    sleep();
    break;
  default:
    assert(0);
  }

  return false;
}

void Process::sleep() {
  if (!woke)
    return;

  Process **head_wokep = &system->head_woke;
  Process *head_woke = *head_wokep;

  if (next_woke)
    next_woke->prev_woke = prev_woke;
  if (prev_woke)
    prev_woke->next_woke = next_woke;
  if (head_woke == this)
    *head_wokep = next_woke;

  next_woke = NULL;
  prev_woke = NULL;
 
  woke = false;
}

}


