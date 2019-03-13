#define __MAKEMORE_SESSION_CC__ 1
#include "session.hh"
#include "process.hh"
#include "agent.hh"
#include "server.hh"
#include "cudamem.hh"

namespace makemore {

Session::Session(Agent *_agent) {
  agent = _agent;
  assert(agent);
  server = agent->server;
  assert(server);
  assert(server->urb);

  who = new Urbite(server->urb);

  prev_reader = NULL;
  next_reader = NULL;
  prev_writer = NULL;
  next_writer = NULL;
  head_sproc = NULL;

  Command shfunc = find_command("sh");
  assert(shfunc);
  strvec no_args;

  IO *shell_in = new IO;
  shell_in->link_writer(this);

  IO *shell_out = new IO;
  shell_out->link_reader(this);

  shell = new Process(
    server->system,
    this,
    shfunc,
    no_args,
    shell_in,
    shell_out
  );
}

Session::~Session() {
  if (shell) {
    shell->in->unlink_writer(this);
    shell->out->unlink_reader(this);
  }

  while (head_sproc)
    delete head_sproc;
  assert(shell == NULL);
  delete who;

  for (auto ptrlen : cudavar) {
    void *ptr = ptrlen.first;
    cufree(ptr);
  }
  cudavar.clear();
}

void Session::link_sproc(Process *p) {
  p->prev_sproc = NULL;
  p->next_sproc = head_sproc;
  if (head_sproc) {
    assert(!head_sproc->prev_sproc);
    head_sproc->prev_sproc = p;
  }
  head_sproc = p;
}

void Session::unlink_sproc(Process *p) {
  if (p->next_sproc)
    p->next_sproc->prev_sproc = p->prev_sproc;
  if (p->prev_sproc)
    p->prev_sproc->next_sproc = p->next_sproc;
  if (head_sproc == p)
    head_sproc = p->next_sproc;
  p->prev_sproc = NULL;
  p->next_sproc = NULL;

  if (p == shell)
    shell = NULL;
}

void *Session::cumakevar(unsigned int len) {
  uint8_t *p;
  cumake(&p, len);
  cudavar.push_back(std::make_pair(p, len));
  return ((void *)p);
}

}
