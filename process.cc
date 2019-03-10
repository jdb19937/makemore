#define __MAKEMORE_PROCESS_CC
#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "process.hh"
#include "agent.hh"
#include "strutils.hh"

namespace makemore {

using namespace std;


Process::Process(Server *_server, const Urbite *_who, Command _cmd, const strvec &_pre, OutputType _out_type, OutputHandle _out) {
  server = _server;
  who = _who ? new Urbite(*_who) : NULL;

  state = NULL;
  inqm = 32;
  inqn = 0;
  cmd = _cmd;
  pre = _pre;
  done = false;

  out_type = _out_type;
  out = _out;

  switch (out_type) {
  case OUTPUT_TO_NULL:
    break;
  case OUTPUT_TO_AGENT:
    out.agent->add_process_ref(this);
    break;
  case OUTPUT_TO_PROCESS:
    out.process->add_process_ref(this);
    break;
  default:
    assert(0);
  }

  strvec no_args;
  assert( cmd(server, who, this, COMMAND_STATE_INITIALIZE, no_args) );
}

void Process::deref() {
  bool ret;
  switch (out_type) {
  case OUTPUT_TO_AGENT:
    ret = out.agent->remove_process_ref(this);
    assert(ret);
    break;
  case OUTPUT_TO_PROCESS:
    ret = out.process->remove_process_ref(this);
    assert(ret);
    break;
  case OUTPUT_TO_NULL:
    break;
  default:
    assert(0);
  }
}

Process::~Process() {
  deref();

  for (auto process : process_refs) {
    assert(process->out_type == OUTPUT_TO_PROCESS);
    assert(process->out.process == this);
    process->out_type = OUTPUT_TO_NULL;
    process->out.process = NULL;
  }

  if (who) 
    delete who;

  assert(state == NULL);
}

bool Process::out_ready() const {
  switch (out_type) {
  case OUTPUT_TO_NULL:
    return true;
  case OUTPUT_TO_AGENT:
    return (out.agent->outbufn == 0);
  case OUTPUT_TO_PROCESS:
    return out.process->in_ready();
  default: 
    assert(0);
  }
}

bool Process::put_out(const strvec &outvec, bool force) {
  switch (out_type) {
  case OUTPUT_TO_NULL:
    return true;
  case OUTPUT_TO_PROCESS:
    return out.process->put_in(outvec, force);
  case OUTPUT_TO_AGENT:
    if (!force && !out_ready())
      return false;

    out.agent->write(outvec);
    return true;
  default:
    assert(0);
  }
}

bool Process::run() {
  if (done)
    return true;

  if (in_empty()) {
    if (process_refs.empty()) {
      strvec no_args;
      if (cmd(server, who, this, COMMAND_STATE_SHUTDOWN, no_args)) {
        done = true;
      }
      return done;
    }
    return false;
  }

  const strvec &in0 = peek_in();

  if (cmd(server, who, this, COMMAND_STATE_RUNNING, in0)) {
    pop_in();
  }

  return done;
}

}

#if MAIN
  using namespace makemore;
  using namespace std;

static bool cmd_echo(Process *p, const strvec &argv) {
  fprintf(stderr, "here [%d] %s\n", argv.size(), argv[0].c_str());
  p->put_out(argv);
  return true;
}

int main() {

  strvec argv;
  argv.push_back("hi");
  argv.push_back("there");
  Process proc(cmd_echo, argv);
  proc.outctx = stdout;
  proc.outfunc = output_to_file;

  strvec eof;
  proc.put_in(eof);

  while (1) {
    bool ret = proc.run();
    fprintf(stderr, "run=%d\n", ret);
    if (ret)
      break;
  }
}

#endif

