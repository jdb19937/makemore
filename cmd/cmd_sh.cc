#include <server.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>

using namespace makemore;

extern "C" bool mainmore(Server *, Urbite *, Process *, CommandState state, const strvec &);

bool mainmore(
  Server *server,
  Urbite *who,
  Process *process,
  CommandState state,
  const strvec &_args
) {
  static strvec no_args;

  assert (state == COMMAND_STATE_INITIALIZE);
  strvec &args = process->pre;

fprintf(stderr, "sh args=[%s]\n", joinwords(args).c_str());

again:
  unsigned int argsi;
  unsigned int argsn = args.size();

  unsigned int parens = 0;
  for (argsi = 0; argsi < argsn; ++argsi) {
    if (args[argsi] == "(") {
      ++parens;
    } else if (args[argsi] == ")") {
      if (parens == 0)
        goto fail;
      --parens;
    } else if (parens == 0 && args[argsi] == ";") {
      break;
    }
  }

  if (argsi < argsn) {
    assert(parens == 0);
    assert(args[argsi] == ";");
    strvec cargs(args.begin(), args.begin() + argsi);
    strvec dargs(args.begin() + argsi + 1, args.end());

    assert(cargs.size() == argsi);
    assert(dargs.size() + argsi + 1 == argsn);

    strvec eargs;
    eargs.resize(1);
    eargs[0] = "then";
    catstrvec(eargs, dargs);
    unsigned int eargsn = eargs.size();
    assert(eargsn == dargs.size() + 1);

    Process *child = server->add_process(
      who, mainmore, eargs, process->out_type, process->out
    );

    args = cargs;
    argsn = args.size();

    process->deref();
    process->out_type = Process::OUTPUT_TO_PROCESS;
    process->out.process = child;
    child->add_process_ref(process);
  }

  for (argsi = 0; argsi < argsn; ++argsi) {
    if (args[argsi] == "(") {
      ++parens;
    } else if (args[argsi] == ")") {
      if (parens == 0)
        goto fail;
      --parens;
    } else if (parens == 0 && args[argsi] == "|") {
      break;
    }
  }

  if (argsi < argsn) {
    assert(parens == 0);
    assert(args[argsi] == "|");
    strvec cargs(args.begin(), args.begin() + argsi);
    strvec dargs(args.begin() + argsi + 1, args.end());

    assert(cargs.size() == argsi);
    assert(dargs.size() + argsi + 1 == argsn);

    Process *child = server->add_process(
      who, mainmore, dargs, process->out_type, process->out
    );

    args = cargs;
    argsn = args.size();

    process->deref();
    process->out_type = Process::OUTPUT_TO_PROCESS;
    process->out.process = child;
    child->add_process_ref(process);
  }

  while (argsn > 0 && args[0] == "(") {
    if (args[argsn - 1] != ")")
      goto fail;

    assert(argsn >= 2);
    argsn -= 2;
    for (unsigned int i = 0; i < argsn; ++i)
      args[i] = args[i + 1];
    args.resize(argsn);
  }

  for (unsigned int i = 0; i < argsn; ++i) {
    if (args[i] == ";" || args[i] == "|")
      goto again;
  }

  parens = 0;
  for (unsigned int i = 0; i < argsn; ++i) {
    if (args[i] == "(")
      ++parens;
    else if (args[i] == ")") {
      if (parens == 0)
        goto fail;
      --parens;
    }

    assert(args[i] != ";");
    assert(args[i] != "|");
  }
  if (parens != 0)
    goto fail;

  if (argsn == 0)
    goto fail;

  {
    Command command = find_command(args[0]);
    if (!command)
      goto fail;
    process->cmd = command;

    strvec rest(args.begin() + 1, args.end());
    args = rest;
fprintf(stderr, "cmd=%s args=%s\n", args[0].c_str(), joinwords(rest).c_str());

    return command(server, who, process, state, no_args);
  }

fail:
  Command command = find_command("echo");
  assert(command);
  process->cmd = command;

  {
    strvec failargs;
    failargs.resize(1);
    failargs[0] = "error";
    catstrvec(failargs, args);
    args = failargs;
  }

  return command(server, who, process, state, no_args);
}
