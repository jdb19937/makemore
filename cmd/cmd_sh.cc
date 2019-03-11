#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>

using namespace makemore;

extern "C" void mainmore(Process *);



Process *do_shell(
  Process *shell,
  Process *inproc, Agent *inagent, Process *outproc, Agent *outagent,
  const strvec &_args
) {

  strvec args = _args;
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

  if (parens != 0)
    goto fail;

  if (argsi < argsn) {
    assert(parens == 0);
    assert(args[argsi] == ";");
    strvec cargs(args.begin(), args.begin() + argsi);
    strvec dargs(args.begin() + argsi + 1, args.end());

    assert(cargs.size() == argsi);
    assert(dargs.size() + argsi + 1 == argsn);

    Command then = find_command("then");
    assert(then);

    Process *child = new Process(
      shell->system, shell->who, then, dargs,
      NULL, outproc,
      NULL, outagent
    );

    if (outproc) {
      assert(outproc->inproc == NULL);
      outproc->inproc = child;
    }
    outproc = child;
    outagent = NULL;

    args = cargs;
    argsn = args.size();
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

  assert(parens == 0);

  if (argsi < argsn) {
    assert(args[argsi] == "|");
    strvec cargs(args.begin(), args.begin() + argsi);
    strvec dargs(args.begin() + argsi + 1, args.end());

    assert(cargs.size() == argsi);
    assert(dargs.size() + argsi + 1 == argsn);

    Process *child = new Process(
      shell->system, shell->who, mainmore, dargs,
      NULL, outproc,
      NULL, outagent
    );

    if (outproc) {
      assert(outproc->inproc == NULL);
      outproc->inproc = child;
    }
    outproc = child;
    outagent = NULL;

    args = cargs;
    argsn = args.size();
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
    if (args[i] == ";" || args[i] == "|") {
      goto again;
    }
  }
  for (unsigned int i = 0; i < argsn; ++i) {
    assert(args[i] != ";");
    assert(args[i] != "|");
  }

  if (argsn == 0)
    goto fail;

  {
    Command command = find_command(args[0]);
    if (!command)
      goto fail;

    {
      strvec rest(args.begin() + 1, args.end());
      args = rest;
    }

fprintf(stderr, "made main[%s] outproc=%lu inproc=%lu\n", joinwords(args).c_str(), (unsigned long)outproc, (unsigned long)inproc);
    Process *main = new Process(
      shell->system, shell->who, command, args,
      inproc, outproc, NULL, outagent
    );

    if (inproc) {
      assert(inproc->outproc == NULL);
      inproc->outproc = main;
    }

    if (outproc) {
      assert(outproc->inproc == NULL);
      outproc->inproc = main;
    }

    if (inproc) {
      while (shell->inqn) {
        strvec *buffered = shell->read();
        assert(buffered);
        main->put(*buffered);
      }
    }
  }
  return NULL;

fail:
  Command echo = find_command("echo");
  assert(echo);

  {
    strvec failargs;
    failargs.resize(1);
    failargs[0] = "error";
    catstrvec(failargs, args);
    args = failargs;
  }
  
  new Process(
    shell->system, shell->who, echo, args,
    NULL, NULL, NULL, outagent
  );

  return NULL;
}

void mainmore(
  Process *process
) {
fprintf(stderr, "sh reading done=%d (pre=%s)\n", process->coro->done, joinwords(process->args).c_str());

  if (process->args.size() == 0) {
    while (const strvec *argsp = process->read()) {
      do_shell(process, NULL, NULL, process->outproc, process->outagent, *argsp);
    }
  } else {
    Process *outproc = process->outproc;
    if (outproc) {
      assert(outproc->inproc == process);
      outproc->inproc = NULL;
      process->outproc = NULL;
    }

    Process *inproc = process->inproc;
    if (inproc) {
      assert(inproc->outproc == process);
      inproc->outproc = NULL;
      process->inproc = NULL;
    }
     
    do_shell(process, inproc, process->inagent, outproc, process->outagent, process->args);
  }

  process->coro->finish();
}
