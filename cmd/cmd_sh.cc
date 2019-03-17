#include <string>

#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>
#include <session.hh>

using namespace makemore;
using namespace std;

extern "C" void mainmore(Process *);



Process *do_shell(
  Process *shell, bool use_input, const strvec *argsp
) {
  IO *out = shell->otab[0];
  const strvec &args0 = *argsp;
  strvec args;

  args = args0;


again:
  string cmd;
  Command func;

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

    Process *child = new Process(
      shell->system, shell->session, "then", dargs,
      shell->itab[0], out
    );

    child->itab.resize(2);
    child->itab[1] = new IO;
    child->itab[1]->link_reader(child);
fprintf(stderr, "it1=%ld\n", (long)child->itab[1]);
    out = child->itab[1];

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
      shell->system, shell->session, "sh", dargs,
      NULL, out
    );
    out = child->itab[0];

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

  cmd = args[0];
  func = find_command(cmd);
  if (!func)
    goto fail;

  {
    strvec args1 = strvec(args.begin() + 1, args.end());

    args.resize(args1.size());
    for (unsigned int i = 0, j = 0, n = args1.size(); i < n; ++i) {
      if (!args1[i].length()) {
        args[j++] = "";
        continue;
      }

      switch (args1[i][0]) {
      case '\\':
        args[j++] = args1[i].c_str() + 1;
        break;
      case '$':
        args[j++] = shell->session->wordvar[args1[i].c_str() + 1];
        break;
      case '@':
        {
          Line tmpline;
          tmpline = shell->session->linevar[args1[i].c_str() + 1];
          strvec tmp;
          line_to_strvec(tmpline, &tmp);

          assert(args.size() + tmp.size() > 0);
          args.resize(args.size() + tmp.size() - 1);
          for (auto word : tmp)
            args[j++] = word;
          break;
        }
      default:
        args[j++] = args1[i];
        break;
      }
    }

    Process *main;
    if (use_input) {
fprintf(stderr, "hi args=[%s]\n", joinwords(args).c_str());
      if (out == shell->otab[0]) {
fprintf(stderr, "here1\n");
        strvec bak_args = args;
        shell->args = args;
        shell->cmd = cmd;

        func(shell);

        shell->args = bak_args;
        shell->cmd = "sh";
      } else {
fprintf(stderr, "here2\n");
        unsigned int otabn = shell->otab.size();
        assert(otabn);

        shell->otab.resize(otabn + 1);
        shell->otab[otabn] = shell->otab[0];
        shell->otab[0] = out;
        shell->otab[0]->link_writer(shell);

        strvec bak_args = args;
        shell->cmd = cmd;
        shell->args = args;

fprintf(stderr, "calling func\n");
        func(shell);
fprintf(stderr, "back from calling func\n");

        shell->args = bak_args;
        shell->cmd = "sh";

        shell->otab[0]->unlink_writer(shell);
        shell->otab[0] = shell->otab[otabn];
        shell->otab.resize(otabn);
fprintf(stderr, "back from calling func %u\n", otabn);
      }

#if 0
      main = new Process(
        shell->system, shell->session, cmd, args,
        shell->in, out
      );

      assert(shell->in->nreaders >= 2);

      shell->in->unlink_reader(shell);
      IO *nullio = new IO;
      shell->in = nullio;
      nullio->link_reader(shell);
#endif
    } else {
      main = new Process(
        shell->system, shell->session, cmd, args,
        NULL, out
      );
    }
  }
  return NULL;

fail:
  {
    strvec failargs;
    failargs.resize(1);
    failargs[0] = "error";
    catstrvec(failargs, args);
    args = failargs;
  }
  
  new Process(
    shell->system, shell->session, "echo", args,
    NULL, shell->otab[0]
  );

  return NULL;
}

void mainmore(
  Process *process
) {
fprintf(stderr, "sh args=[%s]\n", joinwords(process->args).c_str());

  if (process->args.size() == 0) {
    strvec args;
    while (process->read(&args)) {
      do_shell(process, false, &args);
    }
  } else {
    do_shell(process, true, &process->args);
  }
}
