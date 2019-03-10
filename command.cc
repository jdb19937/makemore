#define __MAKEMORE_COMMAND_CC__ 1
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <dlfcn.h>

#include <list>
#include <string>
#include <vector>
#include <map>

#include "command.hh"

namespace makemore {

using namespace std;

static map<string, Command> cmdtab;
static map<string, void *> dlptab;
static string cmdsopre = "./cmd/cmd_";
static string cmdsosuf = ".so";

Command find_command(const std::string &cmdstr) {
  auto cmdtabi = cmdtab.find(cmdstr);
  if (cmdtabi != cmdtab.end())
    return cmdtabi->second;

  for (unsigned int i = 0, n = cmdstr.length(); i < n; ++i)
    if (!((cmdstr[i] >= 'a' && cmdstr[i] <= 'z') || cmdstr[i] == '_'))
      return NULL;

  std::string dlfn = cmdsopre + cmdstr + cmdsosuf;
  void *dlp = dlopen(dlfn.c_str(), RTLD_NOW);
  if (!dlp) {
    fprintf(stderr, "dlopen %s: %s\n", dlfn.c_str(), dlerror());
    return NULL;
  }

  Command cmdfunc = (Command)dlsym(dlp, "mainmore");
  if (!cmdfunc) {
    fprintf(stderr, "dlsym %s: %s\n", dlfn.c_str(), dlerror());
    dlclose(dlp);
    return NULL;
  }

  assert(dlptab.find(cmdstr) == dlptab.end());
  dlptab[cmdstr] = dlp;

  cmdtab[cmdstr] = cmdfunc;

  return cmdfunc;
}

}
