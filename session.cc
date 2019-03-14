#define __MAKEMORE_SESSION_CC__ 1
#include <stdlib.h>
#include <sys/types.h>
#include <dirent.h>

#include "session.hh"
#include "process.hh"
#include "agent.hh"
#include "server.hh"
#include "cudamem.hh"

namespace makemore {

using namespace std;

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

  strvec no_args;

  IO *shell_in = new IO;
  shell_in->link_writer(this);

  IO *shell_out = new IO;
  shell_out->link_reader(this);

  shell = new Process(
    server->system,
    this,
    "sh",
    no_args,
    shell_in,
    shell_out
  );

  loadvars();
}

void Session::loadvars() {
  wordvar.clear();
  linevar.clear();
  gridvar.clear();

  who->make_home_dir();
  DIR *dp = opendir(who->home_dir().c_str());
  assert(dp);

  struct dirent *de;
  while ((de = readdir(dp))) {
    const char *name = de->d_name;
    if (*name == '.')
      continue;
    unsigned int namelen = strlen(name);
    if (namelen < 6)
       continue;

#if 0
    if (!strcmp(name + namelen - 5, ".grid")) {
      this->load_grid(string(name, namelen - 5));
    }
    if (!strcmp(name + namelen - 5, ".line")) {
      this->load_line(string(name, namelen - 5));
    }
#endif
    if (!strcmp(name + namelen - 5, ".word")) {
      this->load_word(string(name, namelen - 5));
    }

  }
  closedir(dp);
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

bool Session::save_grid(const string &var) {
  if (!Parson::valid_tag(var))
    return false;
  auto vari = gridvar.find(var);
  if (vari == gridvar.end())
    return false;

  string tmpfn = who->home_dir() + "/" + var + ".grid.tmp";
  FILE *fp = fopen(tmpfn.c_str(), "w");
  if (!fp)
    return false;

  for (auto linevec : vari->second) {
    std::string linestr = moretpenc(linevec, '\t');
    size_t written = fwrite(linestr.data(), 1, linestr.length(), fp);
    if (written != linestr.length()) {
      fclose(fp);
      (void) ::unlink(tmpfn.c_str());
      return false;
    }
  }

  (void) fclose(fp);

  string fn = who->home_dir() + "/" + var + ".grid";
  int ret = ::rename(tmpfn.c_str(), fn.c_str());
  if (ret != 0) {
    (void) ::unlink(tmpfn.c_str());
    return false;
  }

  return true;
}

bool Session::save_line(const string &var) {
  if (!Parson::valid_tag(var))
    return false;
  auto vari = linevar.find(var);
  if (vari == linevar.end())
    return false;

  string tmpfn = who->home_dir() + "/" + var + ".line.tmp";
  FILE *fp = fopen(tmpfn.c_str(), "w");
  if (!fp)
    return false;

  const strvec &linevec = vari->second;
  std::string linestr = moretpenc(linevec, '\t');
  size_t written = fwrite(linestr.data(), 1, linestr.length(), fp);
  if (written != linestr.length()) {
    fclose(fp);
    (void) ::unlink(tmpfn.c_str());
    return false;
  }

  (void) fclose(fp);

  string fn = who->home_dir() + "/" + var + ".line";
  int ret = ::rename(tmpfn.c_str(), fn.c_str());
  if (ret != 0) {
    (void) ::unlink(tmpfn.c_str());
    return false;
  }

  return true;
}


bool Session::save_word(const string &var) {
  if (!Parson::valid_tag(var))
    return false;
  auto vari = wordvar.find(var);
  if (vari == wordvar.end())
    return false;

  string tmpfn = who->home_dir() + "/" + var + ".word.tmp";
  FILE *fp = fopen(tmpfn.c_str(), "w");
  if (!fp)
    return false;

  const string &wordstr = vari->second;
  size_t written = fwrite(wordstr.data(), 1, wordstr.length(), fp);
  if (written != wordstr.length()) {
    fclose(fp);
    (void) ::unlink(tmpfn.c_str());
    return false;
  }

  (void) fclose(fp);

  string fn = who->home_dir() + "/" + var + ".word";
  int ret = ::rename(tmpfn.c_str(), fn.c_str());
  if (ret != 0) {
    (void) ::unlink(tmpfn.c_str());
    return false;
  }

  return true;
}

bool Session::load_word(const string &var) {
  if (!Parson::valid_tag(var))
    return false;

  string fn = who->home_dir() + "/" + var + ".word";
  FILE *fp = fopen(fn.c_str(), "r");
  if (!fp)
    return false;

  string wordstr;
  while (1) {
    char buf[1024];
    size_t nread = fread(buf, 1, sizeof(buf), fp);
    if (nread > 0)
      wordstr += string(buf, nread);
    if (nread != sizeof(buf)) {
      if (!feof(fp)) {
        fclose(fp);
        return false;
      }

      break;
    }
  }
  (void) fclose(fp);

  wordvar[var] = wordstr;
  return true;
}


}
