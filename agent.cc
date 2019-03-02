#define __MAKEMORE_AGENT_CC__ 1

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>

#include "agent.hh"
#include "server.hh"
#include "urbite.hh"

namespace makemore {

using namespace std;

static string _ipstr(uint32_t ip) {
  char buf[INET_ADDRSTRLEN];
  const char *retbuf = inet_ntop(AF_INET, &ip, buf, INET_ADDRSTRLEN);
  assert(retbuf == buf);
  return string(buf);
}

Agent::Agent(class Server *_server, const char *nom, int _s, uint32_t _ip) {
  server = _server;
  s = _s;
  ip = _ip;
  ipstr = _ipstr(ip);

  inbufj = 0;
  inbufk = 0;
  inbufn = 0;
  inbufm = 65536;
  inbuf = NULL; // new char[inbufm];

  outbufn = 0;
  outbufm = 65536;
  outbuf = NULL; // new char[outbufm];

  who = new Urbite(nom ? nom : ipstr, server->urb);
}

Agent::~Agent() {
  if (inbuf)
    delete[] inbuf;
  if (outbuf)
    delete[] outbuf;

  if (s >= 0)
    ::close(s);

  if (who)
    delete who;
}

void Agent::close() {
  if (s >= 0)
    ::close(s);
  s = -1;
}

bool Agent::slurp() {
  assert(inbufn <= inbufm);
  if (inbufn == inbufm)
    return false;

  if (!inbuf)
    inbuf = new char[inbufm];

  ssize_t ret = ::read(s, inbuf + inbufn, inbufm - inbufn);
  if (ret < 1)
    return false;

  inbufn += ret;
  assert(inbufn <= inbufm);
  return true;
}

void Agent::parse(vector<vector<string> > *lines) {
  unsigned int inbufi = 0;
  lines->clear();

  while (1) {
    vector<string> words;
    bool got_words = false;

    if (inbufk == 0) {
      assert(inbufj <= inbufn);
      const char *x = inbuf + inbufi;
      const char *p = inbuf + inbufj;
      const char *q = inbuf + inbufn;
      while (p < q && *p != '\n')
        ++p;

      inbufj = p - inbuf;
      if (p == q) {
        if (inbufi > 0) {
          memmove(inbuf, inbuf + inbufi, inbufn - inbufi);
          inbufj -= inbufi;
          inbufn -= inbufi;
        }
        return;
      }

      assert(*p == '\n');
      splitwords(string(x, p - x), &words);
      got_words = true;

      unsigned long extra = 0;
      for (auto word : words)
        if (*word.c_str() == '<')
          extra += strtoul(word.c_str() + 1, NULL, 0);
      inbufk = extra;
    }

    assert(inbufj <= inbufn);
    long remaining = inbufn - inbufj;
    if (remaining < inbufk) {
      inbufj = inbufn;
      inbufk -= remaining;

      if (inbufi > 0) {
        memmove(inbuf, inbuf + inbufi, inbufn - inbufi);
        inbufj -= inbufi;
        inbufn -= inbufi;
      }
      return;
    }

    inbufj += inbufk;
    inbufk = 0;

    const char *x = inbuf + inbufi;
    const char *p = x;
    const char *q = inbuf + inbufn;
    while (*p != '\n' && p < q)
      ++p;
    assert(*p == '\n');
    assert(p < q);
    assert(p <= inbuf + inbufj);

    if (!got_words)
      splitwords(string(x, p - x), &words);

    ++inbufj;
    ++p;

    unsigned int off = 0;
    for (unsigned int wi = 0, wn = words.size(); wi < wn; ++wi) {
      const std::string &word = words[wi];
      if (*word.c_str() == '<') {
        unsigned int len = strtoul(word.c_str() + 1, NULL, 0);
        words[wi] = string(p + off, len);
        off += len;
      }
    }
    assert(p + off == inbuf + inbufj);

    lines->push_back(words);

//fprintf(stderr, "inbufi=%u inbufj=%u inbufk=%u inbufn=%u line=[%s]\n",
//inbufi,inbufj,inbufk,inbufn, line.c_str());

    inbufi = inbufj;
  }
}

#if 0
void Agent::parse(vector<string> *lines) {
  lines->clear();
  if (inbufn == 0)
    return;
  if (!inbuf)
    inbuf = new char[inbufm];

  const char *p = inbuf + inbufn - 1;
  while (p > inbuf && *p != '\n')
    --p;
  if (p == inbuf) {
    if (*p == '\n') {
      memmove(inbuf, inbuf + 1, inbufn - 1);
      --inbufn;
    }
    return;
  }

  assert(*p == '\n' && p > inbuf);
  splitlines(string(inbuf, p - inbuf), lines);

  ++p;
  unsigned int k = p - inbuf;
  assert(k <= inbufn);
  memmove(inbuf, p, inbufn - k);
  inbufn -= k;
}
#endif

void Agent::printf(const char *fmt, ...) {
  char buf[65536];

  va_list ap;
  va_start(ap, fmt);

  (void) vsnprintf(buf, sizeof(buf) - 1, fmt, ap);

  write(string(buf));
}


void Agent::write(const string &str) {
  if (outbufn || s < 0) {
    assert(outbufn < outbufm);
    unsigned int l = outbufm - outbufn;
    unsigned int k = str.length();
    if (k > l)
      k = l;
    
    if (!outbuf)
      outbuf = new char[outbufm];
    memcpy(outbuf + outbufn, str.data(), k);
    outbufn += k;
    return;
  }

  ssize_t ret = ::write(s, str.data(), str.length());
  if (ret == str.length())
    return;
  if (ret < 1) {
    unsigned int l = outbufm;
    unsigned int k = str.length();
    if (k > l)
      k = l;

    if (ret < 1) {
      if (!outbuf)
        outbuf = new char[outbufm];
      memcpy(outbuf, str.data(), k);
      outbufn = k;
      return;
    }
  }

  assert(ret < str.length());
  {
    unsigned int l = outbufm;
    unsigned int k = str.length() - ret;
    if (k > l)
      k = l;

    if (!outbuf)
      outbuf = new char[outbufm];
    memcpy(outbuf, str.data() + ret, k);
    outbufn = k;
  }
}

void Agent::flush() {
  assert(s >= 0);

  if (outbufn == 0)
    return;
  if (!outbuf)
    outbuf = new char[outbufm];
  ssize_t ret = ::write(s, outbuf, outbufn);
  if (ret < 1)
    return;
  if (ret == outbufn) {
    outbufn = 0;
    return;
  }

  assert(ret < outbufn);
  memmove(outbuf, outbuf + ret, outbufn - ret);
  outbufn -= ret;
}


static void cleanwords(vector<string> *words)  {
  char buf[64];

  for (auto i = words->begin(); i != words->end(); ++i) {
    const std::string &word = *i;

    if (word.length() < 64) {
      const char *wordp = word.c_str();
      bool has_nonword = false;
      for (unsigned int i = 0; wordp[i]; ++i) {
        if (isspace(wordp[i]) || !isprint(wordp[i])) {
          has_nonword = true;
          break;
        }
      }
      if (!has_nonword)
        continue;
    }

    sprintf(buf, "!<%lu", word.length());
    *i = string(buf);
  }
}

void Agent::command(const vector<string> &words) {
  if (!words.size())
    return;

  vector<vector<string> > thread;
  splitthread(words, &thread, "|");
  assert(thread.size());
  if (!thread[0].size())
    return;

  string cmd = thread[0][0];
  auto hi = server->cmdtab.find(cmd);

  if (hi == server->cmdtab.end()) {
    fprintf(stderr, "unknown cmd %s, asking brane [%s]\n", cmd.c_str(), joinwords(words).c_str());

    vector<vector<string> > out;
    server->urb->brane1->ask(who->parson(), thread, &out);

    for (auto outi = out.begin(); outi != out.end(); ++outi) {
      this->write(joinwords(*outi) + "\n");
    }
  } else {
    vector<string> cwords = thread[0];
    cleanwords(&cwords);
    {
      string logstr;
#if 0
      logstr += "log ";
      logstr += ipstr + " ";
#endif
      logstr += join(cwords, " ");
      server->notify(who->nom, logstr, this);
    }

    vector<string> arg;
    arg.resize(thread[0].size() - 1);
    for (unsigned int argi = 0, argn = thread[0].size() - 1; argi < argn; ++argi)
      arg[argi] = thread[0][argi + 1];

    vector<vector<string> > ctx;
    ctx.resize(thread.size() - 1);
    for (unsigned int ctxi = 0, ctxn = thread.size() - 1; ctxi < ctxn; ++ctxi)
      ctx[ctxi] = thread[ctxi + 1];

    Server::Handler h = hi->second;
    assert(h);

    (void) h(this, ctx, cmd, arg);
  }
}

}
