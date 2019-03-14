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

#include <openssl/ssl.h>

#include "agent.hh"
#include "server.hh"
#include "urbite.hh"
#include "strutils.hh"
#include "process.hh"

namespace makemore {

using namespace std;

static string _ipstr(uint32_t ip) {
  char buf[INET_ADDRSTRLEN];
  const char *retbuf = inet_ntop(AF_INET, &ip, buf, INET_ADDRSTRLEN);
  assert(retbuf == buf);
  return string(buf);
}

Agent::Agent(class Server *_server, int _s, uint32_t _ip, bool _secure) {
  server = _server;
  s = _s;
  ip = _ip;
  ipstr = _ipstr(ip);
  secure = _secure;

  inbufj = 0;
  inbufk = 0;
  inbufn = 0;
  inbufm = (1 << 20);
  inbuf = NULL; // new char[inbufm];

  outbufn = 0;
  outbufm = (1 << 20);
  outbuf = NULL; // new char[outbufm];

  if (secure) {
    ssl = SSL_new(_server->ssl_ctx);
    SSL_set_fd(ssl, s);
    ssl_status = SSL_ERROR_WANT_READ;
  } else {
    ssl = NULL;
    ssl_status = 0;
  }

  proto = UNKNOWN;
  httpkeep = true;

  session = new Session(this);
}

Agent::~Agent() {
  if (inbuf)
    delete[] inbuf;
  if (outbuf)
    delete[] outbuf;

  this->close();

  if (ssl)
    SSL_free(ssl);

  delete session;
}

void Agent::close() {
  if (s >= 0)
    ::close(s);
  s = -1;
}

bool Agent::slurp() {
  if (ssl_status) {
    ssl_status = SSL_get_error(ssl, SSL_accept(ssl));
    return true;
  }

  assert(inbufn <= inbufm);
  if (inbufn == inbufm)
    return false;

  if (!inbuf)
    inbuf = new char[inbufm];

  ssize_t ret;
  if (secure) {
    ret = ::SSL_read(ssl, inbuf + inbufn, inbufm - inbufn);
  } else {
    ret = ::read(s, inbuf + inbufn, inbufm - inbufn);
  }
  if (ret < 1)
    return false;

  inbufn += ret;
  assert(inbufn <= inbufm);
  return true;
}

void Agent::parse() {
  unsigned int inbufi = 0;

  while (1) {
    strvec words;
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

      ++inbufj;
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
    assert(p < inbuf + inbufj);

    if (!got_words)
      splitwords(string(x, p - x), &words);

    ++p;

    unsigned int off = 0;
    for (unsigned int wi = 0, wn = words.size(); wi < wn; ++wi) {
      const std::string &word = words[wi];
fprintf(stderr, "wordlen=%u wi=%u\n", word.length(), wi);
      if (*word.c_str() == '<') {
        unsigned int len = strtoul(word.c_str() + 1, NULL, 0);
fprintf(stderr, "len=%u\n", len);
        words[wi] = string(p + off, len);
        off += len;
      }
    }
    assert(p + off == inbuf + inbufj);

for (auto w : words ) { fprintf(stderr, "word=[%u]%s\n", w.length(), w.c_str()); }

    linebuf.push_back(words);

//fprintf(stderr, "inbufi=%u inbufj=%u inbufk=%u inbufn=%u line=[%s]\n",
//inbufi,inbufj,inbufk,inbufn, line.c_str());

    inbufi = inbufj;
    got_words = false;
  }
}

#if 0
void Agent::parse(strvec *lines) {
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

bool Agent::write(const strvec &words) {
  std::string extra = "";
  strvec nwords = words;

  for (auto wordi = nwords.begin(); wordi != nwords.end(); ++wordi) {
    string &word = *wordi;
    unsigned int wordn = word.length();
//fprintf(stderr, "wordn=%u\n", wordn);
    if (wordn == 0 || wordn > 255 || word[0] == '<' || hasspace(word) || hasnull(word)) {
      extra += word;
      char buf[64];
      sprintf(buf, "<%u", wordn);
      word = buf;
    }
  }

  return write(joinwords(nwords) + "\n" + extra);
}

bool Agent::write(const string &str) {
  if (s < 0)
    return false;
  if (str.length() + outbufn > outbufm)
    return false;

  if (outbufn) {
    assert(outbufn <= outbufm);
    unsigned int l = outbufm - outbufn;
    unsigned int k = str.length();
    assert(k <= l);
    
    if (!outbuf)
      outbuf = new char[outbufm];
    memcpy(outbuf + outbufn, str.data(), k);
    outbufn += k;
    return true;
  }

  ssize_t ret;
  if (secure) {
    ret = ::SSL_write(ssl, str.data(), str.length());
  } else {
    ret = ::write(s, str.data(), str.length());
  }
  if (ret == str.length())
    return true;
  if (ret < 1) {
    unsigned int k = str.length();
    assert(k <= outbufm);

    if (!outbuf)
      outbuf = new char[outbufm];
    memcpy(outbuf, str.data(), k);
    outbufn = k;
    return true;
  }

  assert(ret > 0);
  assert(ret < str.length());
  {
    unsigned int k = str.length() - ret;
    assert(k <= outbufm);

    if (!outbuf)
      outbuf = new char[outbufm];
    memcpy(outbuf, str.data() + ret, k);
    outbufn = k;
  }
  return true;
}

void Agent::flush() {
  assert(s >= 0);
  if (ssl_status) {
    ssl_status = SSL_get_error(ssl, SSL_accept(ssl));
    return;
  }

  if (outbufn == 0) {
    IO *shell_out = session->shell->out;

    while (shell_out->can_get()) {
      strvec *vec = shell_out->peek();
      assert(vec);

      if (!this->write(*vec)) {
        break;
      }

      strvec *ret = shell_out->get();
      assert(ret);
    }

    return;
  }

  assert(outbufn > 0);
  assert(outbuf);

  ssize_t ret;
  if (secure) {
    ret = ::SSL_write(ssl, outbuf, outbufn);
  } else {
    ret = ::write(s, outbuf, outbufn);
  }

  if (ret < 1)
    return;
  if (ret < outbufn) {
    memmove(outbuf, outbuf + ret, outbufn - ret);
    outbufn -= ret;
    return;
  }

  assert(ret == outbufn);
  outbufn = 0;


  IO *shell_out = session->shell->out;
  while (shell_out->can_get()) {
    strvec *vec = shell_out->peek();
    assert(vec);

    if (!this->write(*vec)) {
      break;
    }

    strvec *ret = shell_out->get();
    assert(ret);
  }
}


static void cleanwords(strvec *words)  {
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

void Agent::command(const strvec &cmd) {
  if (cmd.empty())
    return;

  Process *shell = session->shell;
  assert(shell);
  assert(shell->in->can_put());
  shell->in->put(cmd);
}

#if 0
void Agent::command(const strvec &words) {
  if (!words.size())
    return;

  bool multi = false;
  for (auto word : words) {
    if (word == ";") {
      multi = true;
      break;
    }
  }
  if (multi) {
    vector<strvec> states;
    splitthread(words, &states, ";");
    for (auto state : states) {
      this->command(state);
    }
    return;
  }

  vector<strvec> thread;
  splitthread(words, &thread, "|");
  assert(thread.size());
  if (!thread[0].size())
    return;

  string cmd = thread[0][0];
  auto hi = server->cmdtab.find(cmd);

  fprintf(stderr, "got cmd %s [%s]\n", cmd.c_str(), joinwords(words).c_str());
  if (hi == server->cmdtab.end()) {
    fprintf(stderr, "unknown cmd %s, asking brane [%s]\n", cmd.c_str(), joinwords(words).c_str());

    vector<strvec> out;
    server->urb->brane1->ask(who->parson(), thread, &out);

    for (auto outi = out.begin(); outi != out.end(); ++outi) {
      this->write(joinwords(*outi) + "\n");
    }
  } else {
    strvec cwords = thread[0];
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

    strvec arg;
    arg.resize(thread[0].size() - 1);
    for (unsigned int argi = 0, argn = thread[0].size() - 1; argi < argn; ++argi)
      arg[argi] = thread[0][argi + 1];

    vector<strvec> ctx;
    ctx.resize(thread.size() - 1);
    for (unsigned int ctxi = 0, ctxn = thread.size() - 1; ctxi < ctxn; ++ctxi)
      ctx[ctxi] = thread[ctxi + 1];

    Server::Handler h = hi->second;
    assert(h);

    (void) h(this, ctx, cmd, arg);
  }
}
#endif

}
