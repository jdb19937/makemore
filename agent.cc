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
#include "encgen.hh"
#include "urbite.hh"
#include "numutils.hh"
#include "strutils.hh"
#include "imgutils.hh"
#include "process.hh"
#include "warp.hh"
#include "pose.hh"
#include "partrait.hh"

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
  active = time(NULL);

  inbufj = 0;
  inbufk = 0;
  inbufn = 0;
  inbufm = (1 << 20);
  inbuf = NULL; // new char[inbufm];

  outbufn = 0;
  outbufm = (10 << 20);
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
      if (*word.c_str() == '<') {
        unsigned int len = strtoul(word.c_str() + 1, NULL, 0);
        words[wi] = string(p + off, len);
        off += len;
      }
    }
    assert(p + off == inbuf + inbufj);

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

bool Agent::write(const Line &wv) {
  strvec sv;
  line_to_strvec(wv, &sv);
  return write(sv);
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
fprintf(stderr, "here ret=%ld len=%ld\n", ret, str.length());
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
    assert(session->shell->otab.size() >= 1);
    IO *shell_out = session->shell->otab[0];
    assert(shell_out);

    while (shell_out->can_get()) {
      Line *vec = shell_out->peek();
      assert(vec);

      if (!this->write(*vec)) {
        break;
      }

      Line *ret = shell_out->get();
      assert(ret == vec);
      delete vec;
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


  IO *shell_out = session->shell->otab[0];
  while (shell_out->can_get()) {
    Line *vec = shell_out->peek();
    assert(vec);

    if (!this->write(*vec)) {
      break;
    }

    Line *ret = shell_out->get();
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
  assert(shell->itab[0]->can_put());

  unsigned int wn = cmd.size();
  Line *wp = new Line(wn);
  Line &wr = *wp;
  for (unsigned int wi = 0; wi < wn; ++wi)
    wr[wi] = cmd[wi];
  shell->itab[0]->put(wp);
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

static std::string htmlp(const std::string &nom, unsigned int dim) {
  char buf[4096];
  sprintf(buf, 
//    "<a href='/%s'><img width='%u' height='%u' style='image-rendering: pixelated' src='/%s.png'/></a>",
    "<a href='/%s'><img width='%u' height='%u' src='/%s.png'/></a>",
    nom.c_str(), dim, dim, nom.c_str()
  );
  return buf;
}

void Agent::handle_http() {
  std::string req = httpbuf[0];
  strvec reqwords;
  splitwords(req, &reqwords);

  std::string host;

  for (auto kv : httpbuf) {
    if (strbegins(kv, "Cookie: ")) {
      std::string v = kv.c_str() + 8;
    } else if (strbegins(kv, "Host: ")) {
      std::string v = kv.c_str() + 6;
      host = v;
    }
  }

fprintf(stderr, "host=[%s]\n", host.c_str());

fprintf(stderr, "req=[%s]\n", req.c_str());
  if (reqwords[0] != "GET") {
    this->close();
    return;
  }

  std::string path = reqwords[1];
  if (path[0] != '/') {
    this->close();
    return;
  }

  if (host != "peaple.io") {
    this->printf("HTTP/1.1 302 Redirect\r\n");
    this->printf("Connection: close\r\n");
    this->printf("Location: https://peaple.io/\r\n\r\n");
    this->close();
    return;
  }

  std::string query;
  if (const char *p = strchr(path.c_str(), '?')) {
    query = p + 1;
    path = std::string(path.c_str(), p - path.c_str());
  }

#if 0
  if (path == "/") {
    std::string fn = server->urb->dir + "/index.html";
    std::string html = makemore::slurp(fn);
    
    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: text/html\r\n");
    this->printf("Content-Length: %lu\r\n", html.length());
    this->printf("\r\n");
    this->write(html);
    return;
  }
#endif

  if (path == "/test.png" || path == "/mandelbrot.png") {
    std::string png = makemore::slurp(std::string(".") + path);

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: image/png\r\n");
    this->printf("Content-Length: %lu\r\n", png.length());
    this->printf("\r\n");
    this->write(png);
    return;
  }

  if (path == "/pic.png") {
    int which = atoi(query.c_str());

    if (which < 0) {
      this->printf("HTTP/1.1 404 Not Found\r\n");
      this->printf("Connection: keep-alive\r\n");
      this->printf("Content-Type: text/plain\r\n");
      this->printf("Content-Length: 0\r\n");
      this->printf("\r\n");
      return;
    }

    which %= server->urb->srcimages.size();
    string imagefn = server->urb->srcimages[which];
    string png = makemore::slurp(imagefn);

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: image/png\r\n");
    this->printf("Content-Length: %lu\r\n", png.length());
    this->printf("\r\n");
    this->write(png);
    return;
  }

  if (path == "/set_tags.txt") {
    map<string, string> cgi;
    cgiparse(query, &cgi);

    int which = -1;
    set<string> seen;
    strvec new_tags;
    for (auto kv : cgi) {
      const std::string &k = kv.first;
      if (seen.count(k))
        continue;
      seen.insert(k);

      const string &v = kv.second;
      if (k == "i") {
        which = atoi(v.c_str());
        continue;
      }

      if (k[0] == '#') {
        string nk = k.c_str() + 1;
        if (v == "1") {
          new_tags.push_back(nk);
        } else {
          new_tags.push_back(nk + ":" + v);
        }
      }
    }

    if (which < 0) {
      this->printf("HTTP/1.1 404 Not Found\r\n");
      this->printf("Connection: keep-alive\r\n");
      this->printf("Content-Type: text/plain\r\n");
      this->printf("Content-Length: 0\r\n");
      this->printf("\r\n");
      return;
    }

    which %= server->urb->srcimages.size();
    string imagefn = server->urb->srcimages[which];
    string png = makemore::slurp(imagefn);

    unsigned int w, h;
    uint8_t *tmp;
    pngrgb(png, &w, &h, &tmp);
    png = "";
    rgbpng(tmp, w, h, &png, &new_tags);
    new_tags.clear();
    pngrgb(png, w, h, tmp, &new_tags);
    delete[] tmp;

    std::string txt;
    for (auto tag : new_tags) {
      txt += std::string("#") + tag + "\n";
    }
    makemore::spit(png, imagefn);

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: text/plain\r\n");
    this->printf("Content-Length: %lu\r\n", txt.length());
    this->printf("\r\n");
    this->write(txt);
    return;
  }

  if (path == "/get_tags.txt") {
    int which = atoi(query.c_str());
    if (which < 0) {
      this->printf("HTTP/1.1 404 Not Found\r\n");
      this->printf("Connection: keep-alive\r\n");
      this->printf("Content-Type: text/plain\r\n");
      this->printf("Content-Length: 0\r\n");
      this->printf("\r\n");
      return;
    }

    which %= server->urb->srcimages.size();
    string imagefn = server->urb->srcimages[which];

    Partrait prt;
    prt.load(imagefn);

    std::string txt;
    for (auto tag : prt.tags) {
      txt += std::string("#") + tag + "\n";
    }

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: text/plain\r\n");
    this->printf("Content-Length: %lu\r\n", txt.length());
    this->printf("\r\n");
    this->write(txt);
    return;
  }

#if 0
  if (path == "/tagenc.png") {
    unsigned int which = atoi(query.c_str());
    string imagefn = server->urb->images[which % server->urb->images.size()];
    string png0 = makemore::slurp(imagefn);
    uint8_t tmp[64 * 64 * 3];

    vector<string> tags;
    pnglab(png0, 64, 64, tmp, &tags);
    labdequant(tmp, 64 * 64 * 3, server->urb->egd->tgtbuf);

    server->urb->egd->encode();
    server->urb->egd->generate();

    std::string png;
    labquant(server->urb->egd->tgtbuf, 64 * 64 * 3, tmp);
    labpng(tmp, Parson::dim, Parson::dim, &png);

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: image/png\r\n");
    this->printf("Content-Length: %lu\r\n", png.length());
    this->printf("\r\n");
    this->write(png);
    return;
  }
#endif

  if (path == "/tagraw.png") {
    unsigned int which = atoi(query.c_str());
    string imagefn = server->urb->images[which % server->urb->images.size()];
    string png = makemore::slurp(imagefn);

    // pnglab(png, 64, 64, parson.target, &tags);
    // for (auto tag : tags) {
    //   parson.add_tag(tag.c_str());
    // }

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: image/png\r\n");
    this->printf("Content-Length: %lu\r\n", png.length());
    this->printf("\r\n");
    this->write(png);
    return;
  }

  if (path == "/tagger.html" || path == "/cam.html" || path == "/edit.html" || path == "/memory.html") {
    std::string html = makemore::slurp(path.c_str() + 1);

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: text/html\r\n");
    this->printf("Content-Length: %lu\r\n", html.length());
    this->printf("\r\n");
    this->write(html);
    return;
  }

  if (path == "/favicon.ico") {
    std::string fn = server->urb->dir + "/favicon.ico";
    std::string favicon = makemore::slurp(fn);
    
    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: image/x-icon\r\n");
    this->printf("Content-Length: %lu\r\n", favicon.length());
    this->printf("\r\n");
    this->write(favicon);
    return;
  }


  if (path == "/edit") {
    map<string, string> cgi;
    cgiparse(query, &cgi);
    string nom = cgi["nom"];
    string op = cgi["op"];

    if (op == "loadgens") {
      server->urb->enc->load();
      for (auto g : server->urb->gens) {
        g.second->load();
      }
      string txt = "ok, reloaded gens";

      this->printf("HTTP/1.1 200 OK\r\n");
      this->printf("Connection: keep-alive\r\n");
      this->printf("Content-Type: text/plain\r\n");
      this->printf("Content-Length: %lu\r\n", txt.length());
      this->printf("\r\n");
      this->write(txt);
      return;
    }

    if (!Parson::valid_nom(nom)) {
      this->printf("HTTP/1.1 404 Not Found\r\n");
      this->printf("Connection: keep-alive\r\n");
      this->printf("Content-Type: text/plain\r\n");
      this->printf("Content-Length: 0\r\n");
      this->printf("\r\n");
      return;
    }


    Parson *parson = server->urb->find(nom);
    if (!parson) {
      this->printf("HTTP/1.1 404 Not Found\r\n");
      this->printf("Connection: keep-alive\r\n");
      this->printf("Content-Type: text/plain\r\n");
      this->printf("Content-Length: 0\r\n");
      this->printf("\r\n");
      return;
    }

    
    string txt = "huh";
    if (op == "scramble") {
      strcpy(parson->srcfn, "");
      for (unsigned int j = 0; j < Parson::ncontrols; ++j)
        parson->controls[j] = randgauss();
      txt = "ok, scrambled " + nom;
    } else if (op == "pickreal") {
      Supergen *gen = server->urb->get_gen(parson->gen);
      Zone *sks = gen->zone;
     
      Parson *skp = sks->pick();
      assert(skp);

      std::string srcfn = skp->srcfn;
      assert(srcfn.length());
      strcpy(parson->srcfn, skp->srcfn);

      Partrait prt;
      prt.load(srcfn);
      memset(parson->tags, 0, sizeof(parson->tags));

      Styler *sty = server->urb->get_sty(parson->sty);
      assert(sty);
//      server->urb->enc->encode(prt, parson, sty);

      txt = "ok, pickreal " + nom;
    } else if (op == "resketch") {
      std::string skstag = cgi["sks"];
      if (skstag.length() && Parson::valid_tag(skstag.c_str()))
        strcpy(parson->sks, skstag.c_str());

      Zone *sks = server->urb->sks0;
      if (!strcmp(parson->sks, "shampane"))
        sks = server->urb->sks1;

      Parson *skp = sks->pick();
      assert(skp);

      parson->skid = sks->dom(skp);
      memcpy(parson->sketch, skp->sketch, sizeof(parson->sketch));
      txt = "ok, resketched " + nom;
    } else if (op == "add_tag") {
      std::string tag = cgi["tag"];
      parson->add_tag(tag.c_str());
      txt = "ok, added tag " + tag;
    } else if (op == "del_tag") {
      std::string tag = cgi["tag"];
      parson->del_tag(tag.c_str());
      txt = "ok, deleted tag " + tag;
    } else if (op == "set_gen") {
      std::string tag = cgi["gen"];
      if (Parson::valid_tag(tag.c_str())) {
        strcpy(parson->gen, tag.c_str());
        txt = "ok, set gen " + tag;
      }
    } else if (op == "set_sty") {
      std::string tag = cgi["sty"];
      if (Parson::valid_tag(tag.c_str())) {
        strcpy(parson->sty, tag.c_str());
        txt = "ok, set sty " + tag;
      }
    }

    this->printf("HTTP/1.1 200 Ok\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: text/plain\r\n");
    this->printf("Content-Length: %lu\r\n", txt.length());
    this->printf("\r\n");
    this->write(txt);
    return;
  }



  std::string nom;
  if (path == "/") {
    nom = Parson::gen_nom();
    this->printf("HTTP/1.1 302 Redirect\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: text/plain\r\n");
    this->printf("Content-Length: 0\r\n");
    this->printf("Location: /%s\r\n", nom.c_str());
    this->printf("\r\n", nom.c_str());
    return;
  }

  nom = path.c_str() + 1;
  std::string ext;
  if (const char *p = strchr(nom.c_str(), '.')) {
    ext = p + 1;
    nom = std::string(nom.c_str(), p - nom.c_str());
  }

  std::string func = "html";
  if (const char *p = strchr(nom.c_str(), '/')) {
    func = p + 1;
    nom = std::string(nom.c_str(), p - nom.c_str());
  }

//fprintf(stderr, "nom=[%s] is_png=%d\n", 

  if (!Parson::valid_nom(nom)) {
    this->printf("HTTP/1.1 404 Not Found\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: text/plain\r\n");
    this->printf("Content-Length: 0\r\n");
    this->printf("\r\n");
    return;
  }

  if (func == "edit") {
    std::string html = makemore::slurp("edit.html");

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: text/html\r\n");
    this->printf("Content-Length: %lu\r\n", html.length());
    this->printf("\r\n");
    this->write(html);
    return;
  }

  if (ext == "json") {
    Parson *parson = server->urb->make(nom, 0);
    assert(parson);

    std::string json;
    json += "{\n";
    json += string("  \"nom\": \"") + parson->nom + "\",\n";
    json += string("  \"gen\": \"") + parson->gen + "\",\n";
    json += string("  \"sty\": \"") + parson->sty + "\",\n";
    json += string("  \"sks\": \"") + parson->sks + "\",\n";
    json += string("  \"srcfn\": \"") + parson->srcfn + "\",\n";

    char numstr[256];
    sprintf(numstr, "%u", parson->skid);
    json += string("  \"skid\": ") + numstr + ",\n";

    json += string("  \"parens\": [");
    for (unsigned int j = 0; j < 2; ++j) {
      const char *paren = parson->parens[j];
      if (!*paren)
        continue;
      if (j)
        json += ", ";
      json += string("\"") + paren + string("\"");
    }
    json += "],\n";

    json += string("  \"frens\": [");
    for (unsigned int j = 0; j < 2; ++j) {
      const char *fren = parson->frens[j];
      if (!*fren)
        continue;
      if (j)
        json += ", ";
      json += string("\"") + fren + string("\"");
    }
    json += "],\n";

    json += string("  \"tags\": [");
    for (unsigned int j = 0; j < Parson::ntags; ++j) {
      const char *tag = parson->tags[j];
      if (!*tag)
        continue;

      if (j)
        json += ", ";
      json += string("\"") + tag + string("\"");
    }
    json += "],\n";

    sprintf(numstr, "%lf", parson->angle);
    json += string("  \"angle\": ") + numstr + ",\n";
    sprintf(numstr, "%lf", parson->stretch);
    json += string("  \"stretch\": ") + numstr + ",\n";
    sprintf(numstr, "%lf", parson->skew);
    json += string("  \"skew\": ") + numstr + "\n";

    json += "}\n";

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: text/json\r\n");
    this->printf("Content-Length: %lu\r\n", json.length());
    this->printf("\r\n");
    this->write(json);
    return;
  }

  if (ext == "png") {
    Parson *parson = server->urb->make(nom, 0);

#if 0
    Pose pose = Pose::STANDARD;
    pose.angle = parson->angle * 0.5;
    pose.stretch = (parson->stretch - 1.0) * 0.5 + 1.0;
    pose.skew = parson->skew * 0.25;
#endif

//fprintf(stderr, "[%lf %lf %lf]\n", pose.angle, pose.stretch, pose.skew);

    Superenc *enc = server->urb->enc;
    Supergen *gen = server->urb->get_gen(parson->gen);
//gen->load();
    Styler *sty = server->urb->get_sty(parson->sty);

    Partrait showpar(256, 256);

double *tmp = new double[Parson::ncontrols];
sty->generate(*parson, tmp);
gen->generate(tmp, &showpar);
delete[] tmp;

#if 0
    gen->generate(*parson, &showpar, sty);

assert(showpar.alpha);
assert(showpar.w == 256 && showpar.h == 256);
for (unsigned int j = 0; j < 256 * 256; ++j) {
  double a = (double)showpar.alpha[j] / 255.0;
  // a = (a - 0.2) / 0.6;
  // a = (a - 0.5) / 0.01;
  if (a > 1.0) a = 1.0;
  if (a < 0.0) a = 0.0;

  if (a < 0.3) a = 0.3;
  showpar.alpha[j] = (uint8_t)(a * 255.0);
}
#endif

    std::string png;
    showpar.to_png(&png);

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: image/png\r\n");
    this->printf("Content-Length: %lu\r\n", png.length());
    this->printf("\r\n");
    this->write(png);
    return;
  }

  if (ext == "sketch.png") {
    Parson *parson = server->urb->make(nom, 0);

    Partrait showpar(8, 8);
    labrgb(parson->sketch, 192, showpar.rgb);

    std::string png;
    showpar.to_png(&png);

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: image/png\r\n");
    this->printf("Content-Length: %lu\r\n", png.length());
    this->printf("\r\n");
    this->write(png);
    return;
  }


  {
    Parson *parson = server->urb->make(nom, 0, 2);
    assert(parson);

    Parson *paren0 = server->urb->make(parson->parens[0], 0);
fprintf(stderr, "paren0=%s\n", parson->parens[0]);
    assert(paren0);
    Parson *paren1 = server->urb->make(parson->parens[1], 0);
    assert(paren1);

    std::string htmlbuf;
    htmlbuf += std::string("<html><head>");
    htmlbuf += std::string("<title>") + nom + "</title>";

    htmlbuf += "<meta property='og:title' content='" + nom + "'>";
    htmlbuf += "<meta property='og:description' content='" + nom + "'>";
    htmlbuf += "<meta property='og:image' content='https://peaple.io/" + nom + ".png'>";
    htmlbuf += "<meta property='og:image:type' content='image/png'>";
    htmlbuf += "<meta property='og:image:width' content='64'>";
    htmlbuf += "<meta property='og:image:height' content='64'>";
    htmlbuf += "<meta property='og:url' content='https://peaple.io/" + nom + "'>";
    htmlbuf += "<meta property='og:type' content='article'>";
    htmlbuf += "<link rel='canonical' href='https://peaple.io/" + nom + "'/>";

    htmlbuf += "</head><body bgcolor='#000000'><center>";

    htmlbuf += htmlp(paren0->parens[0], 128);
    htmlbuf += htmlp(paren0->parens[1], 128);
    htmlbuf += htmlp(paren1->parens[0], 128);
    htmlbuf += htmlp(paren1->parens[1], 128);
    htmlbuf += "<br>";
    htmlbuf += htmlp(paren0->nom, 256);
    htmlbuf += htmlp(paren1->nom, 256);
    htmlbuf += "<br>";
    htmlbuf += htmlp(parson->nom, 512);
    htmlbuf += "</center></body></html>\n";

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: text/html\r\n");
    this->printf("Set-Cookie: foo=baz\r\n");
    this->printf("Content-Length: %u\r\n", htmlbuf.length());
    this->printf("\r\n");
    this->write(htmlbuf);

    return;
  }
}

}
