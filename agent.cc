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
#include <dirent.h>

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
#include "autocompleter.hh"
#include "mob.hh"
#include "fractals.hh"

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

fprintf(stderr, "agent command %s\n", cmd[0].c_str());

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

static std::string htmlpe(const std::string &nom, unsigned int dim) {
  char buf[4096];
  sprintf(buf, 
//    "<a href='/%s'><img width='%u' height='%u' style='image-rendering: pixelated' src='/%s.png'/></a>",
    "<a href='/%s/edit'><img width='%u' height='%u' src='/%s.png'/></a>",
    nom.c_str(), dim, dim, nom.c_str()
  );
  return buf;
}

void Agent::http_notfound() {
  this->printf("HTTP/1.1 404 Not Found\r\n");
  this->printf("Connection: keep-alive\r\n");
  this->printf("Content-Type: text/plain\r\n");
  this->printf("Content-Length: 0\r\n");
  this->printf("\r\n");
}

void Agent::http_denied() {
  this->printf("HTTP/1.1 403 Permission Denied\r\n");
  this->printf("Connection: keep-alive\r\n");
  this->printf("Content-Type: text/plain\r\n");
  this->printf("Content-Length: 0\r\n");
  this->printf("\r\n");
}

void Agent::handle_http() {
  std::string req = httpbuf[0];
  strvec reqwords;
  splitwords(req, &reqwords);

  std::string host;
  std::string cookie;

  for (auto kv : httpbuf) {
    if (strbegins(kv, "Cookie: ")) {
      std::string v = kv.c_str() + 8;
      cookie = v;
    } else if (strbegins(kv, "Host: ")) {
      std::string v = kv.c_str() + 6;
      host = v;
    }
  }

#if 0
  std::string usernom, session;
  if (cookie.length()) {
    std::vector<std::string> kv;
    split(cookie, ';', &kv);
    for (auto k : kv) {
      const char *p = k.c_str();
      while (isspace(*p))
        ++p;
      if (!strncmp(p, "usernom=", 8)) {
        usernom = p + 8;
      } else if (!strncmp(p, "session=", 8)) {
        session = p + 8;
      }
    }
  }
fprintf(stderr, "cookie=[%s] usernom=[%s] session=[%s]\n", cookie.c_str(), usernom.c_str(), session.c_str());
#endif



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

  if (path == "/julia.png") {
    map<string, string> cgi;
    cgiparse(query, &cgi);

    unsigned int r = randuint();
    if (cgi["r"] != "")
      r = strtoul(cgi["r"].c_str(), NULL, 0);

    double vdev = 0;
    if (cgi["vdev"] != "")
      vdev = strtod(cgi["vdev"].c_str(), NULL);
    vdev *= 0.75;

    double tone = 0.75;
    if (cgi["tone"] != "")
      tone = strtod(cgi["tone"].c_str(), NULL);

    seedrand(r);
    double ca = randgauss() * vdev;
    double cb = randgauss() * vdev;

    Partrait jprt(256, 256);
    julia(jprt.rgb, ca, cb);

    for (unsigned int j = 0; j < 256 * 256 * 3; ++j)
      jprt.rgb[j] *= tone;

    std::string png;
    jprt.to_png(&png);

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: image/png\r\n");
    this->printf("Content-Length: %lu\r\n", png.length());
    this->printf("\r\n");
    this->write(png);
    return;
  }
  if (path == "/mandelbrot.png") {
    map<string, string> cgi;
    cgiparse(query, &cgi);

    Partrait jprt(256, 256);
    mandelbrot(jprt.rgb);

    std::string png;
    jprt.to_png(&png);

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: image/png\r\n");
    this->printf("Content-Length: %lu\r\n", png.length());
    this->printf("\r\n");
    this->write(png);
    return;
  }
  if (path == "/burning_ship.png") {
    map<string, string> cgi;
    cgiparse(query, &cgi);

    unsigned int r = randuint();
    if (cgi["r"] != "")
      r = strtoul(cgi["r"].c_str(), NULL, 0);

    double vdev = 0.0;
    if (cgi["vdev"] != "")
      vdev = strtod(cgi["vdev"].c_str(), NULL);

    vdev /= 12.0;

    seedrand(r);
    double ra = randgauss() * vdev;
    double rb = randgauss() * vdev;

    Partrait jprt(256, 256);
    burnship(jprt.rgb, ra, rb);

    std::string png;
    jprt.to_png(&png);

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
      http_notfound();
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
      http_notfound();
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
      http_notfound();
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

  if (!strncmp(path.c_str(), "/images/", 8)) {
    std::set<std::string> valid;
    valid.insert("conf.png");
    valid.insert("close.png");
    valid.insert("clear.png");
    valid.insert("stop.png");
    valid.insert("go.png");
    valid.insert("makemore.png");
    valid.insert("mork.png");
    valid.insert("slashfam.png");
    valid.insert("createdby.png");
    valid.insert("frens.png");
    valid.insert("add_to_cart.png");
valid.insert("small_nom.png");
valid.insert("small_pass.png");
valid.insert("pass.png");
valid.insert("pass2.png");
valid.insert("login.png");
valid.insert("logout.png");
valid.insert("upload.png");
valid.insert("load.png");
valid.insert("save.png");
valid.insert("crew.png");
valid.insert("minions.png");
valid.insert("minions.png");
valid.insert("boss.png");
valid.insert("head.png");
valid.insert("encode.png");
valid.insert("enc.png");
valid.insert("active.png");
valid.insert("activity.png");
valid.insert("online.png");
valid.insert("created.png");
valid.insert("popular.png");
valid.insert("more.png");
valid.insert("more.png");
valid.insert("less.png");
valid.insert("plus.png");
valid.insert("map.png");
valid.insert("unmap.png");
valid.insert("minus.png");
valid.insert("spawn.png");
valid.insert("bread.png");
valid.insert("score.png");
valid.insert("owner.png");
valid.insert("owner.png");
valid.insert("comms.png");
valid.insert("script.png");
valid.insert("tribe.png");
valid.insert("prime.png");
valid.insert("claim.png");
valid.insert("induct.png");
valid.insert("switch.png");
valid.insert("mash.png");
valid.insert("blend.png");
valid.insert("goto.png");
valid.insert("link.png");

    valid.insert("wtf1.png");
    valid.insert("wtf2.png");
    valid.insert("wtf3.png");
    valid.insert("wtf4.png");
    valid.insert("wtf5.png");
    valid.insert("wtf6.png");
    valid.insert("wtf7.png");
    valid.insert("wtf8.png");

    valid.insert("rand.png");
    valid.insert("file.png");
    valid.insert("fam.png");
    valid.insert("frens.png");
    valid.insert("doc.png");
    valid.insert("sh.png");
    valid.insert("cam.png");
    valid.insert("get.png");
    valid.insert("mem.png");
    valid.insert("new.png");
    valid.insert("who.png");
    valid.insert("msg.png");
    valid.insert("top.png");
    valid.insert("mob.png");
    valid.insert("don.png");
    valid.insert("auto.png");
    valid.insert("p.png");
    valid.insert("q.png");
    valid.insert("r.png");
    valid.insert("send.png");
    valid.insert("buy.png");
    valid.insert("email.png");
    valid.insert("addfren.png");
    valid.insert("add.png");
    valid.insert("befren.png");
    valid.insert("dashgt.png");
    valid.insert("fren.png");
    valid.insert("pushnom.png");
    valid.insert("pushfam.png");
    valid.insert("capture.png");
    valid.insert("xform.png");
    valid.insert("blend.png");
    valid.insert("add_fren.png");
    valid.insert("gen_nom.png");

    std::string imagefn = path.c_str() + 8;
    if (!valid.count(imagefn)) {
      http_notfound();
    } else {
      string png = makemore::slurp(std::string("images/") + imagefn);

      this->printf("HTTP/1.1 200 OK\r\n");
      this->printf("Connection: keep-alive\r\n");
      this->printf("Content-Type: image/png\r\n");
      this->printf("Content-Length: %lu\r\n", png.length());
      this->printf("Cache-Control: public, max-age=86400\r\n");
      this->printf("\r\n");
      this->write(png);
    }
    return;
  }

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

  // if (path == "/tagger.html" || path == "/cam.html" || path == "/edit.html" || path == "/memory.html" || path == "/autocomplete.html" || path == "/terminal.html") {
  if (path == "/sh" || path == "/popular" || path == "/active" || path == "/conf" || path == "/buy" || path == "/who" || path == "/online" || path == "/top" || path == "/top/" || path == "/top/activity" || path == "/top/online" || path == "/top/popular" || path == "/top/score" || path == "/top/minions" || path == "/comms" || path == "/script") {
    strvec pathparts;
    split(path, '/', &pathparts);
    std::string html = makemore::slurp(pathparts[0] + ".html");

    std::string header = makemore::slurp("header.html");
    std::string url = "https://peaple.io" + path;
    std::string escurl = "https%3A%2F%2Fpeaple.io%2F" + std::string(path.c_str() + 1);

    html = replacestr(html, "$HEADER", header);
    html = replacestr(html, "$URL", url);
    html = replacestr(html, "$ESCURL", escurl);

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: text/html\r\n");
    this->printf("Content-Length: %lu\r\n", html.length());
    this->printf("\r\n");
    this->write(html);
    return;
  }

  if (path == "/autocomplete.css") {
    std::string css = makemore::slurp(path.c_str() + 1);

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: text/css\r\n");
    this->printf("Content-Length: %lu\r\n", css.length());
    this->printf("Cache-Control: public, max-age=86400\r\n");
    this->printf("\r\n");
    this->write(css);
    return;
  }

  if (path == "/autocomplete") {
    map<string, string> cgi;
    cgiparse(query, &cgi);
    string prefix = cgi["prefix"];

    Autocompleter *ac = server->urb->zones[0]->ac;
    assert(ac);

    std::vector<std::string> completions;
    ac->find(prefix, &completions);

    std::string json = "[";
    for (auto ci = completions.begin(); ci != completions.end(); ++ci) {
      if (ci != completions.begin())
        json += ",";
      json += "\n  \"" + *ci + "\"";
    }
    json += "\n]\n";

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: text/json\r\n");
    this->printf("Content-Length: %lu\r\n", json.length());
    this->printf("\r\n");
    this->write(json);
    return;
  }

  if (path == "/gennom.js" || path == "/autocomplete.js" || path == "/aes.js" || path == "/sha256.js" || path == "/crypto.js" || path == "/moretp.js") {
    std::string js = makemore::slurp(path.c_str() + 1);

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: text/javascript\r\n");
    this->printf("Content-Length: %lu\r\n", js.length());
    this->printf("Cache-Control: public, max-age=86400\r\n");
    this->printf("\r\n");
    this->write(js);
    return;
  }

  if (path == "/favicon.ico") {
    std::string fn = server->urb->dir + "/favicon.ico";
    std::string favicon = makemore::slurp(fn);
    
    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: image/x-icon\r\n");
    this->printf("Content-Length: %lu\r\n", favicon.length());
    this->printf("Cache-Control: public, max-age=86400\r\n");
    this->printf("\r\n");
    this->write(favicon);
    return;
  }


  if (path == "/top/minions.json") {
    map<string, string> cgi;
    cgiparse(query, &cgi);
    unsigned int n = 256;
    if (cgi["n"] != "")
      n = strtoul(cgi["n"].c_str(), NULL, 0);

    std::string json = "{";
    unsigned int k = 0;
    Zone *z = server->urb->zones[0];
    z->scrup();
    auto crw_nom = z->crw_nom;
    for (auto q = crw_nom.rbegin(); q != crw_nom.rend(); ++q) {
      if (q->first == 0)
        break;

      if (k > 0)
        json += ",";
      json += "\n";

      char buf[256];
      sprintf(buf, "  \"%s\": %u", q->second.c_str(), q->first);
      json += buf;

      if (++k >= n)
        break;
    }
    json += "\n}\n";

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: text/json\r\n");
    this->printf("Content-Length: %lu\r\n", json.length());
    this->printf("\r\n");
    this->write(json);
    return;
  }
  if (path == "/top/score.json") {
    map<string, string> cgi;
    cgiparse(query, &cgi);
    unsigned int n = 256;
    if (cgi["n"] != "")
      n = strtoul(cgi["n"].c_str(), NULL, 0);

    std::string json = "{";
    unsigned int k = 0;
    Zone *z = server->urb->zones[0];
    z->scrup();
    auto scr_nom = z->scr_nom;
    for (auto q = scr_nom.rbegin(); q != scr_nom.rend(); ++q) {
      if (q->first == 0)
        break;

      if (k > 0)
        json += ",";
      json += "\n";

      char buf[256];
      sprintf(buf, "  \"%s\": %u", q->second.c_str(), q->first);
      json += buf;

      if (++k >= n)
        break;
    }
    json += "\n}\n";

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: text/json\r\n");
    this->printf("Content-Length: %lu\r\n", json.length());
    this->printf("\r\n");
    this->write(json);
    return;
  }
  if (path == "/top/activity.json") {
    map<string, string> cgi;
    cgiparse(query, &cgi);
    unsigned int n = 256;
    if (cgi["n"] != "")
      n = strtoul(cgi["n"].c_str(), NULL, 0);

    std::string json = "{";
    unsigned int k = 0;
    Zone *z = server->urb->zones[0];
    z->actup();
    auto act_nom = z->act_nom;
    for (auto q = act_nom.rbegin(); q != act_nom.rend(); ++q) {
      if (k > 0)
        json += ",";
      json += "\n";

      char buf[256];
      sprintf(buf, "  \"%s\": %lf", q->second.c_str(), q->first);
      json += buf;

      if (++k >= n)
        break;
    }
    json += "\n}\n";

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: text/json\r\n");
    this->printf("Content-Length: %lu\r\n", json.length());
    this->printf("\r\n");
    this->write(json);
    return;
  }

  if (path == "/top/online.json") {
    double t = now();
    map<string, string> cgi;
    cgiparse(query, &cgi);
    unsigned int n = 256;
    if (cgi["n"] != "")
      n = strtoul(cgi["n"].c_str(), NULL, 0);

    std::string json = "{";
    unsigned int k = 0;
    Zone *z = server->urb->zones[0];
    z->onlup();
    auto onl_nom = z->onl_nom;
    for (auto q = onl_nom.rbegin(); q != onl_nom.rend(); ++q) {
      if (k > 0)
        json += ",";
      json += "\n";

      char buf[256];
      sprintf(buf, "  \"%s\": %lf", q->second.c_str(), t - q->first);
      json += buf;

      if (++k >= n)
        break;
    }
    json += "\n}\n";

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: text/json\r\n");
    this->printf("Content-Length: %lu\r\n", json.length());
    this->printf("\r\n");
    this->write(json);
    return;
  }

  if (path == "/newcomms.txt") {
    map<string, string> cgi;
    cgiparse(query, &cgi);

    string nom = cgi["nom"];
    if (!Parson::valid_nom(nom)) {
      http_notfound();
      return;
    }
    Parson *parson = server->urb->find(nom);
    if (!parson) {
      http_notfound();
      return;
    }
    string session = cgi["session"];
    if (!server->check_session(nom, session)) {
      http_denied();
      return;
    }

    char buf[16];
    sprintf(buf, "%d", parson->newcomms ? 1 : 0);
    std::string txt = buf;

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: text/plain\r\n");
    this->printf("Content-Length: %lu\r\n", txt.length());
    this->printf("\r\n");
    this->write(txt);
    return;
  }

  if (path == "/claim.txt") {
    map<string, string> cgi;
    cgiparse(query, &cgi);

    string usernom = cgi["usernom"];
    string session = cgi["session"];
    if (!server->check_session(usernom, session)) {
      http_denied();
      return;
    }
    Parson *user = server->urb->find(usernom);
    if (!user) {
      http_denied();
      return;
    }
    if (!*user->owner) {
      http_denied();
      return;
    }

    user->acted = time(NULL);

    string nom = cgi["nom"];
    if (!Parson::valid_nom(nom)) {
      http_notfound();
      return;
    }
    Parson *parson = server->urb->find(nom);
    if (!parson) {
      http_notfound();
      return;
    }

    if (*parson->owner && strcmp(user->owner, parson->owner)) {
      http_denied();
      return;
    }

    strcpy(parson->owner, user->owner);
    if (!strcmp(user->owner, user->nom)) {
      ++user->ncrew;
    } else {
      if (Parson *uo = server->urb->find(user->owner)) {
        ++uo->ncrew;
      }
    }

    parson->pass[0] = '*';
    memcpy(parson->pubkey, user->pubkey, sizeof(Parson::pubkey));

    std::string txt = "ok";
    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: text/plain\r\n");
    this->printf("Content-Length: %lu\r\n", txt.length());
    this->printf("\r\n");
    this->write(txt);
    return;
  }



  if (path == "/comms.json") {
    map<string, string> cgi;
    cgiparse(query, &cgi);

    string nom = cgi["nom"];
    if (!Parson::valid_nom(nom)) {
      http_notfound();
      return;
    }
    Parson *parson = server->urb->find(nom);
    if (!parson) {
      http_notfound();
      return;
    }
    string session = cgi["session"];
    if (!server->check_session(nom, session)) {
      http_denied();
      return;
    }

    parson->acted = time(NULL);
    parson->newcomms = 0;

    std::string json = "{";
    std::string home = server->urb->dir + "/home/" + nom;
    if (DIR *dp = opendir(home.c_str())) {
      int i = 0;
      struct dirent *de;
      while ((de = readdir(dp))) {
        const char *fn = de->d_name;
        const char *ext = strrchr(fn, '.');
        if (!ext || strcmp(ext, ".dat"))
          continue;

        std::string qnom(fn, ext - fn);
        if (!Parson::valid_nom(qnom))
          continue;
        if (!server->urb->find(qnom))
          continue;

        struct stat st;
        int ret = ::stat((home + "/" + fn).c_str(), &st);
        if (ret != 0)
          continue;

        char buf[256];
        sprintf(buf, "%s\n  \"%s\": [%lu, %lu]",
          i ? "," : "", qnom.c_str(), st.st_mtime, st.st_atime);
        json += buf;
        ++i;
      }
      closedir(dp);
    }
    json += "\n}";

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: text/json\r\n");
    this->printf("Content-Length: %lu\r\n", json.length());
    this->printf("\r\n");
    this->write(json);
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
      http_notfound();
      return;
    }


    Parson *parson = server->urb->make(nom, 0, 2);
    if (!parson) {
      http_notfound();
      return;
    }

    
    string txt = "huh";
    if (op == "scramble") {
      for (unsigned int j = 0; j < Parson::ncontrols; ++j)
        parson->controls[j] = randgauss();
      txt = "ok, scrambled " + nom;
    } else if (op == "tone") {
      double mul = 1.0;
      if (cgi["mul"] != "")
        mul = strtod(cgi["mul"].c_str(), NULL);
      for (unsigned int j = 0; j < Parson::ncontrols; ++j)
        parson->controls[j] *= mul;
      txt = "ok, toned " + nom;
    } else if (op == "norm") {
      double val = 1.0;
      if (cgi["val"] != "")
        val = strtod(cgi["val"].c_str(), NULL);

      double q = 0;

      for (unsigned int j = 0; j < Parson::ncontrols; ++j)
        q += parson->controls[j] * parson->controls[j];
      q /= Parson::ncontrols;
      q = sqrt(q);

      if (q < 1e-6)
        q = 1e-6;
      q = 1.0 / q; 
      q *= val;

      for (unsigned int j = 0; j < Parson::ncontrols; ++j) {
        parson->controls[j] *= q;
//        parson->variations[j] *= q;
      }

      txt = "ok, normed " + nom;
    } else if (op == "addvec") {
      double val = 0.0;
      if (cgi["val"] != "")
        val = strtod(cgi["val"].c_str(), NULL);

      double mul = 1.0;
      if (cgi["mul"] != "")
        mul = strtod(cgi["mul"].c_str(), NULL);

      double vdev = 0.0;
      if (cgi["vdev"] != "")
        vdev = strtod(cgi["vdev"].c_str(), NULL);

      unsigned int r = randuint();
      if (cgi["r"] != "")
        r = strtoul(cgi["r"].c_str(), NULL, 0);

      if (Parson::valid_nom(cgi["vec"])) {
        if (Parson *vec = server->urb->find(cgi["vec"])) {
          seedrand(r);

          for (unsigned int j = 0; j < Parson::ncontrols; ++j) {
            //parson->controls[j] += mul * vec->controls[j];
            double ma = parson->controls[j];
            double va = parson->variations[j];
            double vb = vec->variations[j];
            double mb = vec->controls[j] + randgauss() * vdev * sqrt(vb);
            double vbk = pow(vb, mul);

            double vbg;
            if (fabs(vb - 1) < 1e-6) {
               vbg = mul;
            } else {
               vbg = (vbk - 1.0) / (vb - 1.0);
            }
            parson->controls[j] = vbk * ma + vbg * mb;

            parson->variations[j] = va * vbk;

            // parson->variations[j] = (1.0 - mul) * va + mul * (vb + (ma - mb) * (ma - mb));

            // parson->variations[j] = (1.0 - mul) * va + mul * ((ma - mb) * (ma - mb));
          }
        }
      }

      double q = 0;
      for (unsigned int j = 0; j < Parson::ncontrols; ++j)
        q += parson->controls[j] * parson->controls[j];
      q /= Parson::ncontrols;
      q = sqrt(q);

      if (val > 0) {
        for (unsigned int j = 0; j < Parson::ncontrols; ++j) {
           parson->controls[j]   *= val / q;
//           parson->variations[j] *= val / q;
        }
        q = val;
      }

      char buf[256];
      sprintf(buf, "%lf", q);
      txt = buf;
    } else if (op == "blend") {
      double mul = 1.0;
      if (cgi["mul"] != "")
        mul = strtod(cgi["mul"].c_str(), NULL);

      double vdev = 0.0;
      if (cgi["vdev"] != "")
        vdev = strtod(cgi["vdev"].c_str(), NULL);

      unsigned int r = randuint();
      if (cgi["r"] != "")
        r = strtoul(cgi["r"].c_str(), NULL, 0);

      if (Parson::valid_nom(cgi["vec"])) {
        if (Parson *vec = server->urb->find(cgi["vec"])) {
          seedrand(r);
          for (unsigned int j = 0; j < Parson::ncontrols; ++j) {
            //parson->controls[j] += mul * vec->controls[j];
            double ma = parson->controls[j];
            double va = parson->variations[j];
            double vb = vec->variations[j];
            double mb = vec->controls[j] + randgauss() * sqrt(vb) * vdev;
            parson->controls[j] = (1.0 - mul) * ma + mul * mb;
            // parson->variations[j] = (1.0 - mul) * va + mul * (vb + (ma - mb) * (ma - mb));
            parson->variations[j] = (1.0 - mul) * va + mul * ((ma - mb) * (ma - mb));
          }
        }
      }

      double q = 0;
      for (unsigned int j = 0; j < Parson::ncontrols; ++j)
        q += parson->controls[j] * parson->controls[j];
      q /= Parson::ncontrols;
      q = sqrt(q);

      char buf[256];
      sprintf(buf, "%lf", q);
      txt = buf;
    } else if (op == "bread") {
      unsigned int r = randuint();
      if (cgi["r"] != "")
        r = strtoul(cgi["r"].c_str(), NULL, 0);

      std::string childnom;
      Parson *child;

      if (Parson::valid_nom(cgi["vec"])) {
        if (Parson *vec = server->urb->find(cgi["vec"])) {
          seedrand(r);

          childnom = Parson::bread_nom(parson->nom, vec->nom, randuint() % 2);
          child = server->urb->make(childnom);

          for (unsigned int j = 0; j < Parson::ncontrols; ++j) {
            if (randuint() % 2) {
              child->controls[j] = parson->controls[j];
              child->variations[j] = parson->variations[j];
            } else {
              child->controls[j] = vec->controls[j];
              child->variations[j] = vec->variations[j];
            }
          }

          child->set_parens(parson->nom, vec->nom);
          parson->add_fren(child->nom);
          vec->add_fren(child->nom);

          child->add_fren(parson->nom);
          child->add_fren(vec->nom);
        }
      }

      txt = childnom;
    } else if (op == "mash") {
      double mul = 1.0;
      if (cgi["mul"] != "")
        mul = strtod(cgi["mul"].c_str(), NULL);

      if (Parson::valid_nom(cgi["vec"])) {
        if (Parson *vec = server->urb->find(cgi["vec"])) {
          for (unsigned int j = 0; j < Parson::ncontrols; ++j) {
            if (randrange(0, 1) < mul) {
              parson->controls[j] = vec->controls[j];
              parson->variations[j] = vec->variations[j];
            }
          }
        }
      }

      double q = 0;
      for (unsigned int j = 0; j < Parson::ncontrols; ++j)
        q += parson->controls[j] * parson->controls[j];
      q /= Parson::ncontrols;
      q = sqrt(q);

      char buf[256];
      sprintf(buf, "%lf", q);
      txt = buf;
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
    } else if (op == "addfren") {
      std::string fren = cgi["fren"];
      if (fren == "") {
        Parson *q = server->urb->zones[0]->pick();
        fren = q->nom;
      }

      if (Parson::valid_nom(fren.c_str())) {
        parson->add_fren(fren.c_str());

        Parson *fp = server->urb->make(fren, 0);
        fp->add_fren(parson->nom);

        txt = "ok, added fren " + fren;
      } else {
        txt = "oops";
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
  if (path == "/new") {
    nom = Parson::gen_nom();

    this->printf("HTTP/1.1 302 Redirect\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: text/plain\r\n");
    this->printf("Content-Length: 0\r\n");
    this->printf("Location: /%s\r\n", nom.c_str());
    this->printf("\r\n", nom.c_str());
    return;
  }

  if (strbegins(path, "/new/")) {
    const char *p = path.c_str() + 5;
    const char *ext = strrchr(p, '.');
    std::string tplnom = ext ? std::string(p, ext - p) : p;
    if (!ext)
      ext = "html";
    else
      ++ext;

    if (strcmp(ext, "txt") && strcmp(ext, "html")) {
      this->http_notfound();
      return;
    }

    if (!Parson::valid_nom(tplnom)) {
      this->http_notfound();
      return;
    }
    Parson *tpl = server->urb->find(tplnom);
    if (!tpl) {
      this->http_notfound();
      return;
    }

    Parson *prs = NULL;
    std::string newnom;
    do {
      newnom = Parson::gen_nom();
      prs = server->urb->find(newnom);
    } while (prs);

    prs = server->urb->make(newnom, 0);
    if (!prs) {
      this->http_notfound();
      return;
    }

    for (unsigned int j = 0; j < Parson::ncontrols; ++j) {
      prs->variations[j] = tpl->variations[j];
      prs->controls[j] = tpl->controls[j] + randgauss() * sqrt(tpl->variations[j]);
    }

    prs->set_parens(tpl->nom, tpl->nom);

    if (!strcmp(ext, "html")) {
      this->printf("HTTP/1.1 302 Redirect\r\n");
      this->printf("Connection: keep-alive\r\n");
      this->printf("Content-Type: text/plain\r\n");
      this->printf("Content-Length: 0\r\n");
      this->printf("Location: /%s\r\n", newnom.c_str());
      this->printf("\r\n", newnom.c_str());
    } else if (!strcmp(ext, "txt")) {
      this->printf("HTTP/1.1 200 OK\r\n");
      this->printf("Connection: keep-alive\r\n");
      this->printf("Content-Type: text/plain\r\n");
      this->printf("Content-Length: %lu\r\n", newnom.length());
      this->printf("\r\n");
      this->printf("%s", newnom.c_str());
    }
    return;
  }

  if (path == "/") {
    Parson *prs = server->urb->zones[0]->pick();
    // nom = Parson::gen_nom();
    nom = prs->nom;

    this->printf("HTTP/1.1 302 Redirect\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: text/plain\r\n");
    this->printf("Content-Length: 0\r\n");
    this->printf("Location: /%s\r\n", nom.c_str());
    this->printf("\r\n", nom.c_str());
    return;
  }

  nom = path.c_str() + 1;
  std::string ext = "html";
  if (const char *p = strchr(nom.c_str(), '.')) {
    ext = p + 1;
    nom = std::string(nom.c_str(), p - nom.c_str());
  }

  std::string func = "file";
  if (const char *p = strchr(nom.c_str(), '/')) {
    func = p + 1;
    nom = std::string(nom.c_str(), p - nom.c_str());
  }

//fprintf(stderr, "nom=[%s] is_png=%d\n", 

  if (!Parson::valid_nom(nom)) {
    http_notfound();
    return;
  }

  if (func == "mob" && ext == "png") {
    Supergen *gen = server->urb->default_gen;
    Styler *sty = server->urb->default_sty;

    Partrait mpar;
    make_mob(gen, sty, NULL, &mpar);
    std::string png;
    mpar.to_png(&png);

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: image/png\r\n");
    this->printf("Content-Length: %lu\r\n", png.length());
    this->printf("\r\n");
    this->write(png);
    return;
  }

  if (ext == "dat") {
    map<string, string> cgi;
    cgiparse(query, &cgi);
    unsigned int off = strtoul(cgi["off"].c_str(), NULL, 0);

    std::string convnom = func;
    if (!Parson::valid_nom(convnom)) {
      http_notfound();
      return;
    }

    std::string fn = server->urb->dir + "/home/" + nom + "/" + convnom + ".dat";
    FILE *fp = fopen(fn.c_str(), "r");
    if (!fp) {
      this->printf("HTTP/1.1 200 OK\r\n");
      this->printf("Connection: keep-alive\r\n");
      this->printf("Content-Type: application/octet-stream\r\n");
      this->printf("Content-Length: 0\r\n");
      this->printf("\r\n");
      return;
    }
    std::string dat = makemore::slurp(fp);

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: application/octet-stream\r\n");
    this->printf("Content-Length: %lu\r\n", dat.length());
    this->printf("\r\n");
    this->write(dat);
    return;
  }

  if (ext == "json" && func == "crew") {
    server->urb->zones[0]->crwup();
    auto crewmap = server->urb->zones[0]->crewmap[nom];

    char buf[256];
    std::string json = "[";
    for (unsigned int i = 0; i < crewmap.size(); ++i) {
      if (i > 0)
        json += ",";
      json += "\n";
      sprintf(buf, "  \"%s\"", crewmap[i].c_str());
      json += buf;
    }
    json += "\n]\n";

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: text/json\r\n");
    this->printf("Content-Length: %lu\r\n", json.length());
    this->printf("\r\n");
    this->write(json);
    return;
  }

  if (ext == "json" && func == "source") {
    map<string, string> cgi;
    cgiparse(query, &cgi);

    Parson *prs = server->urb->find(nom);
    if (!prs) {
      this->http_notfound();
      return;
    }

    if (!*prs->srcfn) {
      this->http_notfound();
      return;
    }

    FILE *fp = fopen(prs->srcfn, "r");
    if (!fp) {
      this->http_notfound();
      return;
    }
    Partrait prt;
    prt.load(fp);
    fclose(fp);
 
    Triangle tri = prt.get_mark();
    Triangle tribak = tri;

    bool changed = false;
    if (cgi["px"] != "") { changed = true; tri.p.x = strtod(cgi["px"].c_str(), NULL); }
    if (cgi["py"] != "") { changed = true; tri.p.y = strtod(cgi["py"].c_str(), NULL); }
    if (cgi["qx"] != "") { changed = true; tri.q.x = strtod(cgi["qx"].c_str(), NULL); }
    if (cgi["qy"] != "") { changed = true; tri.q.y = strtod(cgi["qy"].c_str(), NULL); }
    if (cgi["rx"] != "") { changed = true; tri.r.x = strtod(cgi["rx"].c_str(), NULL); }
    if (cgi["ry"] != "") { changed = true; tri.r.y = strtod(cgi["ry"].c_str(), NULL); }
    if (changed) {
      prt.set_mark(tri);
      prt.save(prs->srcfn);

      Partrait stdprt(256, 256);
      stdprt.set_pose(Pose::STANDARD);
      prt.warp(&stdprt);

      Styler *sty = server->urb->get_sty(prs->sty);
      assert(sty);
      server->urb->enc->encode(stdprt, prs->controls);
      sty->encode(prs->controls, prs);
    }

    std::string json;
    char jbuf[256];
    json += "{\n";
    sprintf(jbuf, "  \"p\": [%d, %d],\n", (int)tri.p.x, (int)tri.p.y); json += jbuf;
    sprintf(jbuf, "  \"q\": [%d, %d],\n", (int)tri.q.x, (int)tri.q.y); json += jbuf;
    sprintf(jbuf, "  \"r\": [%d, %d]\n", (int)tri.r.x, (int)tri.r.y); json += jbuf;
    json += "}\n";

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: text/json\r\n");
    this->printf("Content-Length: %lu\r\n", json.length());
    this->printf("\r\n");
    this->write(json);
    return;
  }

  if (ext == "png" && func == "source") {
    Parson *prs = server->urb->find(nom);
    if (!prs) {
      this->http_notfound();
      return;
    }

    if (!*prs->srcfn) {
      this->http_notfound();
      return;
    }

    FILE *fp = fopen(prs->srcfn, "r");
    if (!fp) {
      this->http_notfound();
      return;
    }
    std::string png = makemore::slurp(fp);
    fclose(fp);

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: image/png\r\n");
    this->printf("Content-Length: %lu\r\n", png.length());
    this->printf("\r\n");
    this->write(png);
    return;
  }

  if (ext == "html") {
  if (func == "enc" || func == "file" || func == "fam" || func == "frens" || func == "xform" || func == "mem" || func == "mob" || func == "top" || func == "msg") {
    std::string html = makemore::slurp(func + ".html");

    std::string header = makemore::slurp("header.html");
    std::string subhead = makemore::slurp("subhead.html");
    std::string url = "https://peaple.io/" + nom + "/" + func;
    std::string escurl = "https%3A%2F%2Fpeaple.io%2F" + nom + "%2F" + func;
    html = replacestr(html, "$HEADER", header);
    html = replacestr(html, "$SUBHEAD", subhead);
    html = replacestr(html, "$URL", url);
    html = replacestr(html, "$ESCURL", escurl);
    html = replacestr(html, "$NOM", nom);

    unsigned int r = randuint();
    char rbuf[256];
    sprintf(rbuf, "%u", r);
    html = replacestr(html, "$RAND", rbuf);

    Parson *prs = server->urb->find(nom);
    html = replacestr(html, "$LOCKED", prs && prs->has_pass() ? "1" : "0");

    std::string hexpk;
    if (prs)
      hexpk = to_hex(std::string((char *)prs->pubkey, 128));
    html = replacestr(html, "$HEXPUBKEY", hexpk);

    std::string owner = "";
    if (prs && Parson::valid_nom(prs->owner))
      owner = prs->owner;
    html = replacestr(html, "$MAKER", owner);

    char sbuf[256];
    sprintf(sbuf, "%llu", prs ? prs->score : 0ULL);
    html = replacestr(html, "$SCORE", sbuf);
    sprintf(sbuf, "%llu", prs ? prs->ncrew : 0ULL);
    html = replacestr(html, "$NCREW", sbuf);

    char abuf[256];
    sprintf(abuf, "%lf", prs ? prs->activity() : 0.0);
    html = replacestr(html, "$ACTIVITY", abuf);

    char obuf[256];
    sprintf(obuf, "%u", prs && !strcmp(prs->owner, prs->nom) ? prs->acted : 0);
    html = replacestr(html, "$ONLINE", obuf);

    char tbuf[256];
    sprintf(tbuf, "%u", prs ? prs->created : (unsigned int)time(NULL));
    html = replacestr(html, "$CREATED", tbuf);

    char nvbuf[256];
    strcpy(nvbuf, "0.0");
    if (prs) {
      double z = 0;
      for (unsigned int j = 0; j < Parson::ncontrols; ++j)
        z += prs->controls[j] * prs->controls[j];
      z /= (double)Parson::ncontrols;
      z = sqrt(z);
      sprintf(nvbuf, "%lf", z);
    }
    html = replacestr(html, "$DEV", nvbuf);

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: text/html\r\n");
    this->printf("Content-Length: %lu\r\n", html.length());
    this->printf("\r\n");
    this->write(html);
    return;
  }
  }

  if (ext == "json") {
    Parson *parson = server->urb->make(nom, 0, 2);
    assert(parson);

    Parson *paren0 = server->urb->make(parson->parens[0], 0);
    Parson *paren1 = server->urb->make(parson->parens[1], 0);

    std::string json;
    json += "{\n";
    json += string("  \"nom\": \"") + parson->nom + "\",\n";
    json += string("  \"gen\": \"") + parson->gen + "\",\n";
    json += string("  \"sty\": \"") + parson->sty + "\",\n";
    json += string("  \"sks\": \"") + parson->sks + "\",\n";
    json += string("  \"srcfn\": \"") + parson->srcfn + "\",\n";

double dev = 0;
for (unsigned int j = 0; j < Parson::ncontrols; ++j) {
  dev += parson->controls[j] * parson->controls[j];
}
dev /= (double)Parson::ncontrols;
dev = sqrt(dev);

    char numstr[256];
    sprintf(numstr, "%lf", dev);
    json += string("  \"tone\": ") + numstr + ",\n";

    std::vector<Parson*> kids;
    server->urb->zones[0]->scan_kids(parson->nom, &kids, 7);
    json += string("  \"kids\": [");
    int j = 0;
    for (auto kid : kids) {
      if (j)
        json += ", ";
      json += string("\"") + kid->nom + string("\"");
      ++j;
    }
    json += "],\n";

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

    json += string("  \"gparens\": [");
    for (unsigned int j = 0; j < 4; ++j) {
      const char *gparen = (j < 2 ? paren0 : paren1)->parens[j % 2];
      if (!*gparen)
        continue;
      if (j)
        json += ", ";
      json += string("\"") + gparen + string("\"");
    }
    json += "],\n";

    json += string("  \"frens\": [");
    for (unsigned int j = 0; j < Parson::nfrens; ++j) {
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
    map<string, string> cgi;
    cgiparse(query, &cgi);

    double dev = 1.0;
    if (cgi["dev"] != "")
      dev = strtod(cgi["dev"].c_str(), NULL);

    double vdev = 0.0;
    if (cgi["vdev"] != "")
      vdev = strtod(cgi["vdev"].c_str(), NULL);

    unsigned int r = randuint();
    if (cgi["r"] != "")
      r = strtoul(cgi["r"].c_str(), NULL, 0);

    unsigned int dim = 256;
    if (cgi["dim"] != "")
      dim = strtoul(cgi["dim"].c_str(), NULL, 0);
    if (dim > 256)
      dim = 256;
    if (dim < 32)
      dim = 32;

    Parson *parson = server->urb->make(nom, 0);

parson->visit(1);
if (*parson->owner) {
  if (Parson *m = server->urb->find(parson->owner)) {
    ++m->score;
  }
}


//fprintf(stderr, "[%lf %lf %lf]\n", pose.angle, pose.stretch, pose.skew);

    Superenc *enc = server->urb->enc;
    Supergen *gen = server->urb->get_gen(parson->gen);
//gen->load();
    Styler *sty = server->urb->get_sty(parson->sty);

    Partrait genpar(256, 256);


double *tmp = new double[Parson::ncontrols];
memcpy(tmp, parson->controls, sizeof(double) * Parson::ncontrols);

seedrand(r);

fprintf(stderr, "dev=%lf\n", dev);
for (unsigned int j = 0; j < Parson::ncontrols; ++j) {
  tmp[j] += randgauss() * sqrt(parson->variations[j]) * vdev;
  tmp[j] *= dev;
}

sty->tag_cholo["base"]->generate(tmp, tmp);
gen->generate(tmp, &genpar);
delete[] tmp;


Partrait showpar(dim, dim);
Pose pose = Pose::STANDARD;
pose.center *= (double)dim / 256.0;
pose.scale *= (double)dim / 256.0;

showpar.set_pose(pose);
genpar.warp(&showpar);

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

  if (ext == "pubkey.dat") {
    uint8_t pubkey[128];
    memset(pubkey, 0, 128);
    Parson *parson = server->urb->find(nom);
    if (parson)
      memcpy(pubkey, parson->pubkey, 128);

    this->printf("HTTP/1.1 200 OK\r\n");
    this->printf("Connection: keep-alive\r\n");
    this->printf("Content-Type: application/octet-stream\r\n");
    this->printf("Content-Length: %lu\r\n", 128);
    this->printf("\r\n");

    std::string spubkey((char *)pubkey, 128);
    this->write(spubkey);
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

this->printf("HTTP/1.1 302 Redirect\r\n");
this->printf("Connection: keep-alive\r\n");
this->printf("Content-Type: text/plain\r\n");
this->printf("Content-Length: 0\r\n");
this->printf("Location: /%s\r\n", nom.c_str());
this->printf("\r\n");
return;


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
    htmlbuf += htmlpe(parson->nom, 512);
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
