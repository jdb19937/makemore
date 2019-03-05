#define __MAKEMORE_SERVER_CC__ 1
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/fcntl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <netinet/in.h>
#include <string.h>

#include <string>
#include <vector>
#include <list>

#include "strutils.hh"
#include "parson.hh"
#include "server.hh"
#include "ppm.hh"
#include "commands.hh"
#include "agent.hh"

namespace makemore {

std::map<std::string, Server::Handler> Server::default_cmdtab;

using namespace std;

void Server::nonblock(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  assert(flags > 0);
  int ret = fcntl(fd, F_SETFL, flags | O_NONBLOCK);
  assert(ret == 0);
}

static inline double realtime() {
  clock_t c = clock();
  return ((double)c / (double)CLOCKS_PER_SEC);
}

static bool _openssl_initialized = false;

Server::Server(const std::string &_urbdir) {
  urbdir = _urbdir;
  port = 0;
  s = -1;
  urb = NULL;

  cmdtab = default_cmdtab;

  session_key.resize(64);
  for (unsigned int i = 0;  i < 64; ++i)
    session_key[i] = randuint() % 256;



  if (!_openssl_initialized) {
    SSL_load_error_strings();	
    OpenSSL_add_ssl_algorithms();
    _openssl_initialized = true;
  }


  ssl_s = -1;
  ssl_port = 0;
  ssl_ctx = SSL_CTX_new(SSLv23_server_method());
  assert(ssl_ctx);

  SSL_CTX_set_ecdh_auto(ctx, 1);

  std::string certdir = urbdir + "/certs";

  assert(
    SSL_CTX_use_certificate_chain_file(
      ssl_ctx,
      (certdir + "/fullchain.pem").c_str()
    ) > 0
  );

  assert(
    SSL_CTX_use_certificate_file(
      ssl_ctx,
      (certdir + "/cert.pem").c_str(),
      SSL_FILETYPE_PEM
    ) > 0
  );

  assert(
    SSL_CTX_use_PrivateKey_file(
      ssl_ctx,
      (certdir + "/privkey.pem").c_str(),
      SSL_FILETYPE_PEM
    ) > 0
  );
}


Server::~Server() {
  kill();
  close();

  SSL_CTX_free(ssl_ctx);
}


bool Server::check_session(const std::string &nom, const std::string &session) {
  if (session.length() != 80)
    return false;
  if (!Parson::valid_nom(nom))
    return false;

  string hash(session, 0, 32);
  string nonce(session, 32, 32);

  string expstr(session, 64, 16);
  uint64_t expires = strtoul(expstr.c_str(), NULL, 16);
  
  time_t now = time(NULL);
  if (expires - 0x2645751311064591 < now)
    return false;

  Parson *parson = urb->find(nom);
  if (!parson)
    return false;

  uint8_t rehashbin[32];
  SHA256_CTX sha;
  SHA256_Init(&sha);
  SHA256_Update(&sha, (const uint8_t *)session_key.data(), session_key.length() + 1);
  SHA256_Update(&sha, (uint8_t *)parson->pass, sizeof(Parson::pass));
  SHA256_Update(&sha, (const uint8_t *)nom.data(), nom.length() + 1);
  SHA256_Update(&sha, (const uint8_t *)nonce.data(), nonce.length() + 1);
  SHA256_Update(&sha, (const uint8_t *)&expires, sizeof(expires));
  SHA256_Final(rehashbin, &sha);

  string rehash = to_hex(string((char *)rehashbin, 16));
  if (rehash != hash)
    return false;

  return true;
}


std::string Server::make_session(const std::string &nom, unsigned long duration) {
  string noncebin;
  noncebin.resize(16);
  for (unsigned int i = 0, n = noncebin.size(); i < n; ++i)
    noncebin[i] = randuint() % 256;
  string nonce = to_hex(noncebin);

  time_t now = time(NULL);
  uint64_t expires = now + duration;
  expires += 0x2645751311064591;
  char expbuf[17];
  sprintf(expbuf, "%016lX", expires);
  string expstr = string(expbuf, 16);

  char pass[32];
  if (Parson *parson = urb->find(nom)) {
    assert(sizeof(Parson::pass) == sizeof(pass));
    memcpy(pass, parson->pass, sizeof(pass));
  } else {
    memset(pass, 0, sizeof(pass));
  }

  uint8_t hashbin[32];
  SHA256_CTX sha;
  SHA256_Init(&sha);
  SHA256_Update(&sha, (uint8_t *)session_key.data(), session_key.length() + 1);
  SHA256_Update(&sha, (uint8_t *)pass, sizeof(pass));
  SHA256_Update(&sha, (uint8_t *)nom.c_str(), nom.length() + 1);
  SHA256_Update(&sha, (uint8_t *)nonce.c_str(), nonce.length() + 1);
  SHA256_Update(&sha, (uint8_t *)&expires, sizeof(expires));
  SHA256_Final(hashbin, &sha);

  string hash = to_hex(string((char *)hashbin, 16));

  string session = hash + nonce + expstr;
  assert(session.length() == 80);

  return session;
}
 


void Server::open() {
  int ret;

  this->close();
  s = socket(PF_INET, SOCK_STREAM, 0);
  assert(s >= 0);

  int reuse = 1;
  ret = setsockopt(s, SOL_SOCKET, SO_REUSEPORT, (const char*)&reuse, sizeof(reuse));
  assert(ret == 0);
  ret = setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse));
  assert(ret == 0);

  nonblock(s);



  ssl_s = socket(PF_INET, SOCK_STREAM, 0);
  assert(ssl_s >= 0);

  reuse = 1;
  ret = setsockopt(ssl_s, SOL_SOCKET, SO_REUSEPORT, (const char*)&reuse, sizeof(reuse));
  assert(ret == 0);
  ret = setsockopt(ssl_s, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse));
  assert(ret == 0);

  nonblock(ssl_s);
}

void Server::close() {
  if (s >= 0) {
    ::close(s);
    port = 0;
    s = -1;
  }

  if (ssl_s >= 0) {
    ::close(ssl_s);
    ssl_port = 0;
    ssl_s = -1;
  }
}

void Server::bind(uint16_t _port, uint16_t _ssl_port) {
  assert(s >= 0);
  assert(ssl_s >= 0);
  int ret;
  struct sockaddr_in sin;

  sin.sin_family = AF_INET;
  sin.sin_port = htons(_port);
  sin.sin_addr.s_addr = htonl(INADDR_ANY);
  ret = ::bind(s, (struct sockaddr *)&sin, sizeof(sin));
  assert(ret == 0);

  sin.sin_family = AF_INET;
  sin.sin_port = htons(_ssl_port);
  sin.sin_addr.s_addr = htonl(INADDR_ANY);
  ret = ::bind(ssl_s, (struct sockaddr *)&sin, sizeof(sin));
  assert(ret == 0);

  port = _port;
  ssl_port = _ssl_port;
}

void Server::listen(int backlog) {
  assert(s >= 0);
  int ret = ::listen(s, backlog);
  assert(ret == 0);

  assert(ssl_s >= 0);
  ret = ::listen(ssl_s, backlog);
  assert(ret == 0);
}

void Server::setup() {
  fprintf(stderr, "opening urb %s\n", urbdir.c_str());
  urb = new Urb(urbdir.c_str(), 2);
  fprintf(stderr, "opened urb %s\n", urbdir.c_str());

  seedrand();
}


void Server::start(unsigned int kids) {
assert(0);

#if 0
  for (unsigned int i = 0; i < kids; ++i) {
    fprintf(stderr, "forking\n");
    if (pid_t pid = fork()) {
      pids.push_back(pid);
      fprintf(stderr, "forked pid=%d\n", pid);
      continue;
    }

    this->setup();
    while (1) {
      this->accept();
    }

    assert(0);
  }
#endif
}

static std::string ipstr(uint32_t ip) {
  char buf[INET_ADDRSTRLEN];
  const char *retbuf = inet_ntop(AF_INET, &ip, buf, INET_ADDRSTRLEN);
  assert(retbuf == buf);
  return std::string(buf);
}

void Server::accept() {
  fprintf(stderr, "accepting\n");

  for (unsigned int i = 0; i < 16; ++i) {
    struct sockaddr_in cin;
    socklen_t cinlen = sizeof(cin);
    int c = ::accept(s, (struct sockaddr *)&cin, &cinlen);
    if (c < 0)
      return;

    nonblock(c);

    uint32_t agent_ip;
    assert(sizeof(struct in_addr) == 4);
    memcpy(&agent_ip, &cin.sin_addr, 4);
    fprintf(stderr, "accepted %s\n", ipstr(agent_ip).c_str());

    Agent *agent = new Agent(this, NULL, c, agent_ip);
    assert(agent->who);

    nom_agent.insert(make_pair(agent->who->nom, agent));
    agents.insert(agent);
  }
}

void Server::ssl_accept() {
  fprintf(stderr, "accepting\n");
  for (unsigned int i = 0; i < 16; ++i) {
    struct sockaddr_in cin;
    socklen_t cinlen = sizeof(cin);
    int c = ::accept(ssl_s, (struct sockaddr *)&cin, &cinlen);
    if (c < 0)
      return;

    nonblock(c);

    uint32_t agent_ip;
    assert(sizeof(struct in_addr) == 4);
    memcpy(&agent_ip, &cin.sin_addr, 4);
    fprintf(stderr, "accepted %s (ssl)\n", ipstr(agent_ip).c_str());

    Agent *agent = new Agent(this, NULL, c, agent_ip, true);
    assert(agent->who);

    nom_agent.insert(make_pair(agent->who->nom, agent));
    agents.insert(agent);
  }
}

void Server::renom(Agent *agent, const std::string &newnom) {
  assert(agent->who);
  if (agent->who->nom == newnom)
    return;

  auto naip = nom_agent.equal_range(agent->who->nom);
  for (auto nai = naip.first; nai != naip.second; ++nai) {
    if (nai->second == agent) {
      nom_agent.erase(nai);
      break;
    }
  }

  agent->who->become(newnom);

  nom_agent.insert(make_pair(agent->who->nom, agent));
}

void Server::notify(const std::string &nom, const std::string &msg, const Agent *exclude) {
  auto naip = nom_agent.equal_range(nom);

  for (auto nai = naip.first; nai != naip.second; ++nai) {
    const std::string &nom = nai->first;
    Agent *agent = nai->second;

    if (agent != exclude)
      agent->write(msg + "\n");
  }
}


void Server::wait() {
  for (auto pidi = pids.begin(); pidi != pids.end(); ++pidi) {
    pid_t pid = *pidi;
    fprintf(stderr, "waiting for pid=%d\n", pid);
    assert(pid == ::waitpid(pid, NULL, 0));
    fprintf(stderr, "waited\n");
  }
  pids.clear();
}

void Server::kill() {
  for (auto pidi = pids.begin(); pidi != pids.end(); ++pidi) {
    pid_t pid = *pidi;
    fprintf(stderr, "killing pid=%d\n", pid);
    ::kill(pid, 9);
    fprintf(stderr, "killed\n");
  }
}

void Server::websockify(uint16_t ws_port, const char *keydir) {
  assert(port > 0);
  pid_t ws_pid;

  ws_pid = fork();
  assert(ws_pid != -1);
  if (ws_pid) {
    pids.push_back(ws_pid);
    return;
  }

  char buf[256];
  sprintf(buf, ":%hu", port);
  std::string portstr = buf;
  sprintf(buf, "%hu", ws_port);
  std::string ws_portstr = buf;

  if (keydir) {
    string certstr = string(keydir) + string("/cert.pem");
    string privkeystr = string(keydir) + string("/privkey.pem");
    execlp("websockify",
      "websockify",
      "--cert", certstr.c_str(),
      "--key", privkeystr.c_str(),
      ws_portstr.c_str(), portstr.c_str(),
      NULL
    );
  } else {
    execlp("websockify",
      "websockify", ws_portstr.c_str(), portstr.c_str(),
      NULL
    );
  }

  assert(0);
}


void Server::select() {
  FD_ZERO(fdsets + 0);
  FD_ZERO(fdsets + 1);
  FD_ZERO(fdsets + 2);

  int nfds = s + 1;
  if (ssl_s >= nfds)
    nfds = ssl_s + 1;

  FD_SET(s, fdsets + 0);
  FD_SET(ssl_s, fdsets + 0);

  for (auto agent : agents) {
    if (agent->s < 0)
      continue;
    if (agent->s >= nfds)
      nfds = agent->s + 1;

    switch (agent->ssl_status) {
    case SSL_ERROR_WANT_READ:
      FD_SET(agent->s, fdsets + 0);
      break;
    case SSL_ERROR_WANT_WRITE:
      FD_SET(agent->s, fdsets + 1);
      break;
    case SSL_ERROR_NONE:
      if (agent->outbufn)
        FD_SET(agent->s, fdsets + 1);
      if (agent->inbufn < agent->inbufm)
        FD_SET(agent->s, fdsets + 0);
      break;
    default:
      fprintf(stderr, "got ssl status %d\n", agent->ssl_status);
      agent->close();
      break;
    }
  }

  struct timeval timeout;
  timeout.tv_sec = 0;
  timeout.tv_usec = 50000;

  fprintf(stderr, "calling select\n");
  int ret = ::select(nfds, fdsets + 0, fdsets + 1, fdsets + 2, &timeout);
  fprintf(stderr, "select ret=%d\n", ret);
}

void Server::main() {
  while (1) {
    auto ai = agents.begin();
    while (ai != agents.end()) {
      Agent *agent = *ai;
      if (agent->s < 0) {
        fprintf(stderr, "closing %s\n", ipstr(agent->ip).c_str());

        auto naip = nom_agent.equal_range(agent->who->nom);
        for (auto nai = naip.first; nai != naip.second; ++nai) {
          if (nai->second == agent) {
            nom_agent.erase(nai);
            break;
          }
        }

        agents.erase(ai++);
        delete agent;
        continue;
      }
      ++ai;
    }
 

    this->select();

    if (FD_ISSET(s, fdsets + 0))
      this->accept();
    if (FD_ISSET(ssl_s, fdsets + 0))
      this->ssl_accept();

    ai = agents.begin();
    while (ai != agents.end()) {
      Agent *agent = *ai;

      if (FD_ISSET(agent->s, fdsets + 0)) {
        if (!agent->slurp()) {
          fprintf(stderr, "closing %s\n", ipstr(agent->ip).c_str());

          auto naip = nom_agent.equal_range(agent->who->nom);
          for (auto nai = naip.first; nai != naip.second; ++nai) {
            if (nai->second == agent) {
              nom_agent.erase(nai);
              break;
            }
          }

          agents.erase(ai++);
          delete agent;
          continue;
        }
      }

      ++ai;
    }

    for (auto agent : agents) {
      if (FD_ISSET(agent->s, fdsets + 1)) {
        agent->flush();
      }
    }

    for (unsigned int i = 0; i < 256; ++i) {
      Parson *parson = urb->zones[0]->pick();
      if (!parson)
         continue;
      char *reqstr = parson->popbrief();
      if (!reqstr)
         continue;

      string req(reqstr);
      memset(reqstr, 0, Parson::briefsize);

      vector<string> reqwords;
      splitwords(req, &reqwords);

      Agent agent(this, parson->nom, -1, 0x0100007F);
      agent.command(reqwords);

      if (!agent.outbufn)
        continue;
      string rspstr = string(agent.outbuf, agent.outbufn);
fprintf(stderr, "rspstr=%s\n", rspstr.c_str());
      vector<string> rsps;
      splitlines(rspstr, &rsps);

      for (auto rspi = rsps.rbegin(); rspi != rsps.rend(); ++rspi) {
        parson->pushbrief(*rspi + " | " + req);
      }
    }


    for (auto agent : agents) {
      if (agent->ssl_status)
        continue;

      vector<vector<string > > lines;
      agent->parse(&lines);
      if (lines.size() == 0)
        continue;

      if (agent->proto == Agent::UNKNOWN) {
        const vector<string> &words = lines[0];
        if (words.size()) {
          const string &word = words[0];
          if (word == "GET") {
            agent->proto = Agent::HTTP;
          } else {
            agent->proto = Agent::MORETP;
          }
        } else {
          agent->proto = Agent::MORETP;
        }
      }

      switch (agent->proto) {
      case Agent::MORETP:
        for (auto line : lines) {
          agent->command(line);
        }
        break;
      case Agent::HTTP:
        for (auto line : lines) {
          if (line.size() == 0) {
            vector<string> words;
            words.push_back("http");
            agent->command(words);
            agent->httpbuf.clear();
          } else {
            if (agent->httpbuf.size() >= 64) {
              agent->close();
            } else {
              agent->httpbuf.push_back(joinwords(line));
            }
          }
        }
        break;
      default:
        assert(0);
      }
    }
  }

}

}
