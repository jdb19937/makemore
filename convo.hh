#ifndef __MAKEMORE_CONVO_HH__
#define __MAKEMORE_CONVO_HH__ 1

#include "shibbomore.hh"

namespace makemore {

#define DECAY 0.5

struct Convo {
  Shibbomore reqhist;
  Shibbomore rsphist;
  Shibbomore req;

  Convo() {

  }

  void build(const std::string &str, double decay = DECAY);
  void build(const std::vector<std::string> &strparts, double decay = DECAY);
  void build(const std::vector<Shibbomore> &shibs, double decay = DECAY);

  void add(const std::string &rspstr, const std::string &reqstr, double decay = DECAY);
  void add(const Shibbomore &rspshib, const Shibbomore &reqshib, double decay = DECAY);
};

#if 0
struct Wideconvo {
  Convo convo[2];

  Wideconvo() {
  }

  Wideconvo(const std::vector<std::string> &strparts, double decay = DECAY) {
    build(strparts, decay);
  }

  void build(const std::vector<std::string> &strparts, double decay = DECAY);
  void build(const std::vector<Shibboleth> &hshibs, const std::vector<Shibboleth> &tshibs, double decay = DECAY);

  void add(const std::string &rspstr, const std::string &reqstr, double decay = DECAY);
};
#endif

#undef DECAY

}
#endif
