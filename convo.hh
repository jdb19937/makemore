#ifndef __MAKEMORE_CONVO_HH__
#define __MAKEMORE_CONVO_HH__ 1

#include "shibbomore.hh"

namespace makemore {

#define DECAY 0.5

struct Convo {
  Shibbomore history;
  Shibbomore current;

  Convo() {

  }

  void build(const std::string &str, double decay = DECAY);
  void build(const std::vector<std::string> &words, double decay = DECAY);
  void build(const std::vector<std::vector<std::string> > &thread, double decay = DECAY);
  void build(const std::vector<Shibbomore> &shmores, double decay = DECAY);

  void add(const std::string &str, double decay = DECAY);
  void add(const Shibbomore &shmore, double decay = DECAY);
};

#undef DECAY

}
#endif
