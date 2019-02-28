#define __MAKEMORE_CONVO_CC__ 1

#include <vector>
#include <string>

#include "convo.hh"
#include "strutils.hh"

namespace makemore {

using namespace std;

void Convo::build(const std::string &str, double decay) {
  vector<string> strparts;
  split(str.c_str(), ',', &strparts);
  build(strparts, decay);
}

void Convo::build(const vector<string> &strparts, double decay) {
  vector<Shibbomore> shmores;

  unsigned int n = strparts.size();
  shmores.resize(n);
  for (unsigned int i = 0; i < n; ++i)
    shmores[i].encode(strparts[i].c_str());

  build(shmores, decay);
}

void Convo::build(const vector<Shibbomore> &shmores, double decay) {
  history.clear();
  current.clear();

  for (unsigned int i = 0, n = shmores.size(); i < n; ++i)
    add(shmores[i], decay);
}

void Convo::add(const std::string &str, double decay) {
  Shibbomore shmore;
  shmore.encode(str.c_str());
  add(shmore, decay);
}

void Convo::add(const Shibbomore &shmore, double decay) {
  history.mul(decay);
  history.add(shmore);
  current = shmore;
}


}
