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
  vector<Shibbomore> shibs;

  unsigned int n = strparts.size();
  shibs.resize(n);
  for (unsigned int i = 0; i < n; ++i)
    shibs[i].encode(strparts[i].c_str());

  build(shibs, decay);
}

void Convo::build(const vector<Shibbomore> &shibs, double decay) {
  reqhist.clear();
  rsphist.clear();
  req.clear();

  unsigned int n = shibs.size();
  if (n == 0)
    return;

  req = shibs[n - 1];
  for (int i = (int)n - 2; i >= 0; --i) {
    Shibbomore *x = ((n - i) % 2) ? &reqhist : &rsphist;
    x->mul(decay);
    x->add(shibs[i]);
  }
}

void Convo::add(const std::string &rspstr, const std::string &reqstr, double decay) {
  Shibbomore rspshib, reqshib;
  rspshib.encode(rspstr.c_str());
  reqshib.encode(reqstr.c_str());
  add(rspshib, reqshib);
}

void Convo::add(const Shibbomore &rspshib, const Shibbomore &reqshib, double decay) {
  rsphist.mul(decay);
  rsphist.add(rspshib);

  reqhist.mul(decay);
  reqhist.add(req);

  req = reqshib;
}



}
