#define __MAKEMORE_SHIBBOMORE_CC__ 1
#include <vector>
#include <string>
#include <algorithm>

#include "shibbomore.hh"
#include "shibboleth.hh"
#include "strutils.hh"
#include "closest.hh"
#include "vocab.hh"

#include "sha256.c"

namespace makemore {

using namespace std;

void Shibbomore::encode(const char *str) {
  clear();

  vector<string> vec;
  split(str, ' ', &vec);
  encode(vec);
}

void Shibbomore::encode(const vector<string>& vec) {
  vector<string> frontvec = vec;

  vector<string> backvec;
  if (frontvec.size() > 3) {
    unsigned int n = frontvec.size();
    backvec.resize(n - 3);
    for (unsigned int j = 3; j < n; ++j)
      backvec[j - 3] = frontvec[j];
    frontvec.resize(3);
  }

  if (frontvec.size() > 0) front[0].add(frontvec[0].c_str());
  if (frontvec.size() > 1) front[1].add(frontvec[1].c_str());
  if (frontvec.size() > 2) front[2].add(frontvec[2].c_str());

  backleth.encode(backvec);
}

std::string Shibbomore::decode(const Vocab &vocab) const {
  if (front[0].empty())
    return "";
  string outstr = vocab.closest(front[0]);

  const char *w;
  if (!front[1].empty()) {
    outstr += " ";
    outstr += vocab.closest(front[1]);

    if (!front[2].empty()) {
      outstr += " ";
      outstr += vocab.closest(front[2]);
    }
  }

  std::string backstr = backleth.decode(vocab);
  if (backstr != "") {
    outstr += " ";
    outstr += backstr;
  }

  return outstr;
}

void Shibbomore::save(FILE *fp) const {
  front[0].save(fp);
  front[1].save(fp);
  front[2].save(fp);
  backleth.save(fp);
}

void Shibbomore::load(FILE *fp) {
  front[0].load(fp);
  front[1].load(fp);
  front[2].load(fp);
  backleth.load(fp);
}

}
