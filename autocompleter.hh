#ifndef __MAKEMORE_AUTOCOMPLETER_HH__
#define __MAKEMORE_AUTOCOMPLETER_HH__ 1

#include <assert.h>

#include <map>
#include <string>
#include <vector>
#include <set>

namespace makemore {

struct Autocompleter {
  unsigned int min_prefix_length;
  std::map<std::string, std::set<std::string> > dict;

  Autocompleter() {
    min_prefix_length = 1;
  }

  ~Autocompleter() {
  }

  void add(const std::string &x) {
    for (unsigned int i = min_prefix_length; i < x.length(); ++i) {
      std::string xpre(x.c_str(), i);
      dict[xpre].insert(x);
    }
  }

  void find(const std::string &xpre, std::vector<std::string> *rp) {
    auto rsi = dict.find(xpre);
    if (rsi == dict.end())
      return;
    const std::set<std::string> &rs = rsi->second;

    for (auto ri = rs.begin(); ri != rs.end(); ++ri) {
      rp->push_back(*ri);
    }
  }
};

}
#endif
