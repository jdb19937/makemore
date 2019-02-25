#ifndef __MAKEMORE_WILDMORE_HH__
#define __MAKEMORE_WILDMORE_HH__ 1

#include <vector>
#include <string>
#include <map>

#include "hashbag.hh"
#include "wildleth.hh"
#include "shibbomore.hh"

namespace makemore {

struct Wildmore {
  uint8_t front3;
  uint8_t _pad[7];
  Wildleth backwild;

  Wildmore() {
    front3 = 0;
  }

  void clear() {
    front3 = 0;
    backwild.clear();
  }

  void parse(const std::vector<std::string> &);
  void parse(const std::string &);
  void parse(const char *str);

  void mutate(Shibbomore *shib);

  void save(FILE *fp) const;
  void load(FILE *fp);
};
 
}

#endif
