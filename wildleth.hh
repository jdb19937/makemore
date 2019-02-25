#ifndef __MAKEMORE_WILDLETH_HH__
#define __MAKEMORE_WILDLETH_HH__ 1

#include <vector>
#include <string>
#include <map>

#include "hashbag.hh"
#include "shibboleth.hh"

namespace makemore {

struct Wildleth {
  struct Entry {
    Hashbag ctx;
    Hashbag tmp;

    bool consec_prev, consec_next;
    bool is_head, is_rear;
    uint32_t _pad;
  };

  std::vector<Entry> map;

  void clear() {
    map.clear();
  }

  Entry *wild_head() {
    if (map.size() < 1)
      return NULL;
    Entry *head = &*map.begin();
    if (!head->is_head)
      return NULL;
    return head;
  }

  Entry *wild_rear() {
    if (map.size() < 1)
      return NULL;
    Entry *rear = &*map.rbegin();
    if (!rear->is_rear)
      return NULL;
    return rear;
  }

  void parse(const std::vector<std::string> &);
  void parse(const std::string &);
  void parse(const char *str);

  void mutate(Shibboleth *shib);

  void save(FILE *fp) const;
  void load(FILE *fp);
};
 
}

#endif
