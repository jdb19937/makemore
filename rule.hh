#ifndef __MAKEMORE_RULE_HH__
#define __MAKEMORE_RULE_HH__ 1

#include <assert.h>

#include <vector>
#include <string>
#include <map>

#include "shibbomore.hh"
#include "wildmore.hh"
#include "vocab.hh"


namespace makemore {

struct Rule {
  Hashbag tags;
  std::vector<Shibbomore> req, rsp;
  std::vector<Wildmore> wild;
  bool prepared;

  Rule() {
    prepared = false;
  }

  void copy(const Rule &r) {
    tags = r.tags;
    req = r.req;
    rsp = r.rsp;
    wild = r.wild;
    prepared = r.prepared;
  }

  Rule(const Rule &r) {
    copy(r);
  }

  Rule &operator = (const Rule &r) {
    copy(r);
    return *this;
  }

  unsigned int parse(const char *line);

  void prepare() {
    assert(!prepared);

    unsigned int nreq = req.size();
    assert(wild.size() == nreq);

    for (unsigned int i = 0; i < nreq; ++i)
      wild[i].mutate(&req[i]);
    prepared = true;
  }

  void load(FILE *fp);
  void save(FILE *fp) const;
};

}

#endif
