#ifndef __MAKEMORE_RULE_HH__
#define __MAKEMORE_RULE_HH__ 1

#include "shibboleth.hh"
#include "wildmap.hh"
#include "vocab.hh"

#include <string>
#include <map>

namespace makemore {

struct Rule {
  Shibboleth req, mem;
  Shibboleth cmd, out, nem, buf[4];
  Wildmap reqwild, memwild;
  bool prepared;

  Rule() {
    prepared = false;
  }

  void copy(const Rule &r) {
    req = r.req;
    mem = r.mem;
    cmd = r.cmd;
    out = r.out;
    nem = r.nem;
    buf[0] = r.buf[0];
    buf[1] = r.buf[1];
    buf[2] = r.buf[2];
    buf[3] = r.buf[3];
    reqwild = r.reqwild;
    memwild = r.memwild;
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
    reqwild.mutate(&req);
    memwild.mutate(&mem);
    prepared = true;
  }
};

}

#endif
