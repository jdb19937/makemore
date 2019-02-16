#ifndef __MAKEMORE_RULE_HH__
#define __MAKEMORE_RULE_HH__ 1

#include "shibboleth.hh"
#include "wildmap.hh"
#include "vocab.hh"

#include <string>
#include <map>

namespace makemore {

struct Rule {
  Shibboleth req, mem, aux;
  Shibboleth cmd, out, nem, bux;
  Shibboleth reg1, reg2;

  Wildmap reqwild, memwild, auxwild;
  unsigned int multiplicity;
  bool prepared;

  Rule() {
    prepared = false;
    multiplicity = 1;
  }

  void copy(const Rule &r) {
    req = r.req;
    mem = r.mem;
    aux = r.aux;
    cmd = r.cmd;
    out = r.out;
    nem = r.nem;
    bux = r.bux;
    reg1 = r.reg1;
    reg2 = r.reg2;
    reqwild = r.reqwild;
    memwild = r.memwild;
    auxwild = r.auxwild;
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
    auxwild.mutate(&aux);
    prepared = true;
  }

  void load(FILE *fp);
  void save(FILE *fp) const;
};

}

#endif
