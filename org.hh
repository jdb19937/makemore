#ifndef __MAKEMORE_ORG_HH__
#define __MAKEMORE_ORG_HH__ 1

#include <vector>
#include <algorithm>

#include "parson.hh"
#include "pipeline.hh"

namespace makemore {

struct Org {
  unsigned int n;
  std::vector<Parson*> member;

  Org();
  ~Org();

  struct CmpCenterV {
    bool operator()(const Parson *p, const Parson *q) {
      return (p->centerv() < q->centerv());
    }
  };
  struct CmpCenterS {
    bool operator()(const Parson *p, const Parson *q) {
      return (p->centers() < q->centers());
    }
  };
  struct CmpCenterH {
    bool operator()(const Parson *p, const Parson *q) {
      return (p->centerh() < q->centerh());
    }
  };

  struct CmpError2 {
    bool operator()(const Parson *p, const Parson *q) {
      return (p->error2() < q->error2());
    }
  };

  void clear();
  void add(Parson *);
  void pick(class Zone *zone, unsigned int k);

  void sort_centerv() {
    std::sort(member.begin(), member.end(), CmpCenterV());
  }
  void sort_centers() {
    std::sort(member.begin(), member.end(), CmpCenterS());
  }
  void sort_centerh() {
    std::sort(member.begin(), member.end(), CmpCenterH());
  }
  void sort_error2() {
    std::sort(member.begin(), member.end(), CmpError2());
  }
};

}

#endif
