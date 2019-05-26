#ifndef __MAKEMORE_PARSON_HH__
#define __MAKEMORE_PARSON_HH__ 1

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <sys/time.h>

#include <math.h>

#include <string>
#include <map>

#include "hashbag.hh"

namespace makemore {


struct Parson {
  static bool valid_nom(const char *);
  static bool valid_nom(const std::string &s) { return valid_nom(s.c_str()); }
  static bool valid_tag(const char *);
  static bool valid_tag(const std::string &s) { return valid_tag(s.c_str()); }
  static uint64_t hash_nom(const char *nom, unsigned int variant = 0);
  static bool female_nom(const char *);
  static std::string bread_nom(const char *nom0, const char *nom1, uint8_t);
  static void paren_noms(const char *, char *, char *);
  static std::string gen_nom(bool *gender);
  static std::string gen_nom(bool gender);
  static std::string gen_nom();

  const static unsigned int nfrens = 16;
  const static unsigned int ntags = 8;
  const static unsigned int dim = 64;
  const static unsigned int ncontrols = 1024;
  const static unsigned int bufsize = 2048;
  const static unsigned int nbriefs = 8;
  const static unsigned int briefsize = 256;
  typedef char Nom[32];
  typedef char Tag[32];
  typedef char Brief[briefsize];

  Parson() {
    memset(this, 0, sizeof(Parson));
  }

  Parson(const char *_nom) {
    memset(this, 0, sizeof(Parson));
    assert(valid_nom(_nom));
    strcpy(nom, _nom);
  }

  ~Parson() {

  }

  // 27 * 32
  Nom nom;
  Nom frens[nfrens];
  Nom parens[2];
  Tag tags[ntags];

  // 16
  uint32_t created;
  uint32_t revised;
  uint32_t creator;
  uint32_t revisor;

  // 16
  uint32_t visited;
  uint32_t visits;
  double last_activity;

  // 8
  uint32_t generated;
  uint8_t target_lock;
  uint8_t control_lock;
  uint8_t _pad[2];

  // 8
  uint64_t cents;

  // 8
  uint8_t briefptr;
  uint8_t padddy[7];

  // 64
  uint8_t pass[32];
  char salt[32];

  // 808
  uint8_t _fill[808];

  // 256
  char srcfn[256];

  // 8192
  double controls[ncontrols];

  // 1536
  double sketch[192];

  // 32
  double angle;
  double stretch; 
  double skew;
  double recon_err;
  double pmark_err;
  double qmark_err;
  double rmark_err;

  uint8_t pad1[2504];
  uint8_t pad2[4096];

  // 64 * 64 * 3
  uint8_t partrait[dim * dim * 3];

  // 2048
  Brief briefs[nbriefs];

  char *popbrief() {
    assert(briefptr < nbriefs);

    char *brief = briefs[briefptr];
    if (!*brief)
      return NULL;

    brief[briefsize - 1] = 0;

    briefptr = (briefptr + 1) % nbriefs;
    return brief;
  }

  void pushbrief(const std::string &briefstr) {
    assert(briefptr < nbriefs);
    uint8_t newbriefptr = briefptr ? (briefptr - 1) : (nbriefs - 1);
    char *brief = briefs[newbriefptr];

    unsigned int briefstrlen = briefstr.length();
    if (briefstrlen < briefsize) {
      memcpy(brief, briefstr.c_str(), briefstrlen + 1);
    } else {
      unsigned int off = briefstrlen - Parson::briefsize + 4;
      while (off < briefstrlen && briefstr[off] != ' ')
        ++off;
      while (off < briefstrlen && briefstr[off] == ' ')
        ++off;

      std::string truncstr = "... ";
      truncstr += std::string(briefstr.c_str() + off);
      assert(truncstr.length() < Parson::briefsize);

      memcpy(brief, truncstr.c_str(), truncstr.length() + 1);
    }

    briefptr = newbriefptr;
  }

  void set_pass(const std::string &password);
  void clear_pass() {
    set_pass("");
  }
  bool check_pass(const std::string &password) const;

  bool has_pass() const {
    for (unsigned int i = 0; i < sizeof(pass); ++i)
      if (pass[i])
        return true;
    return false;
  }

  bool exists() {
    return (nom[0] != 0);
  }

  void set_parens(const char *anom, const char *bnom);

  bool has_fren(const char *nom);
  void add_fren(const char *fnom);
  void del_fren(const char *fnom);

  bool has_tag(const char *tag);
  void add_tag(const char *tag);
  void del_tag(const char *tag);


  bool fraternal(const Parson *p) const {
    if (*parens[0] && *p->parens[0] && !strcmp(parens[0], p->parens[0]))
      return true;
    if (*parens[1] && *p->parens[0] && !strcmp(parens[1], p->parens[0]))
      return true;
    if (*parens[0] && *p->parens[1] && !strcmp(parens[0], p->parens[1]))
      return true;
    if (*parens[1] && *p->parens[1] && !strcmp(parens[1], p->parens[1]))
      return true;
    return false;
  }

  double activity() const {
    time_t now = time(NULL);
    double dt = (double)(now - visited) / (double)(1 << 20);
    if (dt < 0)
      dt = 0;
    return (last_activity * exp(-dt));
  }

  void visit(unsigned int count = 1) {
    time_t now = time(NULL);
    double dt = (double)(now - visited) / (double)(1 << 20);

    last_activity *= exp(-dt);
    visited = now;

    last_activity += (double)count;
    visits += count;
  }

  void bagtags(Hashbag *h) const {
    h->clear();
    for (unsigned int i = 0; i < ntags; ++i)
      if (*tags[i])
        h->add(tags[i]);
  }

#if 0
  void generate(class Pipeline *pipe, long min_age = 0);
  void _to_pipe(class Pipeline *pipe, unsigned int mbi, bool ex = false);
  void _from_pipe(class Pipeline *pipe, unsigned int mbi);
#endif

  void paste_partrait(class PPM *ppm, unsigned int x0 = 0, unsigned int y0 = 0);

  bool load(FILE *fp);
  void save(FILE *fp);

  double centerh() const;
  double centers() const;
  double centerv() const;
  double error2() const;
};


}

#endif
