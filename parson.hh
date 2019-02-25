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
  static bool valid_tag(const char *);
  static uint64_t hash_nom(const char *nom, unsigned int variant = 0);
  static bool female_nom(const char *);
  static std::string bread_nom(const char *nom0, const char *nom1, uint8_t);
  static void paren_noms(const char *, char *, char *);

  const static unsigned int nfrens = 16;
  const static unsigned int ntags = 8;
  const static unsigned int dim = 64;
  const static unsigned int ncontrols = 1920;
  const static unsigned int bufsize = 1024;
  typedef char Nom[32];
  typedef char Tag[32];

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

  // 1920
  uint8_t controls[ncontrols];

  // 64 * 64 * 3
  uint8_t target[dim * dim * 3];

  // 64 * 64 * 3
  uint8_t partrait[dim * dim * 3];

  // 1024
  char buffer[bufsize];

  char *popbuf(unsigned int *lenp = NULL);
  void pushbuf(const char *cmd, unsigned int n);
  void pushbuf(const char *cmd);

  bool exists() {
    return (nom[0] != 0);
  }

  void initialize(const char *_nom, double mean, double dev);

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

  void generate(class Pipeline *pipe, long min_age = 0);

  void bagtags(Hashbag *h) {
    h->clear();
    for (unsigned int i = 0; i < ntags; ++i)
      if (*tags[i])
        h->add(tags[i]);
  }

  void _to_pipe(class Pipeline *pipe, unsigned int mbi);
  void _from_pipe(class Pipeline *pipe, unsigned int mbi);

  void paste_partrait(class PPM *ppm, unsigned int x0 = 0, unsigned int y0 = 0);
  void paste_target(class PPM *ppm, unsigned int x0 = 0, unsigned int y0 = 0);

  bool load(FILE *fp);
  void save(FILE *fp);

  double centerh() const;
  double centers() const;
  double centerv() const;
  double error2() const;
};

}

#endif
