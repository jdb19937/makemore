#ifndef __MAKEMORE_PROJECT_HH__
#define __MAKEMORE_PROJECT_HH__ 1

#include <stdio.h>

#include <string>

#include "layout.hh"
#include "topology.hh"
#include "multitron.hh"

struct Project {
  Project(const char *_dir, unsigned int _mbn);
  virtual ~Project();

  unsigned int mbn;

  std::string dir;
  Layout *contextlay, *controlslay, *outputlay;

  typedef enum {
    CONTEXT_SOURCE_UNKNOWN,
    CONTEXT_SOURCE_TRAINING,
    CONTEXT_SOURCE_STDIN
  } ContextSource;

  typedef enum {
    CONTROL_SOURCE_UNKNOWN,
    CONTROL_SOURCE_CENTER,
    CONTROL_SOURCE_TRAINING,
    CONTROL_SOURCE_RANDOM,
    CONTROL_SOURCE_STDIN
  } ControlSource;

  typedef enum {
    OUTPUT_FORMAT_UNKNOWN,
    OUTPUT_FORMAT_RAW,
    OUTPUT_FORMAT_PPM
  } OutputFormat;

  virtual void learn(
    FILE *infp,
    double nu,
    double dpres, double fpres, double cpres, double zpres,
    double fcut, double dcut,
    unsigned int i
  ) = 0;

  virtual void generate(
    FILE *infp,
    double dev, int fidelity
  ) = 0;

  virtual const uint8_t *output() const = 0;

  virtual void write_ppm(FILE *fp = stdout) {
    assert(0);
  }

  virtual void load() {
    fprintf(stderr, "loading, nothing to do\n");
  }

  virtual void save() {
    fprintf(stderr, "saving, nothing to do\n");
  }

  virtual void report(const char *prog, unsigned int i) {
    fprintf(stderr, "%s %s i=%u\n", prog, dir.c_str(), i);
  }
};

extern Project *open_project(const char *dir, unsigned int mbn);

struct SimpleProject : Project {
  SimpleProject(const char *_dir, unsigned int _mbn);
  virtual ~SimpleProject();

  bool genwoke;

  Layout *sampleslay;

  Topology *enctop, *gentop, *distop;
  Multitron *enctron, *gentron, *distron;

  Tron *encpasstron, *encgentron;
  Tron *genpasstron, *gendistron;


  double *genin, *encin, *gentgt;
  double *samplesbuf, *contextbuf, *controlbuf, *outputbuf;
  double *genoutbuf;
  uint8_t *bcontextbuf, *boutputbuf;

  double *disin, *distgt, *distgtbuf;
  double *enctgt;

  unsigned int labn, dim;

  virtual void learn(
    FILE *infp,
    double nu,
    double dpres, double fpres, double cpres, double zpres,
    double fcut, double dcut,
    unsigned int i
  );

  virtual void generate(
    FILE *infp,
    double dev, int fidelity
  );

  virtual void write_ppm(FILE *fp = stdout);

  virtual const uint8_t *output() const {
    return boutputbuf;
  }

  virtual void load();
  virtual void save();
  virtual void report(const char *prog, unsigned int i);
};


struct ZoomProject : Project {
  ZoomProject(const char *_dir, unsigned int _mbn);
  virtual ~ZoomProject();

  Topology *enctop, *gentop, *distop;
  Multitron *enctron, *gentron, *distron;

  Tron *encpasstron, *encgentron;
  Tron *genpasstron, *gendistron;

  Layout *lofreqlay, *hifreqlay, *attrslay;


  double *genin, *encin, *gentgt;
  double *lofreqbuf, *hifreqbuf, *controlbuf, *outputbuf, *contextbuf;
  uint8_t *bcontextbuf, *boutputbuf;

  double *disin, *distgt, *distgtbuf;
  double *enctgt;

  unsigned int labn, dim;

  virtual void learn(
    FILE *infp,
    double nu,
    double dpres, double fpres, double cpres, double zpres,
    double fcut, double dcut,
    unsigned int i
  );

  virtual void generate(
    FILE *infp,
    double dev, int fidelity
  );

  virtual void write_ppm(FILE *fp = stdout);

  virtual const uint8_t *output() const {
    return boutputbuf;
  }

  virtual void load();
  virtual void save();
  virtual void report(const char *prog, unsigned int i);
};

#endif

