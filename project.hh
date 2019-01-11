#ifndef __MAKEMORE_PROJECT_HH__
#define __MAKEMORE_PROJECT_HH__ 1

#include <stdio.h>

#include <string>
#include <map>

#include "layout.hh"
#include "topology.hh"
#include "multitron.hh"

struct Project {
  Project(const char *_dir, unsigned int _mbn, std::map<std::string,std::string> *_config = NULL);
  virtual ~Project();

  unsigned int mbn;
  std::map<std::string, std::string> *config;

  std::string dir;
  Layout *contextlay, *controlslay, *outputlay, *targetlay, *adjustlay;
  double *outputbuf, *contextbuf, *dcontrolbuf, *controlbuf, *adjustbuf, *targetbuf;
  uint8_t *boutputbuf, *bcontextbuf, *bcontrolbuf, *badjustbuf, *btargetbuf;

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
    uint8_t *hyper = NULL
  ) = 0;
  virtual void dotarget1() { assert(0); }
  virtual void dotarget2() { assert(0); }
  virtual void readjust() { assert(0); }

  virtual void regenerate(
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

  virtual bool loadcontext(FILE *);
  virtual bool loadcontrols(FILE *);
  virtual bool loadadjust(FILE *);
  virtual void randcontrols(double dev);
  virtual bool loadtarget(FILE *);
  virtual void nullcontrols() { randcontrols(0); }
  virtual void nulladjust();
  virtual bool loadbatch(FILE *);

  virtual void encodeout();
  virtual void encodetgt();
  virtual void encodeadj();
  virtual void encodectrl();
  virtual void encodectx();
};

extern Project *open_project(const char *dir, unsigned int mbn);

struct ImageProject : Project {
  ImageProject(const char *_dir, unsigned int _mbn, std::map<std::string,std::string> *);
  virtual ~ImageProject();

  bool genwoke;

  Layout *sampleslay;
  double *samplesbuf;

  Topology *enctop, *gentop, *distop;
  Multitron *enctron, *gentron, *distron;

  Tron *encpasstron, *encgentron;
  Tron *genpasstron, *gendistron;


  double *genin, *encin, *gentgt, *genfin;
  double *genoutbuf;

  double *disin, *distgt, *distgtbuf;
  double *enctgt;

  virtual void learn(
    FILE *infp,
    double nu,
    double dpres, double fpres, double cpres, double zpres,
    double fcut, double dcut,
    unsigned int i
  );

  virtual void generate(
    uint8_t *hyper = NULL
  );

  virtual void regenerate(
  );

  virtual void write_ppm(FILE *fp = stdout);

  virtual const uint8_t *output() const {
    return boutputbuf;
  }

  virtual void separate();
  virtual void reconstruct();

  virtual void load();
  virtual void save();
  virtual void report(const char *prog, unsigned int i);
};


struct ZoomProject : ImageProject {
  ZoomProject(const char *_dir, unsigned int _mbn, std::map<std::string,std::string> * = NULL);
  virtual ~ZoomProject();

  Layout *lofreqlay, *attrslay;
  double *lofreqbuf;

  virtual void separate();
  virtual void reconstruct();
  virtual void report(const char *prog, unsigned int i);
};

struct PipelineProject : Project {
  std::vector<Project*> stages;

  PipelineProject(const char *_dir, unsigned int _mbn, std::map<std::string,std::string> * = NULL);
  virtual ~PipelineProject();
  
  virtual void learn(
    FILE *infp,
    double nu,
    double dpres, double fpres, double cpres, double zpres,
    double fcut, double dcut,
    unsigned int i
  ) {
    assert(0);
  }

  virtual void generate(
    uint8_t *hyper = NULL
  );

  virtual void dotarget1();
  virtual void dotarget2();
  virtual void readjust(
  );


  virtual void regenerate(
  ) {
    assert(0);
  }

  virtual const uint8_t *output() const;

  virtual void write_ppm(FILE *fp = stdout);

  virtual void load();
  virtual void save();

  virtual void report(const char *prog, unsigned int i) {
    fprintf(stderr, "[PipelineProject] %s %s i=%u\n", prog, dir.c_str(), i);
  }

  virtual void nulladjust();
};

#endif

