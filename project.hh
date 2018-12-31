#ifndef __MAKEMORE_PROJECT_HH__
#define __MAKEMORE_PROJECT_HH__ 1

#include <stdio.h>

#include <string>

#include "dataset.hh"
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
    ControlSource control_source,
    double nu,
    unsigned int i,
    bool seqbatch = false
  ) = 0;

  virtual void generate(
    ContextSource context_source,
    ControlSource control_source,
    bool seqbatch = false
  ) = 0;

  virtual const double *output() const = 0;

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

  Layout *sampleslay;

  Topology *enctop, *gentop, *distop;
  Multitron *enctron, *gentron, *distron;

  Tron *encpasstron, *encgentron, *encdistron;

  Dataset *samples, *context;


  double *genin, *encin, *gentgt;
  double *samplesbuf, *contextbuf, *controlbuf, *outputbuf;
  unsigned int *mbbuf;

  virtual void learn(
    ControlSource control_source,
    double nu,
    unsigned int i,
    bool seqbatch = false
  );

  virtual void generate(
    ContextSource context_source,
    ControlSource control_source,
    bool seqbatch = false
  );

  virtual void write_ppm(FILE *fp = stdout);

  virtual const double *output() const {
    return outputbuf;
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

  Tron *encpasstron, *encgentron, *encdistron;

  Layout *lofreqlay, *hifreqlay, *attrslay;
  Dataset *lofreq, *hifreq, *attrs;


  double *genin, *encin, *gentgt;
  double *attrsbuf, *lofreqbuf, *hifreqbuf, *controlbuf, *outputbuf;
  unsigned int *mbbuf;

  virtual void learn(
    ControlSource control_source,
    double nu,
    unsigned int i,
    bool seqbatch = false
  );

  virtual void generate(
    ContextSource context_source,
    ControlSource control_source,
    bool seqbatch = false
  );

  virtual void write_ppm(FILE *fp = stdout);

  virtual const double *output() const {
    return outputbuf;
  }

  virtual void load();
  virtual void save();
  virtual void report(const char *prog, unsigned int i);
};

struct PipelineProject : Project {
  PipelineProject(const char *_dir, unsigned int _mbn);
  virtual ~PipelineProject();

  std::vector<Project*> projects;
  Project *p0, *p1;

  virtual void learn(
    ControlSource control_source,
    double nu,
    unsigned int i
  ) {
    fprintf(stderr, "can't learn pipeline\n");
  }

  virtual void generate(
    ContextSource context_source,
    ControlSource control_source
  );

  virtual void write_ppm(FILE *fp = stdout);
  virtual const double *output() const;
  virtual void load();

  virtual void save() {
    fprintf(stderr, "won't save pipeline\n");
  }
};




#endif

