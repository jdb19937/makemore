#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <string.h>

#include <string>

#include "cudamem.hh"
#include "pipeline.hh"
#include "parson.hh"
#include "ppm.hh"

#include "sha256.c"

#define ensure(x) do { \
  if (!(x)) { \
    fprintf(stderr, "closing (%s)\n", #x); \
    return; \
  } \
} while (0)

using namespace std;

static inline double realtime() {
  clock_t c = clock();
  return ((double)c / (double)CLOCKS_PER_SEC);
}

Pipeline *open_pipeline() {
  Pipeline *pipe = new Pipeline(1);
  pipe->add_stage(new Project("test8.proj", 1));
  pipe->add_stage(new Project("test16.proj", 1));
  pipe->add_stage(new Project("test32.proj", 1));
  pipe->add_stage(new Project("test64.proj", 1));
  return pipe;
}

ParsonDB *open_parsons() {
  return new ParsonDB("parsons.dat");
}

void makeparson(FILE *learnfp, Pipeline *pipe, Parson *parson);

void makeparens(FILE *learnfp, Pipeline *pipe, ParsonDB *parsons, Parson *parson) {
  Parson *p1 = parsons->find(parson->parens[0]);
  Parson *p2 = parsons->find(parson->parens[1]);

  if (p1->revised && p2->revised)
    return;
  bool update_p1 = !p1->revised;
  bool update_p2 = !p2->revised;

  if (update_p1)
    p1->attrs[39] = 0;
  if (update_p2)
    p2->attrs[39] = 0;

  makeparson(learnfp, pipe, p1);
  makeparson(learnfp, pipe, p2);

  p1->add_fren(p2->nom);
  p1->add_fren(parson->nom);
  p2->add_fren(p1->nom);
  p2->add_fren(parson->nom);

  p1->target_lock = 0;
  p1->control_lock = -1;
  p2->target_lock = 0;
  p2->control_lock = -1;

  unsigned int js = 4;
  assert(Parson::ncontrols % js == 0);

  for (unsigned int i = 0; i < 39; i++) {
    if (i == 20)
      continue;
    if (randrange(0.0, 1.0) < 0.5) {
      if (update_p1)
        p1->attrs[i] = parson->attrs[i];
    } else {
      if (update_p2)
        p2->attrs[i] = parson->attrs[i];
    }
  }
  if (update_p1)
    p1->attrs[39] = 0;
  if (update_p2)
    p2->attrs[39] = 0;

  for (unsigned int i = 40; i < 52; i += 3) {
    // double blend = (randuint() % 2 : 1.0 : 0.0; //randrange(0.0, 1.0);
    double blend = sigmoid(randgauss() * 5);
    if (update_p1) {
      p1->attrs[i+0] = blend * parson->attrs[i+0] + (1-blend) * p1->attrs[i+0];
      p1->attrs[i+1] = blend * parson->attrs[i+1] + (1-blend) * p1->attrs[i+1];
      p1->attrs[i+2] = blend * parson->attrs[i+2] + (1-blend) * p1->attrs[i+2];
    }
    if (update_p2) {
      p2->attrs[i+0] = (1-blend) * parson->attrs[i+0] + blend * p2->attrs[i+0];
      p2->attrs[i+1] = (1-blend) * parson->attrs[i+1] + blend * p2->attrs[i+1];
      p2->attrs[i+2] = (1-blend) * parson->attrs[i+2] + blend * p2->attrs[i+2];
    }
  }

  // tags
  assert(Parson::nattrs == 72);
  for (unsigned int i = 52; i < 68; ++i) {
    if (update_p1)
      p1->attrs[i] = 128;
    if (update_p2)
      p2->attrs[i] = 128;
  }

  if (update_p1)
    p1->attrs[68] = 255;
  if (update_p2)
    p2->attrs[68] = 255;
  for (unsigned int i = 69; i < 72; ++i) {
    if (update_p1)
      p1->attrs[i] = 0;
    if (update_p2)
      p2->attrs[i] = 0;
  }

  for (unsigned int i = 0; i < Parson::ncontrols; i += js) {
    // if (randrange(0,1) < 0.25)
    //   continue;

    if (randrange(0.0, 1.0) < 0.5) {
      if (update_p1) {
        for (unsigned int q = i, qn = i + js; q < qn; ++q) {
          p1->controls[q] = parson->controls[q];
        }
      }
    } else {
      if (update_p2) {
        for (unsigned int q = i, qn = i + js; q < qn; ++q) {
          p2->controls[q] = parson->controls[q];
        }
      }
    }
  }
}

void makebread(FILE *learnfp, Pipeline *pipe, Parson *p1, Parson *p2, Parson *parson, uint8_t gender) {
  uint8_t *ctxbuf = new uint8_t[pipe->ctxlay->n];

  assert(0 == fseek(learnfp, 0, SEEK_END));
  off_t nlearn = ftell(learnfp);
  nlearn /= 12360;

  unsigned int j = 0;
  do {
    fseek(learnfp, ((parson->hash + j++) % nlearn) * 12360, SEEK_SET);
    pipe->load_ctx_bytes(learnfp);
    pipe->save_ctx_bytes(ctxbuf);
  } while (ctxbuf[20] != gender * 255 || ctxbuf[39] != 255);
  pipe->load_out_bytes(learnfp);
  pipe->ctrlock = 0;
  pipe->tgtlock = -1;
  pipe->reencode();

  for (unsigned int i = 0; i < Parson::nattrs; ++i) {
    double blend = randuint() % 2 ? 1.0 : 0.0;
    parson->attrs[i] = (uint8_t)(blend * p1->attrs[i] + (1.0 - blend) * p2->attrs[i]);
  }
  parson->attrs[20] = gender * 255;
  parson->attrs[39] = 255;

  for (unsigned int i = 40; i < 52; i += 3) {
    //double blend = randuint() % 2 : 1.0 : 0.0; //randrange(0.0, 1.0);
    double blend = sigmoid(randgauss() * 5);
    parson->attrs[i+0] = blend * p1->attrs[i+0] + (1 - blend) * p2->attrs[i+0];
    parson->attrs[i+1] = blend * p1->attrs[i+1] + (1 - blend) * p2->attrs[i+1];
    parson->attrs[i+2] = blend * p1->attrs[i+2] + (1 - blend) * p2->attrs[i+2];
  }

  // tags
  assert(Parson::nattrs == 72);
  for (unsigned int i = 52; i < 68; ++i) {
    parson->attrs[i] = 128;
  }
  parson->attrs[68] = 255;
  for (unsigned int i = 69; i < 72; ++i) {
    parson->attrs[i] = 0;
  }


  unsigned int js = 4;
  assert(Parson::ncontrols % js == 0);
  for (unsigned int i = 0; i < Parson::ncontrols; i += js) {
#if 0
    double cj = 0;
    double ck = 0;
    for (unsigned int q = i, qn = i + js; q < qn; ++q) {
      double cs = p1->controls[q];
      double ct = p2->controls[q];
      cj += 4.0 * (cs - 0.5) * (cs - 0.5);
      ck += 4.0 * (ct - 0.5) * (ct - 0.5);
    }
    cj /= (double)js;
    ck /= (double)js;
 
    double cjw = 1.0 - cj;
    double ckw = 1.0 - ck;
    double jprob = 0.5;
    if (cjw + ckw > 0) 
      jprob = cjw / (cjw + ckw);
#endif
    double jprob = 0.5;
    bool fromj = (randrange(0.0, 1.0) < jprob);

    if (fromj) {
      for (unsigned int q = i, qn = i + js; q < qn; ++q) {
        parson->controls[q] = p1->controls[q];
      }
    } else {
      for (unsigned int q = i, qn = i + js; q < qn; ++q) {
        parson->controls[q] = p2->controls[q];
      }
    }
  }

  parson->control_lock = -1;
  parson->target_lock = 0;
  parson->revised = time(NULL);
  parson->revisor = 0;

//  parson->add_fren(p1->nom);
//  parson->add_fren(p2->nom);
  parson->set_parens(p1->nom, p2->nom);

  p1->add_fren(parson->nom);
  p2->add_fren(parson->nom);

  delete[] ctxbuf;
}

void makeparson(FILE *learnfp, Pipeline *pipe, Parson *parson) {
  uint8_t young = parson->attrs[39];
  uint8_t gender = parson->attrs[20];
  if (parson->revised)
    return;

  assert(0 == fseek(learnfp, 0, SEEK_END));
  off_t nlearn = ftell(learnfp);
  nlearn /= 12360;

  uint8_t *ctxbuf0 = new uint8_t[pipe->ctxlay->n];
  double *ctrbuf0 = new double[pipe->ctrlay->n];
  unsigned int j = 0;
  do {
    fseek(learnfp, ((parson->hash + j++) % nlearn) * 12360, SEEK_SET);
    pipe->load_ctx_bytes(learnfp);
    pipe->save_ctx_bytes(ctxbuf0);
  } while (ctxbuf0[20] != gender || ctxbuf0[39] != young);

  pipe->load_out_bytes(learnfp);
  pipe->ctrlock = 0;
  pipe->tgtlock = -1;
  pipe->reencode();
  pipe->generate();
  pipe->tgtlock = 0;
  pipe->ctrlock = -1;
  memcpy(ctrbuf0, pipe->ctrbuf, pipe->ctrlay->n * sizeof(double));

#if 0
  uint8_t *ctxbuf1 = new uint8_t[pipe->ctxlay->n];
  double *ctrbuf1 = new double[pipe->ctrlay->n];
  do {
    fseek(learnfp, ((parson->hash * 31 + j++) % nlearn) * 12360, SEEK_SET);
    pipe->load_ctx_bytes(learnfp);
    pipe->save_ctx_bytes(ctxbuf1);
  } while (ctxbuf1[20] != ctxbuf0[20]);

  pipe->load_out_bytes(learnfp);
  pipe->ctrlock = 0;
  pipe->tgtlock = -1;
  pipe->reencode();
  pipe->generate();
  pipe->tgtlock = 0;
  pipe->ctrlock = -1;
  memcpy(ctrbuf1, pipe->ctrbuf, pipe->ctrlay->n * sizeof(double));

  double blend = 0.0;
  for (unsigned int i = 0; i < pipe->ctxlay->n; ++i)
    ctxbuf0[i] = blend * ctxbuf0[i] + (1.0 - blend) * ctxbuf1[i];
  for (unsigned int i = 0; i < pipe->ctrlay->n; ++i)
    ctrbuf0[i] = blend * ctrbuf0[i] + (1.0 - blend) * ctrbuf1[i];
#endif

  parson->control_lock = -1;
  parson->target_lock = 0;
  assert(sizeof(parson->controls) == pipe->ctrlay->n * sizeof(double));
  memcpy(parson->controls, ctrbuf0, sizeof(parson->controls));
  assert(sizeof(parson->attrs) == pipe->ctxlay->n);
  memcpy(parson->attrs, ctxbuf0, sizeof(parson->attrs));

//fprintf(stderr, "edgemean = %u, %u, %u\n", parson->attrs[40], parson->attrs[41], parson->attrs[42]);
//fprintf(stderr, "edgestddev = %u, %u, %u\n", parson->attrs[43], parson->attrs[44], parson->attrs[45]);

  if (strstr(parson->nom, "norm")) {
    for (unsigned int i = 0; i < parson->ncontrols; ++i) {
      parson->controls[i] = 0.5 + (parson->controls[i] - 0.5) * 0.2;
    }
  }

  if (strstr(parson->nom, "blue")) {
    double dl = 0, da, db;
    rgbtolab(0, 96, 150, &dl, &da, &db);
{
uint8_t r, g,b;
labtorgb(dl,da,db,&r,&g,&b);
fprintf(stderr, "blue=%u,%u,%u\n", r, g, b);
}
    if (dl > 0.99) { dl = 0.99; } if (dl < 0.0) { dl = 0.0; }
    if (da > 0.99) { da = 0.99; } if (da < 0.0) { da = 0.0; }
    if (db > 0.99) { db = 0.99; } if (db < 0.0) { db = 0.0; }
    parson->attrs[46] = (uint8_t)(dl * 255.0);
    parson->attrs[47] = (uint8_t)(da * 255.0);
    parson->attrs[48] = (uint8_t)(db * 255.0);
  }

  if (strstr(parson->nom, "green")) {
    double dl, da, db;
    rgbtolab(32, 133, 24, &dl, &da, &db);
    if (dl > 0.99) { dl = 0.99; } if (dl < 0.0) { dl = 0.0; }
    if (da > 0.99) { da = 0.99; } if (da < 0.0) { da = 0.0; }
    if (db > 0.99) { db = 0.99; } if (db < 0.0) { db = 0.0; }
    parson->attrs[46] = (uint8_t)(dl * 255.0);
    parson->attrs[47] = (uint8_t)(da * 255.0);
    parson->attrs[48] = (uint8_t)(db * 255.0);
  }

  if (strstr(parson->nom, "_hat")) {
    parson->attrs[35] = 255;
  }
  if (strstr(parson->nom, "glass")) {
    parson->attrs[15] = 255;
  }
  if (strstr(parson->nom, "smil")) {
    parson->attrs[31] = 255;
  }
  if (strstr(parson->nom, "young")) {
    parson->attrs[39] = 255;
  }
  if (strstr(parson->nom, "bald")) {
    parson->attrs[4] = 255;
    parson->attrs[28] = 255;
  }

  parson->revised = time(NULL);
  parson->revisor = 0;

  pipe->load_ctx_bytes(parson->attrs);
  memcpy(pipe->ctrbuf, parson->controls, sizeof(parson->controls));
  pipe->generate();

  delete[] ctrbuf0;
  delete[] ctxbuf0;
}

void handle(Pipeline *pipe, ParsonDB *parsons, FILE *infp, FILE *outfp) {
  Parson::Nom fam[ParsonDB::nfam];
  uint8_t cmd[8];
  char nom[32];
  int ret;
  uint32_t ip = 0;
  uint8_t lockbuf[8];
  uint8_t new_ctrlock, new_tgtlock;
  uint8_t *new_fren = new uint8_t[32];
  uint8_t *new_ctx = new uint8_t[pipe->ctxlay->n];
  double *new_ctr = new double[pipe->ctrlay->n];
  uint32_t meta[4];

  unsigned int fixiters = 5;
  double fixblend = 0.2;

  FILE *learnfp = fopen((pipe->final()->dir + "/learn.dat").c_str(), "r");
  assert(learnfp);

  while (1) {
    fprintf(stderr, "syncing pipeline (load)\n");
    pipe->load();
    fprintf(stderr, "synced pipeline (load)\n");

    fprintf(stderr, "loading cmd n=8\n");
    ensure(8 == fread(cmd, 1, 8, infp));
    fprintf(stderr, "loaded cmd[0]=%u\n", cmd[0]);

    fprintf(stderr, "loading nom n=32\n");
    ensure(32 == fread(nom, 1, 32, infp));
    fprintf(stderr, "loaded nom=%s\n", nom);

    ensure(Parson::valid_nom(nom));

    Parson *parson = parsons->find(nom);
    ensure(parson);

    assert(pipe->ctxlay->n == Parson::nattrs);
    assert(pipe->ctrlay->n == Parson::ncontrols);
    assert(sizeof(parson->target) == pipe->outlay->n * sizeof(double));

    assert(sizeof(parson->attrs) == pipe->ctxlay->n);
    pipe->load_ctx_bytes(parson->attrs);

    assert(sizeof(parson->controls) == pipe->ctrlay->n * sizeof(double));
    memcpy(pipe->ctrbuf, parson->controls, sizeof(parson->controls));

    pipe->tgtlock = parson->target_lock;
    pipe->ctrlock = parson->control_lock;

    const uint8_t *response;
    unsigned int responsen;

    switch (cmd[0]) {
    case 0: {
double t0 = realtime();
      ensure( 4 == fread(&ip, 1, 4, infp) );
      ip = ntohl(ip);

      if (parson->created) {
        memcpy(pipe->outbuf, parson->target, pipe->outlay->n * sizeof(double));
      } else {
        makeparson(learnfp, pipe, parson);
        makeparens(learnfp, pipe, parsons, parson);

        pipe->load_ctx_bytes(parson->attrs);
        memcpy(pipe->ctrbuf, parson->controls, sizeof(parson->controls));
      }

      pipe->uptarget();
      memcpy(parson->target, pipe->outbuf, pipe->outlay->n * sizeof(double));


      assert(sizeof(double) * pipe->outlay->n == sizeof(Parson::target));
      response = (uint8_t *)pipe->outbuf;
      responsen = pipe->outlay->n * sizeof(double);
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);



      response = parson->attrs;
      responsen = Parson::nattrs;
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);

      pipe->reencode();

      response = (uint8_t *)pipe->ctrbuf;
      responsen = Parson::ncontrols * sizeof(double);
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);


      pipe->readjust();

      response = (uint8_t *)pipe->adjbuf;
      responsen = sizeof(double) * pipe->adjlay->n;
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);



      fprintf(stderr, "generating\n");
      pipe->generate();
      fprintf(stderr, "generated\n");


      assert(pipe->outlay->n * sizeof(double) == sizeof(Parson::target));
      response = (uint8_t *)pipe->outbuf;
      responsen = sizeof(double) * pipe->outlay->n;

      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);

      memset(lockbuf, 0, sizeof(lockbuf));
//      parson->target_lock = -1;
      lockbuf[0] = parson->target_lock;
      lockbuf[1] = parson->control_lock;
      response = lockbuf;
      responsen = 8;
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);

      response = (uint8_t *)parson->frens;
      responsen = sizeof(parson->frens);
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);

      response = (uint8_t *)parson->parens;
      responsen = sizeof(parson->parens);
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);

      parsons->fill_fam(parson->nom, fam);
      response = (uint8_t *)fam;
      responsen = sizeof(fam);
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);

      meta[0] = parson->creator;
      meta[1] = parson->created;
      meta[2] = parson->revisor;
      meta[3] = parson->revised;

      response = (uint8_t *)meta;
      responsen = sizeof(meta);
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);
double t1 = realtime();
fprintf(stderr, "elapsed %gs\n", t1 - t0);

      break;
    }

    case 1: {
double t0 = realtime();
      new_tgtlock = cmd[1];
      new_ctrlock = cmd[2];

      bool newly_created = false;
      if (parson->created) {
        memcpy(pipe->outbuf, parson->target, pipe->outlay->n * sizeof(double));
        parson->revised = time(NULL);
      } else {
        makeparson(learnfp, pipe, parson);
        makeparens(learnfp, pipe, parsons, parson);
        parson->created = time(NULL);
        newly_created = true;

        pipe->load_ctx_bytes(parson->attrs);
        memcpy(pipe->ctrbuf, parson->controls, sizeof(parson->controls));
      }

      uint8_t hyper[8], zero[8];
      memset(zero, 0, 8);

      assert(sizeof(hyper) == 8);
      ensure(8 == fread(hyper, 1, 8, infp));
      ensure( pipe->ctxlay->n == fread(new_ctx, 1, pipe->ctxlay->n, infp) );
      ensure( pipe->ctrlay->n == fread(new_ctr, sizeof(double), pipe->ctrlay->n, infp) );
      ensure( pipe->adjlay->n == fread(pipe->adjbuf, sizeof(double), pipe->adjlay->n, infp) );
      ensure( 32 == fread(new_fren, 1, 32, infp) );
      ensure( new_fren[31] == 0 );
      ensure( 4 == fread(&ip, 1, 4, infp) );
      ip = ntohl(ip);

      if (newly_created) {
        parson->creator = ip;
      }
      parson->revisor = ip;

      bool ctrchange = (hyper[0] == 1 || hyper[0] == 2 || hyper[0] == 3);
      if (ctrchange) {
        unsigned int off = 0;
        for (unsigned int i = 0; i < 8 && i < pipe->stages.size(); ++i) {
          unsigned int j = off;
          unsigned int jn = off + pipe->stages[i]->ctrlay->n;

          if (hyper[1] & (1 << i)) {
            new_ctrlock |= (1 << i);

            while (j < jn) {
              switch (hyper[0]) {
              case 1:
                parson->controls[j] = sigmoid(randgauss());
                break;
              case 2:
                parson->controls[j] = sigmoid(unsigmoid(parson->controls[j]) * 1.5);
                break;
              case 3:
                parson->controls[j] = sigmoid(unsigmoid(parson->controls[j]) / 1.5);
                break;
              }
              ++j;
            }
          }

          off = jn;
        }
        assert(off == pipe->ctrlay->n);
        memcpy(pipe->ctrbuf, parson->controls, sizeof(parson->controls));
      }

      parson->control_lock = pipe->ctrlock = new_ctrlock;
      parson->target_lock = pipe->tgtlock = new_tgtlock;

      assert(Parson::nattrs == pipe->ctxlay->n);
      bool ctxchange = memcmp(parson->attrs, new_ctx, Parson::nattrs);
      if (ctxchange) {
        memcpy(parson->attrs, new_ctx, Parson::nattrs);
        pipe->load_ctx_bytes(parson->attrs);
      }

 
      if (!ctxchange && !ctrchange) {
        pipe->retarget();
        assert(sizeof(Parson::target) == sizeof(double) * pipe->outlay->n);
        memcpy(parson->target, pipe->outbuf, sizeof(Parson::target));
      }

      pipe->uptarget();
      memcpy(parson->target, pipe->outbuf, pipe->outlay->n * sizeof(double));


      response = (uint8_t *)parson->target;
      responsen = sizeof(parson->target);
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);

      response = parson->attrs;
      responsen = Parson::nattrs;
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);



      pipe->reencode();

      assert(Parson::ncontrols == pipe->ctrlay->n);
      memcpy(parson->controls, pipe->ctrbuf, sizeof(double) * Parson::ncontrols);


      response = (uint8_t *)parson->controls;
      responsen = sizeof(double) * Parson::ncontrols;
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);

      pipe->readjust();

      response = (uint8_t *)pipe->adjbuf;
      responsen = sizeof(double) * pipe->adjlay->n;
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);




      fprintf(stderr, "generating\n");
      pipe->generate();
      fprintf(stderr, "generated\n");


      assert(pipe->outlay->n * sizeof(double) == sizeof(Parson::target));
      response = (uint8_t *)pipe->outbuf;
      responsen = pipe->outlay->n * sizeof(double);
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);


      uint8_t lockbuf[8];
      memset(lockbuf, 0, sizeof(lockbuf));
      lockbuf[0] = parson->target_lock;
      lockbuf[1] = parson->control_lock;
      response = lockbuf;
      responsen = 8;
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);


      char fren0[32];
      ensure( new_fren[31] == 0 );
      strcpy(fren0, (const char *)new_fren);
      if (*fren0) {
        ensure( Parson::valid_nom(fren0) );
        parson->add_fren(fren0);
        Parson *fparson = parsons->find(fren0);
        ensure(fparson);
        fparson->add_fren(nom);

        if (hyper[0] == 4) {
          fprintf(stderr, "cloning %s->%s\n", nom, fren0);
          memset(fparson->target, 0, sizeof(Parson::target));
          memcpy(fparson->controls, parson->controls, sizeof(Parson::controls));
          memcpy(fparson->attrs, parson->attrs, sizeof(Parson::attrs));
          memcpy(fparson->parens[0], parson->nom, 32);
          memcpy(fparson->parens[1], parson->nom, 32);
          //memset(fparson->parens[1], 0, 32);
          fparson->target_lock = 0;
          fparson->control_lock = 0xFF;
          fparson->revised = time(NULL);
          fparson->revisor = ip;
        }

        if (hyper[0] == 5) {
          uint8_t gender = parson->attrs[20] ? 1 : 0;
          if (parson->attrs[20] != fparson->attrs[20]) {
            gender = (randuint() % 2);
          }

          if (!fparson->created)
            makeparson(learnfp, pipe, fparson);

          std::string cnomstr = Parson::bread(nom, fren0, gender);
          const char *cnom = cnomstr.c_str();
          fprintf(stderr, "breading %s + %s -> %s\n", nom, fren0, cnom);

          Parson *cparson = parsons->find(cnom);

          makebread(learnfp, pipe, parson, fparson, cparson, gender);
        }
      }


      response = (uint8_t *)parson->frens;
      responsen = sizeof(parson->frens);
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);

      response = (uint8_t *)parson->parens;
      responsen = sizeof(parson->parens);
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);

      parsons->fill_fam(parson->nom, fam);
      response = (uint8_t *)fam;
      responsen = sizeof(fam);
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);

      meta[0] = parson->creator;
      meta[1] = parson->created;
      meta[2] = parson->revisor;
      meta[3] = parson->revised;

      response = (uint8_t *)meta;
      responsen = sizeof(meta);
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);
double t1 = realtime();
fprintf(stderr, "elapsed %gs\n", t1 - t0);

      break;
    }




    case 3: {
      const uint8_t *response;
      unsigned int responsen;

      if (parson->created) {
        memcpy(pipe->outbuf, parson->target, pipe->outlay->n * sizeof(double));
      } else {
        makeparson(learnfp, pipe, parson);
      }

      pipe->uptarget();
      memcpy(parson->target, pipe->outbuf, pipe->outlay->n * sizeof(double));

      pipe->reencode();

      fprintf(stderr, "generating\n");
      pipe->generate();
      fprintf(stderr, "generated\n");

      assert(pipe->outlay->n * sizeof(double) == sizeof(Parson::target));
      response = (uint8_t *)pipe->outbuf;
      responsen = sizeof(double) * pipe->outlay->n;
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);

      break;
    }

    case 4: {
      const uint8_t *response;
      unsigned int responsen;

      if (parson->created) {
        memcpy(pipe->outbuf, parson->target, pipe->outlay->n * sizeof(double));
      } else {
        makeparson(learnfp, pipe, parson);
      }

      pipe->uptarget();
      memcpy(parson->target, pipe->outbuf, pipe->outlay->n * sizeof(double));

      pipe->reencode();

seedrand();
{
pipe->ctxbuf[6] = randuint() % 2 ? 1.0 : 0;
pipe->ctxbuf[7] = randuint() % 2 ? 1.0 : 0;
pipe->ctxbuf[12] = randuint() % 2 ? 1.0 : 0;
pipe->ctxbuf[27] = randuint() % 2 ? 1.0 : 0;
pipe->ctxbuf[31] = randuint() % 2 ? 1.0 : 0;
pipe->ctxbuf[21] = randuint() % 2 ? 1.0 : 0;
pipe->ctxbuf[23] = randuint() % 2 ? 1.0 : 0;
pipe->ctxbuf[28] = randuint() % 2 ? 1.0 : 0;
pipe->ctxbuf[25] = randuint() % 2 ? 1.0 : 0;
pipe->ctxbuf[29] = randuint() % 2 ? 1.0 : 0;
}

      fprintf(stderr, "generating\n");
      pipe->generate();
      fprintf(stderr, "generated\n");

      assert(pipe->outlay->n * sizeof(double) == sizeof(Parson::target));
      response = (uint8_t *)pipe->outbuf;
      responsen = sizeof(double) * pipe->outlay->n;
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);

      break;
    }


    case 55: {
      const uint8_t *response;
      unsigned int responsen;

      response = (uint8_t *)parson;
      responsen = sizeof(*parson);
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);

      break;
    }

    case 56: {
      char fnom[32];
      fprintf(stderr, "loading fnom n=32\n");
      ensure(32 == fread(fnom, 1, 32, infp));
      fprintf(stderr, "loaded fnom=%s\n", fnom);

      ensure(Parson::valid_nom(fnom));
      parson->add_fren(fnom);
      if (!parson->created) {
        makeparson(learnfp, pipe, parson);
        parson->created = time(NULL);
      }

      Parson *fparson = parsons->find(fnom);
      ensure(fparson);
      fparson->add_fren(nom);

      const uint8_t *response;
      unsigned int responsen;

      response = (uint8_t *)parson;
      responsen = sizeof(*parson);
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);

      break;
    }


    default:
      fprintf(stderr, "bad command cmd=%u\n", cmd[0]);
      return;
    }

    fflush(outfp);
  }
}

int usage() {
  fprintf(stderr, "Usage: servemore project.dir port\n");
  return 1;
}

int main(int argc, char **argv) {
  if (argc < 2)
    return usage();
  uint16_t port = atoi(argv[1]);

  int s, ret;
  struct sockaddr_in sin;

  s = socket(PF_INET, SOCK_STREAM, 0);
  assert(s >= 0);

  int reuse = 1;
  ret = setsockopt(s, SOL_SOCKET, SO_REUSEPORT, (const char*)&reuse, sizeof(reuse));
  assert(ret == 0);
  ret = setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse));
  assert(ret == 0);

  sin.sin_family = AF_INET;
  sin.sin_port = htons(port);
  sin.sin_addr.s_addr = htonl(INADDR_ANY);
  ret = bind(s, (struct sockaddr *)&sin, sizeof(sin));
  assert(ret == 0);

  ret = listen(s, 256);
  assert(ret == 0);






#if 1
  int max_children = 8;

  for (unsigned int i = 0; i < max_children; ++i) {
    fprintf(stderr, "forking\n");
    if (fork()) {
      fprintf(stderr, "forked\n");
      continue;
    }

    fprintf(stderr, "opening pipeline\n");
    Pipeline *pipe = open_pipeline();
    fprintf(stderr, "opened pipeline\n");

    fprintf(stderr, "opening parsons\n");
    ParsonDB *parsons = open_parsons();
    fprintf(stderr, "opened parsons\n");


    while (1) {
      fprintf(stderr, "accepting\n");

      struct sockaddr_in sin;
      socklen_t sinlen = sizeof(sin);
      int c = accept(s, (struct sockaddr *)&sin, &sinlen);
      assert(c != -1);

      fprintf(stderr, "accepted\n");
#endif




#if 0


  int children = 0;
  int max_children = 10;

  while (1) {
    if (max_children > 1) {
      while (children > 0) {
        pid_t ret = waitpid(-1, NULL, WNOHANG);
        assert(ret != -1);
        if (ret > 0) {
          --children;
          fprintf(stderr, "reaped pid=%d children=%d\n", ret, children);
        } else {
          break;
        }
      }
      while (children > max_children) {
        fprintf(stderr, "reaping children=%d\n", children);
        pid_t ret = wait(NULL);
        assert(ret != -1);
        --children;
        fprintf(stderr, "reaped pid=%d children=%d\n", ret, children);
      }
    }

    fprintf(stderr, "accepting children=%d\n", children);

    struct sockaddr_in sin;
    socklen_t sinlen = sizeof(sin);
    int c = accept(s, (struct sockaddr *)&sin, &sinlen);
    assert(c != -1);

    if (max_children > 1) {
      pid_t child = fork();

      if (child) {
        close(c);
        ++children;

        fprintf(stderr, "accepted pid=%d children=%d\n", child, children);
        continue;
      }

      close(s);
    }
#endif











      int c2 = dup(c);
      assert(c2 != -1);

      FILE *infp = fdopen(c, "rb");
      FILE *outfp = fdopen(c2, "wb");

      fprintf(stderr, "handling\n");
      handle(pipe, parsons, infp, outfp);
      fprintf(stderr, "handled\n");

      fclose(infp);
      fclose(outfp);

#if 0
      if (max_children > 1)
        exit(0);
#endif
    }



#if 1
  }


  for (unsigned int i = 0; i < max_children; ++i) {
    fprintf(stderr, "waiting\n");
    wait(NULL);
    fprintf(stderr, "waited\n");
  }
#endif

  return 0;
}

