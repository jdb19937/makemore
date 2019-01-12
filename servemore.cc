#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <netinet/in.h>

#include <string>

#include "cudamem.hh"
#include "pipeline.hh"
#include "parson.hh"

#include "sha256.c"

#define ensure(x) do { \
  if (!(x)) { \
    fprintf(stderr, "closing\n"); \
    return; \
  } \
} while (0)

using namespace std;

Pipeline *open_pipeline() {
  Pipeline *pipe = new Pipeline(1);
  pipe->add_stage(new Project("new8.proj", 1));
  pipe->add_stage(new Project("new16.proj", 1));
  pipe->add_stage(new Project("new32.proj", 1));
  pipe->add_stage(new Project("new64.proj", 1));
  return pipe;
}

ParsonDB *open_parsons() {
  return new ParsonDB("parsons.dat");
}

void handle(Pipeline *pipe, ParsonDB *parsons, FILE *infp, FILE *outfp) {
  uint8_t cmd;
  char nom[32];
  int ret;

  while (1) {
    fprintf(stderr, "syncing pipeline (load)\n");
    pipe->load();
    fprintf(stderr, "synced pipeline (load)\n");

    fprintf(stderr, "loading cmd n=1\n");
    ensure(1 == fread(&cmd, 1, 1, infp));
    fprintf(stderr, "loaded cmd=%u\n", cmd);

    fprintf(stderr, "loading nom n=32\n");
    ensure(32 == fread(nom, 1, 32, infp));
    fprintf(stderr, "loaded nom=%s\n", nom);

    ensure(Parson::valid_nom(nom));

    Parson *parson = parsons->find(nom);
    ensure(parson);
    ensure(!strcmp(parson->nom, nom));


    assert(pipe->ctxlay->n == Parson::nattrs);
    pipe->load_ctx(parson->attrs);

    assert(pipe->ctrlay->n == Parson::ncontrols);
    pipe->load_ctr(parson->controls);

    assert(sizeof(parson->target) == pipe->outlay->n);

    const uint8_t *response;
    unsigned int responsen;

    switch (cmd) {
    case 0: {
      if (parson->created) {
        pipe->load_out(parson->target);
      } else {
        pipe->generate();
        memcpy(parson->target, pipe->boutbuf, pipe->outlay->n);
      }

      fprintf(stderr, "readjusting\n");
      pipe->readjust();
      fprintf(stderr, "readjusted\n");

      response = parson->target;
      responsen = sizeof(parson->target);
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);
//fprintf(stderr, "tgtresponse: "); for (int i = 0; i < 40; ++i) { fprintf(stderr, "%u,", response[i]); } fprintf(stderr, "\n");



      response = parson->attrs;
      responsen = Parson::nattrs;
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);



      response = parson->controls;
      responsen = Parson::ncontrols;
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);



      response = pipe->badjbuf;
      responsen = pipe->adjlay->n;
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);
//fprintf(stderr, "adjresponse: "); for (int i = 0; i < 40; ++i) { fprintf(stderr, "%u,", response[i]); } fprintf(stderr, "\n");



      fprintf(stderr, "generating\n");
      pipe->generate();
      fprintf(stderr, "generated\n");


      assert(pipe->outlay->n == sizeof(Parson::target));
      response = pipe->boutbuf;
      responsen = pipe->outlay->n;

      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);

//fprintf(stderr, "genresponse: "); for (int i = 0; i < 40; ++i) { fprintf(stderr, "%u,", response[i]); } fprintf(stderr, "\n");

      break;
    }

    case 1: {

      uint8_t hyper[8], zero[8];
      memset(zero, 0, 8);

      assert(sizeof(hyper) == 8);
      fprintf(stderr, "loading hyper n=%lu\n", sizeof(hyper));
      ensure(8 == fread(hyper, 1, 8, infp));
      fprintf(stderr, "loaded hyper\n");

      fprintf(stderr, "loading context n=%u\n", pipe->ctxlay->n);
      ensure( pipe->load_ctx(infp) );
      fprintf(stderr, "loaded context n=%u\n", pipe->ctxlay->n);

      fprintf(stderr, "loading controls n=%u\n", pipe->ctrlay->n);
      ensure( pipe->load_ctr(infp) );
      fprintf(stderr, "loaded controls n=%u\n", pipe->ctrlay->n);

      fprintf(stderr, "loading adjust n=%u\n", pipe->adjlay->n);
      ensure( pipe->load_adj(infp) );
      fprintf(stderr, "loaded adjust n=%u\n", pipe->adjlay->n);

      fprintf(stderr, "retargeting\n");
      pipe->retarget();
      fprintf(stderr, "retargeted\n");

      fprintf(stderr, "readjusting\n");
      pipe->readjust();
      fprintf(stderr, "readjusted\n");

      if (hyper[0]) {
        pipe->reencode(hyper[0]);
        pipe->readjust();
      }

      assert(Parson::nattrs == pipe->ctxlay->n);
      memcpy(parson->attrs, pipe->bctxbuf, Parson::nattrs);

      assert(Parson::ncontrols == pipe->ctrlay->n);
      memcpy(parson->controls, pipe->bctrbuf, Parson::ncontrols);

      assert(sizeof(Parson::target) == pipe->outlay->n);
      memcpy(parson->target, pipe->boutbuf, sizeof(Parson::target));
      if (!parson->created) {
        parson->created = time(NULL);
      }

      response = parson->target;
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


      response = parson->controls;
      responsen = Parson::ncontrols;
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);

      response = pipe->badjbuf;
      responsen = pipe->adjlay->n;
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);


      fprintf(stderr, "generating\n");
      pipe->generate();
      fprintf(stderr, "generated\n");


      assert(pipe->outlay->n == sizeof(Parson::target));
      response = pipe->boutbuf;
      responsen = pipe->outlay->n;
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);

      break;
    }




    case 3: {
      const uint8_t *response;
      unsigned int responsen;

      fprintf(stderr, "generating\n");
      pipe->generate();
      fprintf(stderr, "generated\n");

      assert(pipe->outlay->n == sizeof(Parson::target));
      response = pipe->boutbuf;
      responsen = pipe->outlay->n;
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);

      break;
    }
    default:
      fprintf(stderr, "bad command cmd=%u\n", cmd);
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
  int max_children = 2;

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

