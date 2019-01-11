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
#include "project.hh"

#include "sha256.c"

#define ensure(x) do { \
  if (!(x)) { \
    fprintf(stderr, "closing\n"); \
    return; \
  } \
} while (0)

using namespace std;

void handle(Project *proj, FILE *infp, FILE *outfp) {
  uint8_t cmd;
  char nom[32], nomfn[4096];
  int ret;

  uint8_t *zero = new uint8_t[proj->targetlay->n]();

  while (1) {
    fprintf(stderr, "syncing proj (load)\n");
    proj->load();
    fprintf(stderr, "synced proj (load)\n");

    fprintf(stderr, "loading cmd n=1\n");
    ensure(1 == fread(&cmd, 1, 1, infp));
    fprintf(stderr, "loaded cmd=%u\n", cmd);

    fprintf(stderr, "loading nom n=32\n");
    ensure(32 == fread(nom, 1, 32, infp));
    if (nom[31]) {
      fprintf(stderr, "bad nom 1\n");
      return;
    }
    if (nom[0] >= '0' && nom[0] <= '9') {
      fprintf(stderr, "bad nom 2\n");
      return;
    }
    for (unsigned int i = 0; i < 32; ++i) {
      if (!nom[i])
        break;
      if (!(nom[i] >= 'a' && nom[i] <= 'z' || nom[i] == '_' || nom[i] >= '0' && nom[i] <= '9')) {
        fprintf(stderr, "bad nom 3\n");
        return;
      }
    }
    if (!nom[0]) {
      fprintf(stderr, "bad nom 4\n");
      return;
    }

    fprintf(stderr, "loaded nom=%s\n", nom);
    sprintf(nomfn, "%s/saved/%s.dat", proj->dir.c_str(), nom);

    switch (cmd) {
    case 0: {
      fprintf(stderr, "loading context n=%u\n", proj->contextlay->n);
      ensure( proj->loadcontext(infp) );
      fprintf(stderr, "loaded context n=%u\n", proj->contextlay->n);

      fprintf(stderr, "loading controls n=%u\n", proj->controlslay->n);
      ensure( proj->loadcontrols(infp) );
      fprintf(stderr, "loaded controls n=%u\n", proj->controlslay->n);

      const uint8_t *response;
      unsigned int responsen;

      if (FILE *nomfp = fopen(nomfn, "r")) {
        ensure(proj->loadcontext(nomfp));
        ensure(proj->loadcontrols(nomfp));
        ensure(proj->loadadjust(nomfp));
        ensure(proj->loadtarget(nomfp));
        fclose(nomfp);

        fprintf(stderr, "readjusting\n");
        proj->dotarget2();
        proj->readjust();
        fprintf(stderr, "readjusted\n");

        response = proj->btargetbuf;
        responsen = proj->targetlay->n;
      } else {
        uint8_t hash[32];
        SHA256_CTX sha;
        sha256_init(&sha);
        sha256_update(&sha, (const uint8_t *)nom, strlen(nom));
        sha256_final(&sha, hash);

        unsigned int s;
        memcpy(&s, hash, sizeof(s));
        seedrand(s);

        proj->nulladjust();
        proj->randcontrols(1);
        for (int i = 0; i < proj->contextlay->n; ++i)
          proj->contextbuf[i] = randrange(0, 1);

        fprintf(stderr, "generating\n");
        proj->generate();
        fprintf(stderr, "generated\n");

        seedrand();

        response = proj->output() + proj->contextlay->n;
        responsen = proj->outputlay->n - proj->contextlay->n;
      }

      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);
fprintf(stderr, "tgtresponse: "); for (int i = 0; i < 40; ++i) { fprintf(stderr, "%u,", response[i]); } fprintf(stderr, "\n");


      proj->encodectx();

      response = proj->bcontextbuf;
      responsen = proj->contextlay->n;
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);


      proj->encodectrl();

      response = proj->bcontrolbuf;
      responsen = proj->controlslay->n;
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);

      proj->encodeadj();
    
      response = proj->badjustbuf;
      responsen = proj->adjustlay->n;
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);
fprintf(stderr, "adjresponse: "); for (int i = 0; i < 40; ++i) { fprintf(stderr, "%u,", response[i]); } fprintf(stderr, "\n");



      proj->nulladjust();
      fprintf(stderr, "generating\n");
      proj->generate();
      fprintf(stderr, "generated\n");

      assert(proj->outputlay->n == proj->contextlay->n + proj->targetlay->n);
      response = proj->output() + proj->contextlay->n;
      responsen = proj->targetlay->n;

      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);

fprintf(stderr, "genresponse: "); for (int i = 0; i < 40; ++i) { fprintf(stderr, "%u,", response[i]); } fprintf(stderr, "\n");

      break;
    }

    case 1: {
      uint8_t hyper[8];

      assert(sizeof(hyper) == 8);
      fprintf(stderr, "loading hyper n=%lu\n", sizeof(hyper));
      ensure(8 == fread(hyper, 1, 8, infp));
      fprintf(stderr, "loaded hyper\n");

      fprintf(stderr, "loading context n=%u\n", proj->contextlay->n);
      ensure( proj->loadcontext(infp) );
      fprintf(stderr, "loaded context n=%u\n", proj->contextlay->n);

      fprintf(stderr, "loading controls n=%u\n", proj->controlslay->n);
      ensure( proj->loadcontrols(infp) );
      fprintf(stderr, "loaded controls n=%u\n", proj->controlslay->n);

      fprintf(stderr, "loading adjust n=%u\n", proj->adjustlay->n);
      ensure( proj->loadadjust(infp) );
      fprintf(stderr, "loaded adjust n=%u\n", proj->adjustlay->n);

      fprintf(stderr, "dotargeting\n");
      fprintf(stderr, "readjusting\n");
      proj->dotarget1();
      proj->dotarget2();
      proj->readjust();
      fprintf(stderr, "readjusted\n");

      if (memcmp(hyper, zero, 8)) {
        for (int i = 0; i < 16; ++i) {
          fprintf(stderr, "generating\n");
          proj->generate(hyper);
          fprintf(stderr, "generated\n");

          fprintf(stderr, "readjusting\n");
          proj->readjust();
          fprintf(stderr, "readjusted\n");
        }
      }

      if (hyper[1] || hyper[3] || hyper[5] || hyper[7]) {
        fprintf(stderr, "syncing proj (save)\n");
        proj->save();
        fprintf(stderr, "synced proj (save)\n");
      }
 

      if (FILE *nomfp = fopen(nomfn, "w")) {
        ret = fwrite(proj->bcontextbuf, 1, proj->contextlay->n, nomfp);
        assert(ret == proj->contextlay->n);
        ret = fwrite(proj->bcontrolbuf, 1, proj->controlslay->n, nomfp);
        assert(ret == proj->controlslay->n);
        ret = fwrite(proj->badjustbuf, 1, proj->adjustlay->n, nomfp);
        assert(ret == proj->adjustlay->n);
        ret = fwrite(proj->btargetbuf, 1, proj->targetlay->n, nomfp);
        assert(ret == proj->targetlay->n);

        fclose(nomfp);
      }



      const uint8_t *response = proj->btargetbuf;
      unsigned int responsen = proj->targetlay->n;
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);


      response = proj->bcontextbuf;
      responsen = proj->contextlay->n;
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);


      response = proj->bcontrolbuf;
      responsen = proj->controlslay->n;
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);

      response = proj->badjustbuf;
      responsen = proj->adjustlay->n;
      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);



      proj->nulladjust();
      fprintf(stderr, "generating\n");
      proj->generate();
      fprintf(stderr, "generated\n");

      assert(proj->outputlay->n == proj->contextlay->n + proj->targetlay->n);
      response = proj->output() + proj->contextlay->n;
      responsen = proj->targetlay->n;

      fprintf(stderr, "writing n=%u\n", responsen);
      ret = fwrite(response, 1, responsen, outfp);
      ensure(ret == responsen);
      fprintf(stderr, "wrote n=%d\n", responsen);

      break;
    }

    case 3: {
      const uint8_t *response;
      unsigned int responsen;

      if (FILE *nomfp = fopen(nomfn, "r")) {
        ensure(proj->loadcontext(nomfp));
        ensure(proj->loadcontrols(nomfp));
        fclose(nomfp);

        proj->nulladjust();

        fprintf(stderr, "generating\n");
        proj->generate();
        fprintf(stderr, "generated\n");

      } else {
        uint8_t hash[32];
        SHA256_CTX sha;
        sha256_init(&sha);
        sha256_update(&sha, (const uint8_t *)nom, strlen(nom));
        sha256_final(&sha, hash);

        unsigned int s;
        memcpy(&s, hash, sizeof(s));
        seedrand(s);

        proj->nulladjust();
        proj->randcontrols(1);
        for (int i = 0; i < proj->contextlay->n; ++i)
          proj->contextbuf[i] = randrange(0, 1);

        fprintf(stderr, "generating\n");
        proj->generate();
        fprintf(stderr, "generated\n");

        seedrand();
      }

      assert(proj->outputlay->n == proj->contextlay->n + proj->targetlay->n);
      response = proj->output() + proj->contextlay->n;
      responsen = proj->targetlay->n;

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
  if (argc < 3)
    return usage();
  string project_dir = argv[1];
  uint16_t port = atoi(argv[2]);

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
  int max_children = 3;

  for (unsigned int i = 0; i < max_children; ++i) {
    fprintf(stderr, "forking\n");
    if (fork()) {
      fprintf(stderr, "forked\n");
      continue;
    }

    fprintf(stderr, "opening\n");
    Project *proj = open_project(project_dir.c_str(), 1);
    fprintf(stderr, "opened\n");


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
      handle(proj, infp, outfp);
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

