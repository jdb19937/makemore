#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <netinet/in.h>

#include <string>

#include "project.hh"

using namespace std;

void handle(const char *project_dir, FILE *infp, FILE *outfp) {
  Project *proj = open_project(project_dir, 1);

  while (1) {
    fprintf(stderr, "loading context n=%u\n", proj->contextlay->n);
    proj->loadcontext(infp);
    fprintf(stderr, "loaded context n=%u\n", proj->contextlay->n);
     fprintf(stderr, "context: "); for (int i = 0; i < 40; ++i) { fprintf(stderr, "%lf,", proj->contextbuf[i]); } fprintf(stderr, "\n");

    fprintf(stderr, "loading controls n=%u\n", proj->controlslay->n);
    proj->loadcontrols(infp);
    fprintf(stderr, "loaded controls n=%u\n", proj->controlslay->n);
     fprintf(stderr, "controls: "); for (int i = 0; i < 100; ++i) { fprintf(stderr, "%lf,", proj->controlbuf[i]); } fprintf(stderr, "\n");

    fprintf(stderr, "loading adjust n=%u\n", proj->adjustlay->n);
    proj->loadadjust(infp);
    fprintf(stderr, "loaded adjust n=%u\n", proj->adjustlay->n);
     fprintf(stderr, "adjust: "); for (int i = 0; i < 100; ++i) { fprintf(stderr, "%lf,", proj->adjustbuf[i]); } fprintf(stderr, "\n");

    fprintf(stderr, "generating\n");
    proj->generate();
    fprintf(stderr, "generated\n");

    const uint8_t *response = proj->output() + proj->contextlay->n;
    unsigned int responsen = proj->outputlay->n - proj->contextlay->n;

    fprintf(stderr, "writing n=%u\n", responsen);
    fprintf(stderr, "response: "); for (int i = 0; i < 100; ++i) { fprintf(stderr, "%u,", response[i]); } fprintf(stderr, "\n");

    int ret = fwrite(response, 1, responsen, outfp);
    assert(ret == responsen);
    fprintf(stderr, "wrote n=%d\n", responsen);

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

    int c2 = dup(c);
    assert(c2 != -1);

    FILE *infp = fdopen(c, "rb");
    FILE *outfp = fdopen(c2, "wb");


    handle(project_dir.c_str(), infp, outfp);


    fclose(infp);
    fclose(outfp);

    if (max_children > 1)
      exit(0);
  }

  return 0;
}

