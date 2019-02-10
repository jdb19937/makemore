#include <assert.h>

#include "parson.hh"

using namespace makemore;

int main(int argc, char **argv) {
  assert(argc == 2);
  const char *parsons_fn = argv[1];

  ParsonDB *db = new ParsonDB(parsons_fn);

  FILE *dat = fopen("face64.dat", "r");
  assert(dat);

  assert(0 == fseek(dat, 0, SEEK_END));
  off_t nlearn = ftell(dat);
  nlearn /= 12360;

  unsigned int tgtn = Parson::dim * Parson::dim * 3;
  assert(12360 == Parson::nattrs + tgtn);
  uint8_t *tgtbuf = new uint8_t[tgtn];

  unsigned int count = 0;

  while (!feof(stdin)) {
    char buf[256];
    *buf = 0;
    char *ignore = fgets(buf, sizeof(buf) - 1, stdin);
    char *p = strchr(buf, '\n');
    if (!p)
      break;
    *p = 0;

    char nom[32];
    unsigned int cid = (unsigned)atoi(buf);
    assert(cid > 0);
    sprintf(nom, "celeba%u", count);

    Parson *pars = db->find(nom);

    fseek(dat, ((cid - 1) % nlearn) * 12360, SEEK_SET);
    int ret = fread(pars->attrs, 1, Parson::nattrs, dat);
    assert(ret == Parson::nattrs);

    ret = fread(tgtbuf, 1, tgtn, dat);
    assert(ret == Parson::dim * Parson::dim * 3);

    for (unsigned int i = 0; i < tgtn; ++i) {
      pars->target[i] = ((double)tgtbuf[i] + 0.5) / 256.0;
    }

    pars->target_lock = -1;
    pars->control_lock = 0;
    pars->created = time(NULL);
    pars->creator = 0x7F000001;
    pars->revised = time(NULL);
    pars->revisor = 0x7F000001;
    memset(pars->frens, 0, sizeof(pars->frens));
    memset(pars->parens, 0, sizeof(pars->parens));

    fprintf(stderr, "created %s\n", nom);
    ++count;
  }

  return 0;
}
