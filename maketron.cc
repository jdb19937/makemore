#include <string>

#include "supertron.hh"
#include "random.hh"
#include "cudamem.hh"

using namespace makemore;

int usage() {
  fprintf(stderr, "Usage: maketron [--new file.map | ...]\n");
  return 1;
}

int main(int argc, char **argv) {
  seedrand();

  --argc;
  ++argv;
  assert(argc > 0);


  std::string opt = argv[0];
  if (opt == "--new") {
    --argc;
    ++argv;
    assert(argc > 0);
    const char *mapfn = argv[0];

    FILE *mapfp;
    assert(mapfp = fopen(mapfn, "w"));

    Mapfile *mapfile = new Mapfile(mapfn);
    Supertron *st = new Supertron(mapfile);
    mapfile->save();
    delete st;
    delete mapfile;
    fclose(mapfp);
    return 0;
  }

  if (opt == "--adam_b1") {
    --argc;
    ++argv;
    assert(argc > 0);
    const char *mapfn = argv[0];

    --argc;
    ++argv;
    double x = argc > 0 ? strtod(argv[0], NULL) : 1.0;

    Mapfile *mapfile = new Mapfile(mapfn);
    Supertron *st = new Supertron(mapfile);

    for (auto lay : st->layers) {
      Supertron::Layer::Head head;
      decude(lay->head, 1, &head);
      head.adam_b1 = x;
      encude(&head, 1, lay->head);
    }

    mapfile->save();

    delete st;
    delete mapfile;
    return 0;
  }

  if (opt == "--adam_b2") {
    --argc;
    ++argv;
    assert(argc > 0);
    const char *mapfn = argv[0];

    --argc;
    ++argv;
    double x = argc > 0 ? strtod(argv[0], NULL) : 1.0;

    Mapfile *mapfile = new Mapfile(mapfn);
    Supertron *st = new Supertron(mapfile);

    for (auto lay : st->layers) {
      Supertron::Layer::Head head;
      decude(lay->head, 1, &head);
      head.adam_b2 = x;
      encude(&head, 1, lay->head);
    }

    mapfile->save();

    delete st;
    delete mapfile;
    return 0;
  }

  if (opt == "--randomize") {
    --argc;
    ++argv;
    assert(argc > 0);
    const char *mapfn = argv[0];

    --argc;
    ++argv;
    double disp = argc > 0 ? strtod(argv[0], NULL) : 1.0;

    Mapfile *mapfile = new Mapfile(mapfn);
    Supertron *st = new Supertron(mapfile);

    st->randomize(disp);
    mapfile->save();

    delete st;
    delete mapfile;
    return 0;
  }

  if (opt == "--randomize_last") {
    --argc;
    ++argv;
    assert(argc > 0);
    const char *mapfn = argv[0];

    --argc;
    ++argv;
    double disp = argc > 0 ? strtod(argv[0], NULL) : 1.0;

    Mapfile *mapfile = new Mapfile(mapfn);
    Supertron *st = new Supertron(mapfile);

    st->randomize_last(disp);
    mapfile->save();

    delete st;
    delete mapfile;
    return 0;
  }

  if (opt == "--add") {
    --argc;
    ++argv;
    assert(argc > 0);
    const char *mapfn = argv[0];

    --argc;
    ++argv;
    assert(argc == 10);
    Supertron::Layer::Type type = (Supertron::Layer::Type)atoi(argv[0]);
    int iw = atoi(argv[1]), ih = atoi(argv[2]), ic = atoi(argv[3]);
    int ow = atoi(argv[4]), oh = atoi(argv[5]), oc = atoi(argv[6]);
    int d = atoi(argv[7]), s = atoi(argv[8]);
    Supertron::Layer::Activation a = (Supertron::Layer::Activation)atoi(argv[9]);

    Mapfile *mapfile = new Mapfile(mapfn);
    Supertron *st = new Supertron(mapfile);

fprintf(stderr, "addlayer\n");
    st->add_layer(type, iw, ih, ic, ow, oh, oc, d, s, a);
    mapfile->save();

    delete st;
    delete mapfile;
    return 0;
  }

  return usage();
}
