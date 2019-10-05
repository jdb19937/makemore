#define __MAKEMORE_URB_CC__ 1
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "urb.hh"
#include "pipeline.hh"
#include "strutils.hh"
#include "imgutils.hh"
#include "encoder.hh"
#include "generator.hh"
#include "cudamem.hh"

namespace makemore {

using namespace std;

void Urb::add_gen(const std::string &tag, const std::string &projdir) {
  gens[tag] = new Supergen(projdir);
}

void Urb::add_sty(const std::string &tag, const std::string &projdir) {
  stys[tag] = new Styler(projdir);
}

Urb::Urb(const char *_dir, unsigned int _mbn) {
  mbn = _mbn;

  assert(strlen(_dir) < 4000);
  dir = _dir;

  ruffposer = new Autoposer("bestposer.proj");
  fineposer = new Autoposer("newposer.proj");

  Zone *main = new Zone(dir + "/main.zone");
  zones.push_back(main);

  sks0 = new Zone("/spin/dan/easyceleb.dat");
  sks1 = new Zone("/spin/dan/shampane.dat");

  outgoing = new Bus(dir + "/outgoing.bus");

#if 0
  enc = new Encoder("enc.proj", 1);

  add_gen("shampane", "gen.shampane.proj");
  // add_gen("easyceleb", "gen.easyceleb.proj");
  add_gen("lfw", "gen.lfw.proj");
  add_gen("lfw_masks", "gen.lfw_masks.proj");
  default_gen = gens["shampane"];

  add_sty("shampane", "sty.shampane.proj");
  add_sty("easyceleb", "sty.easyceleb.proj");
  add_sty("lfw", "sty.lfw.proj");
  default_sty = stys["shampane"];
#endif



  enc = new Superenc("nenc.proj", 1);

add_gen("n", "ngen.proj");
//add_gen("minidne", "gena.minidne.proj");
#if 0
add_gen("miniceleb", "gena.miniceleb.proj");
add_gen("alpha", "gena.proj");
add_gen("gazetest1", "gena.gazetest1.proj");
add_gen("anderson", "gena.anderson.proj");
#endif
  default_gen = gens["n"];

  add_sty("n", "nsty.proj");
  //add_sty("minidne", "stya.minidne.proj");
#if 0
  add_sty("miniceleb", "stya.miniceleb.proj");
  add_sty("alpha", "stya.proj");
  add_sty("gazetest1", "stya.gazetest1.proj");
  add_sty("anderson", "stya.anderson.proj");
#endif

  default_sty = stys["n"];

  images.clear();
  DIR *dp = opendir((dir + "/images").c_str());
  assert(dp);
  struct dirent *de;
  while ((de = readdir(dp))) {
    if (*de->d_name == '.')
      continue;
    images.push_back(dir + "/images/" + de->d_name);
  }
  closedir(dp);
  sort(images.begin(), images.end());
  assert(images.size());

  srcimages.clear();
  dp = opendir((dir + "/srcimages").c_str());
  assert(dp);
  while ((de = readdir(dp))) {
    if (*de->d_name == '.')
      continue;
    srcimages.push_back(dir + "/srcimages/" + de->d_name);
  }
  closedir(dp);
  sort(srcimages.begin(), srcimages.end());
  assert(srcimages.size());

  {
    std::string home = dir + "/home";
    int ret = ::mkdir(home.c_str(), 0700);
    assert(ret == 0 || ret == -1 && errno == EEXIST);
  }
}

Urb::~Urb() {
  delete enc;

  for (auto i : gens)
    delete i.second;

  for (auto i : stys)
    delete i.second;

  for (Zone *zone : zones) {
    delete zone;
  }

  delete outgoing;
}

static void nomsuffixes(const std::string &nom, std::vector<std::string> *suf) {
  const char *p = nom.c_str();
  const char *q = p + nom.length();

  while (p < q) {
    if (*p == '_' && p[1]) {
      suf->push_back(p + 1);
    }
    ++p;
  }
}

Parson *Urb::make(const std::string &nom, unsigned int tier, unsigned int gens, Parson *child, unsigned int which) {

  if (!Parson::valid_nom(nom))
    return NULL;

  if (Parson *x = find(nom)) {
fprintf(stderr, "found %s\n", nom.c_str());
    if (!*x->parens[0] || !*x->parens[1]) {
      seedrand(Parson::hash_nom(nom.c_str()));
      x->paren_noms(nom.c_str(), x->parens[0], x->parens[1]);
    }

    if (gens) {
      (void) make(x->parens[0], tier, gens - 1, x, 0);
      (void) make(x->parens[1], tier, gens - 1, x, 1);
    }
    return x;
  }

  Parson parson;
  memset(&parson, 0, sizeof(Parson));
  strcpy(parson.nom, nom.c_str());

  double dev = 1.0;
  if (strstr(parson.nom, "norm"))
    dev = 0;

  std::string seednom = nom;
  while (!strncmp(seednom.c_str(), "anti", 4)) {
    seednom = seednom.c_str() + 4;
    dev = -dev;
  }

  seedrand(Parson::hash_nom(seednom.c_str()));

  for (unsigned int k = 0; k < Parson::ncontrols; ++k) {
    parson.controls[k] = randgauss() * dev;
    parson.variations[k] = 1.0;
  }

  {
    std::vector<std::string> suf;
    nomsuffixes(parson.nom, &suf);
    for (auto famnom : suf) {
      if (Parson *fam = find(famnom)) {
        for (unsigned int k = 0; k < Parson::ncontrols; ++k) {
          parson.controls[k] = parson.controls[k] * sqrt(fam->variations[k]) + fam->controls[k];
        }
        break;
      }
    }
  }

  parson.paren_noms(nom.c_str(), parson.parens[0], parson.parens[1]);

  parson.tone = dev;


  {
    Zone *sks = sks0;

    Parson *skp = sks->pick(which ? "female" : "male", 32);
    if (!skp)
      skp = sks->pick();
    assert(skp);

    parson.skid = sks->dom(skp);
    memcpy(parson.sketch, skp->sketch, sizeof(parson.sketch));


#if 0
    std::string srcfn = skp->srcfn;
    assert(srcfn.length());
    strcpy(parson.srcfn, skp->srcfn);

    Partrait prt;
    prt.load(srcfn);

    strcpy(parson.sty, "easyceleb");

    Styler *sty = get_sty(parson.sty);
    assert(sty);
    enc->encode(prt, &parson, sty);
#endif
  }

  if (strstr(parson.nom, "gray")) {
    for (unsigned int k = 0; k < 192; k += 3) {
      parson.sketch[k + 1] = 0;
      parson.sketch[k + 2] = 0;
    }
  }

  parson.angle = randrange(-0.05, 0.05);
  parson.stretch = 1.0 + randrange(-0.05, 0.05);
  parson.skew = randrange(-0.05, 0.05);

  parson.add_tag(which ? "female" : "male");

  strcpy(parson.gen, "alpha");
  strcpy(parson.sks, "easyceleb");

  if (child) {
    seedrand(Parson::hash_nom(child->nom, 93));
    for (unsigned int k = 0; k < Parson::ncontrols; ++k)
      if (randuint() % 2 == which)
        parson.controls[k] = child->controls[k];
    seedrand(Parson::hash_nom(parson.nom, 93));

    // for (unsigned int k = 0; k < 192; ++k)
    //   parson.sketch[k] = 0.5 * parson.sketch[k] + 0.5 * child->sketch[k];

    if (randuint() % 16 == 0) {
      const char *race[8] = {
        "white", "white", "white",
        "black", "black", "black",
        "asian", "hispanic"
      };
      parson.add_tag(race[randuint() % 8]);
    } else {
      if (child->has_tag("white"))
        parson.add_tag("white");
      else if (child->has_tag("black"))
        parson.add_tag("black");
      else if (child->has_tag("hispanic"))
        parson.add_tag("hispanic");
      else if (child->has_tag("asian"))
        parson.add_tag("asian");
    }
  } else {
    const char *race[8] = {
      "white", "white", "white",
      "black", "black", "black",
      "asian", "hispanic"
    };
    parson.add_tag(race[randuint() % 8]);
  }

  int age = randuint() % 3;
  if (age == 0)
    parson.add_tag("young");
  if (age == 2)
    parson.add_tag("old");

  strcpy(parson.sty, "shampane");

#if 0
  int samb = 0;
  samb += parson.has_tag("male");
  samb += parson.has_tag("female");
  int ramb = 0;
  ramb += parson.has_tag("white");
  ramb += parson.has_tag("black");
  ramb += parson.has_tag("hispanic");
  ramb += parson.has_tag("asian");

  if (ramb == 1 && samb == 1) {
    bool isw = parson.has_tag("white");
    bool isb = parson.has_tag("black");
    bool ism = parson.has_tag("male");
    bool isf = parson.has_tag("female");

    if (isw && ism)
      strcpy(parson.sty, "shampane_wm");
    if (isb && isf)
      strcpy(parson.sty, "shampane_bf");
    if (isw && isf)
      strcpy(parson.sty, "shampane_wf");
    if (isb && isf)
      strcpy(parson.sty, "shampane_bf");
  }
#endif

  if (gens) {
    (void) make(parson.parens[0], tier, gens - 1, &parson, 0);
    (void) make(parson.parens[1], tier, gens - 1, &parson, 1);
  }

fprintf(stderr, "%s -> %s, %s\n", nom.c_str(), parson.parens[0], parson.parens[1]);

  time_t now = time(NULL);
  parson.creator = 0x7F000001;
  parson.revisor = 0x7F000001;
  parson.created = now;
  parson.revised = now;
  parson.visited = now;

  Parson *x = this->make(parson, tier);

  return x;
}


Parson *Urb::make(unsigned int tier) {
  string imagefn = images[randuint() % images.size()];
  string png = slurp(imagefn);

  Parson parson;
  memset(&parson, 0, sizeof(Parson));

  vector<string> tags;
  // imglab("png", png, 64, 64, parson.target, &tags);
  // pnglab(png, 64, 64, parson.target, &tags);

  bool gender = 1;
  for (auto tag : tags) {
    if (tag == "female")
      gender = 0;
    parson.add_tag(tag.c_str());
  }

  string nom;
  do {
    nom = Parson::gen_nom(gender);
  } while (find(nom));

  time_t now = time(NULL);
  parson.creator = 0x7F000001;
  parson.revisor = 0x7F000001;
  parson.created = now;
  parson.revised = now;
  parson.visited = now;

  strcpy(parson.nom, nom.c_str());
  return this->make(parson, tier);
}



void Urb::restock(unsigned int n, vector<string> *noms) {
  for (unsigned int i = 0; i < n; ++i) {
    Parson *parson = make();
    assert(parson);
    if (noms)
      noms->push_back(parson->nom);
  }
}

    

void Urb::generate(Parson *p, long min_age) {
//  p->generate(pipe1, min_age);
}

Parson *Urb::find(const std::string &nom, unsigned int *tierp) const {

  for (unsigned int tier = 0, tiers = zones.size(); tier < tiers; ++tier) {
    Zone *zone = zones[tier];

    if (Parson *p = zone->find(nom)) {
      if (tierp)
        *tierp = tier;
      return p;
    }
  }

  return NULL;
}

unsigned int Urb::tier(const Zone *zone) const {
  for (unsigned int tier = 0, tiers = zones.size(); tier < tiers; ++tier) {
    if (zones[tier] == zone)
      return tier;
  }
  assert(0);
}

unsigned int Urb::tier(const Parson *p) const {
  for (unsigned int tier = 0, tiers = zones.size(); tier < tiers; ++tier) {
    Zone *zone = zones[tier];
    if (zone->has(p))
      return tier;
  }
  assert(0);
}

Zone *Urb::zone(const Parson *p) const {
  for (unsigned int tier = 0, tiers = zones.size(); tier < tiers; ++tier) {
    Zone *zone = zones[tier];
    if (zone->has(p))
      return zone;
  }
  return NULL;
}

bool Urb::deport(const char *nom) {
  if (Parson *p = find(nom)) {
    deport(p);
    return true;
  } else {
    return false;
  }
}

void Urb::deport(Parson *x) {
  assert(x->exists());
  _busout(*x);
  memset(x, 0, sizeof(Parson));
}

void Urb::_busout(const Parson &x) {
  outgoing->add(x);
}

Parson *Urb::make(const Parson &x, unsigned int tier) {
  if (tier >= zones.size()) {
    _busout(x);
    return NULL;
  }

  assert(tier < zones.size());
  Zone *zone = zones[tier];

  bool eq;
  Parson q;
  Parson *p = zone->make(x, &eq, &q);

  if (!p)
    return NULL;
  if (eq)
    (void)make(q, tier + 1);

  return p;
}

}
