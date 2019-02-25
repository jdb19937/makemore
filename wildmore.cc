#define __MAKEMORE_WILDMORE_CC__

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <netinet/in.h>

#include <vector>
#include <string>

#include "wildmore.hh"
#include "hashbag.hh"
#include "strutils.hh"

namespace makemore {

using namespace std;

void Wildmore::parse(const char *str) {
  vector<string> words;
  split(str, ' ', &words);
  parse(words);
}

void Wildmore::parse(const std::string &str) {
  parse(str.c_str());
}

static void randfix(const vector<string> &words, vector<string> *fixated) {
  for (auto word : words) {
    if (word == "?") {
      if (randbit()) {
        fixated->push_back("*");
      }
    } else if (word == "**") {
      unsigned int nstars = randuint() % 3;
      for (unsigned int i = 0; i < nstars; ++i)
        fixated->push_back("*");
    } else if (word == "***") {
      unsigned int nstars = randuint() % 4;
      for (unsigned int i = 0; i < nstars; ++i)
        fixated->push_back("*");
    } else {
      fixated->push_back(word);
    }
  }
}

void Wildmore::parse(const vector<string> &_words) {
  vector<string> words;
  randfix(_words, &words);
  unsigned int wn = words.size();

  front3 = 0;
  for (unsigned int wi = 0; wi < 3 && wi < wn; ++wi)
    if (words[wi] == "*")
      front3 |= (1 << wi);

  if (wn > 3) {
    vector<string> lethwords;
    lethwords.resize(wn - 3);
    for (unsigned int wi = 3; wi < wn; ++wi)
      lethwords[wi - 3] = words[wi];
    backwild.parse(lethwords);
  } else {
    backwild.clear();
  }
}

void Wildmore::mutate(Shibbomore *shmore) {
  for (unsigned int i = 0; i < 3; ++i)
    if (front3 & (1 << i))
      shmore->front[i] = Hashbag::random();

  backwild.mutate(&shmore->backleth);
}


void Wildmore::save(FILE *fp) const {
  uint8_t front3pkt[8];
  memset(front3pkt, 0, 8);
  front3pkt[0] = front3;

  size_t ret;
  ret = fwrite(front3pkt, 1, 8, fp);
  assert(ret == 8);

  backwild.save(fp);
}

void Wildmore::load(FILE *fp) {
  uint8_t front3pkt[8];

  size_t ret;
  ret = fread(front3pkt, 1, 8, fp);
  assert(ret == 8);
  front3 = front3pkt[0];
  assert(front3 < 8);

  backwild.load(fp);
}

}
