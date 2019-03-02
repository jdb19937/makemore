#define __MAKEMORE_CONVO_CC__ 1

#include <vector>
#include <string>
#include <algorithm>

#include "convo.hh"
#include "strutils.hh"

namespace makemore {

using namespace std;

void Convo::build(const std::string &str, double decay) {
  vector<string> words;
  splitwords(str, &words);
  build(words);
}

void Convo::build(const vector<std::string> &words, double decay) {
  vector<vector<string> > thread;
  splitthread(words, &thread, "|");
  build(thread, decay);
}

void Convo::build(const vector<vector<string> > &thread, double decay) {
  history.clear();
  current.clear();

  for (auto ti = thread.rbegin(); ti != thread.rend(); ++ti) {
    auto words = *ti;
    Shibbomore shmore;
    shmore.encode(words);
    add(shmore, decay);
  }
}
    
void Convo::build(const vector<Shibbomore> &shmores, double decay) {
  history.clear();
  current.clear();

  for (auto shmi = shmores.rbegin(); shmi != shmores.rend(); ++shmi)
    add(*shmi, decay);
}

void Convo::add(const std::string &str, double decay) {
  Shibbomore shmore;
  shmore.encode(str.c_str());
  add(shmore, decay);
}

void Convo::add(const Shibbomore &shmore, double decay) {
  history.mul(decay);
  history.add(current);
  current = shmore;
}


}
