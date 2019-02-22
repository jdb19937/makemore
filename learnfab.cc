#include <stdio.h>

#include "vocab.hh"
#include "shibboleth.hh"
#include "script.hh"
#include "confab.hh"
#include "brane.hh"
#include "strutils.hh"
#include "hashbag.hh"

#include <map>

using namespace makemore;
using namespace std;

#if 0
map<string, unsigned int> wc;
set<string> common;

void countwords(const std::string &line) {
  vector<string> words;
  split(line.c_str(), ' ', &words);
  for (unsigned int i = 0; i < words.size(); ++i) {
    ++wc[words[i]];
  }
}
#endif

void getsample(const std::string &line,
  Hashbag *prev, Hashbag *tgt)  {

  vector<string> words;
  split(line.c_str(), ' ', &words);
  unsigned int wi, wj;

  assert(words.size() > 1);
  wi = randuint() % (words.size() - 1);
  wj = wi + 1;

//printf("prev: ");
  prev->clear();
  for (unsigned int w = wi; w < wj; ++w) {
    *prev *= 0.7;
    prev->add(words[w].c_str());
//printf("%s ", words[w].c_str());
  }
//printf("\n");

  tgt->clear();
  tgt->add(words[wj].c_str());
//printf("word: %s\n", words[wj].c_str());

#if 0
//printf("rare: ");
  rare->clear();
  for (unsigned int i = 0; i < words.size(); ++i) {
    rare->add(words[i].c_str());
//printf("%s ", words[i].c_str());
  }
  *rare *= (1.0 / words.size());
//printf("\n\n");
#endif

}
  
  

int main() {
  unsigned int mbn = 8;
  seedrand();

fprintf(stderr, "reading lines\n");
  map<string, Hashbag> strbag;

  vector<string> lines;
  string line;
  while (read_line(stdin, &line)) {
    vector<string> words;
    split(line.c_str(), ' ', &words);

    if (words.size() == 0)
      continue;
    if (words[words.size() - 1] != ".")
      words.push_back(".");

    for (unsigned int j = 2; j < words.size(); ++j) {
      strbag[words[j - 2] + " " + words[j - 1]].add(words[j].c_str());
      strbag[words[j - 1]].add(words[j].c_str());
      strbag[""].add(words[j].c_str());
    }


       


    lines.push_back(line);
  }

#if 0
fprintf(stderr, "counting words\n");
  multimap<unsigned int, string> cw;
  for (auto wci = wc.begin(); wci != wc.end(); ++wci)
    cw.insert(make_pair(wci->second, wci->first));

  unsigned int i = 0;
  for (auto cwi = cw.rbegin(); cwi != cw.rend(); ++cwi) {
    if (i >= 1024)
      break;
    common.insert(cwi->second);
    // printf("%u\t%s\n", cwi->first, cwi->second.c_str());
    ++i;
  }
#endif

fprintf(stderr, "go\n");

  Confab confab("test.confab", mbn);
  assert(sizeof(Hashbag) * 1 == confab.ctxlay->n * sizeof(double));
  assert(sizeof(Hashbag) * 1 == confab.tgtlay->n * sizeof(double));
  confab.load();

  unsigned int rounds = 0;
  while (1) {
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      Hashbag prev, word;
      const string &line = lines[randuint() % lines.size()];
      getsample(line, &prev, &word);

      memcpy(confab.ctxbuf + mbi * confab.ctxlay->n, &prev, sizeof(Hashbag));
      memcpy(confab.tgtbuf + mbi * confab.tgtlay->n, &word, sizeof(Hashbag));
    }

    confab.burn(0.001, 0.001);
    confab.condition(0.0001, 0.0001);

    if (rounds % 100 == 0) {
      confab.report("learnfab");
      confab.save();
    }

    ++rounds;
  }

  return 0;

#if 0

  assert(confab.mbn == mbn);

  Shibboleth req, rsp;
  Rule rules[mbn];

  int i = 0;
  while (1) {
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      const Rule *r = script.pick();
      rules[mbi] = *r;
      rules[mbi].prepare();
    }
    brane.burn(rules, mbn, 0.001);

    if (i % 1000 == 0) {
      confab.report("learnfab");
      confab.save();
    }

    ++i;
  }

#endif
  return 0;
}
