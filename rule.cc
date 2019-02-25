#define __MAKEMORE_RULE_CC__ 1

#include <stdio.h>
#include <stdlib.h>
#include <netinet/in.h>

#include "rule.hh"
#include "strutils.hh"

namespace makemore {

using namespace std;

static string norm(string x) {
  const char *p = x.c_str();
 
  string y;
  while (*p) {
    if (isspace(*p)) {
      if (y.length())
        y += " ";
      ++p;
      while (isspace(*p))
        ++p;
    } else {
      char buf[2] = {p[0], 0};
      y += buf;
      ++p;
    }
  }

  if (y[y.length() - 1] == ' ')
    y.erase(y.length() - 1);

  return y;
}

static void clauses(const string &x, vector<string> *cl) {
  cl->clear();
  split(x.c_str(), ',', cl);

  for (auto cli = cl->begin(); cli != cl->end(); ++cli)
    *cli = norm(*cli);
}

unsigned int Rule::parse(const char *line) {
  string reqstr, rspstr;

  std::string tagstr;
  vector<string> tagwords;
  if (const char *tagsep = strchr(line, ':')) {
    string tagstr(line, tagsep);
    line = tagsep + 1;
    split(tagstr.c_str(), ' ', &tagwords);
  }

  const char *sep = strstr(line, "->");
  if (!sep)
    return 0;

  tags.clear();
  if (tagwords.size()) {
    for (auto tagi = tagwords.begin(); tagi != tagwords.end(); ++tagi)
      tags.add(tagi->c_str());
  }

  reqstr = string(line, sep - line);
  vector<string> reqparts;
  clauses(reqstr, &reqparts);

  int multiplicity = 1;
  if (sep[2] == 'x')
    multiplicity = atoi(sep + 3);
  if (multiplicity < 0)
    return 0;

  while (*sep && !isspace(*sep))
    ++sep;
  while (isspace(*sep))
    ++sep;

  rspstr = string(sep);
  vector<string> rspparts;
  clauses(rspstr, &rspparts);


  unsigned int nreq = reqparts.size();
  unsigned int nrsp = rspparts.size();

  wild.resize(nreq);
  for (unsigned int i = 0; i < nreq; ++i)
    wild[i].parse(reqparts[i]);

  req.resize(nreq);
  for (unsigned int i = 0; i < nreq; ++i)
    req[i].encode(reqparts[i].c_str());

  rsp.resize(nrsp);
  for (unsigned int i = 0; i < nrsp; ++i)
    rsp[i].encode(rspparts[i].c_str());

  prepared = false;
  return ((unsigned int)multiplicity);
}

void Rule::save(FILE *fp) const {
  size_t ret;

  unsigned int nreq = req.size();
  assert(nreq == wild.size());
  uint32_t bnreq = htonl(nreq);
  ret = fwrite(&bnreq, 4, 1, fp);
  assert(ret == 1);

  for (unsigned int i = 0; i < nreq; ++i) {
    req[i].save(fp);
    wild[i].save(fp);
  }

  unsigned int nrsp = rsp.size();
  uint32_t bnrsp = htonl(nrsp);
  ret = fwrite(&bnrsp, 4, 1, fp);
  assert(ret == 1);

  for (unsigned int i = 0; i < nrsp; ++i) {
    rsp[i].save(fp);
  }
}

void Rule::load(FILE *fp) {
  size_t ret;

  uint32_t bnreq;
  ret = fread(&bnreq, 4, 1, fp);
  assert(ret == 1);
  unsigned int nreq = ntohl(bnreq);

  req.resize(nreq);
  wild.resize(nreq);
  for (unsigned int i = 0; i < nreq; ++i) {
    req[i].load(fp);
    wild[i].load(fp);
  }


  uint32_t bnrsp;
  ret = fread(&bnrsp, 4, 1, fp);
  assert(ret == 1);
  unsigned int nrsp = ntohl(bnrsp);

  rsp.resize(nrsp);
  for (unsigned int i = 0; i < nrsp; ++i) {
    rsp[i].load(fp);
  }
}


}
