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

unsigned int Rule::parse(const char *line) {
  string reqstr, memstr, auxstr, outstr, nemstr, buxstr, cmdstr, reg1str, reg2str;

  const char *sep = strstr(line, "->");
  assert(sep);

  std::string a = string(line, sep - line);
  const char *ap = a.c_str();
  if (const char *q = strchr(ap, '(')) {
    reqstr = string(ap, q - ap);

    ++q;
    const char *r = strchr(q, ')');
    assert(r);

    memstr = string(q, r - q);

    ++r;
    while (isspace(*r))
      ++r;
    assert(!*r || *r == '(');

    if (*r == '(') {
      ++r;
      q = strchr(r, ')');
      assert(q);
      auxstr = string(r, q - r);
    }
  } else {
    reqstr = a;
  }

  multiplicity = 1;
  if (sep[2] == 'x')
    multiplicity = atoi(sep + 3);
  if (multiplicity > 32)
    multiplicity = 32;
  assert(multiplicity >= 0);
  while (*sep && !isspace(*sep))
    ++sep;
  while (isspace(*sep))
    ++sep;

  string b = string(sep);
  vector<string> bparts;
  split(b.c_str(), ',', &bparts);
  assert(bparts.size());

  string b0 = bparts[0];
  const char *b0p = b0.c_str();
  string c;
  if (const char *q = strchr(b0p, ':')) {
    cmdstr = string(b0p, q);
    c = string(q + 1);
  } else {
    c = b0;
  }

  const char *cp = c.c_str();
  if (const char *q = strchr(cp, '(')) {
    outstr = string(cp, q - cp);

    ++q;
    const char *r = strchr(q, ')');
    assert(r);

    nemstr = string(q, r - q);

    ++r;
    while (isspace(*r))
      ++r;
    assert(!*r || *r == '(');

    if (*r == '(') {
      ++r;
      q = strchr(r, ')');
      assert(q);
      buxstr = string(r, q - r);
    }
  } else {
    outstr = c;
  }
  
  if (bparts.size() >= 2)
    reg1str = bparts[1];
  if (bparts.size() >= 3)
    reg2str = bparts[2];

  reqstr = norm(reqstr);
  memstr = norm(memstr);
  auxstr = norm(auxstr);
  nemstr = norm(nemstr);
  buxstr = norm(buxstr);
  outstr = norm(outstr);
  cmdstr = norm(cmdstr);
  reg1str = norm(reg1str);
  reg2str = norm(reg2str);

#if 1
fprintf(stderr, "reqstr=[%s]\n", reqstr.c_str());
fprintf(stderr, "memstr=[%s]\n", memstr.c_str());
fprintf(stderr, "auxstr=[%s]\n", auxstr.c_str());
fprintf(stderr, "nemstr=[%s]\n", nemstr.c_str());
fprintf(stderr, "buxstr=[%s]\n", buxstr.c_str());
fprintf(stderr, "outstr=[%s]\n", outstr.c_str());
fprintf(stderr, "cmdstr=[%s]\n", cmdstr.c_str());
fprintf(stderr, "reg1str=[%s]\n", reg1str.c_str());
fprintf(stderr, "reg2str=[%s]\n", reg2str.c_str());
#endif

  reqwild.parse(reqstr);
  memwild.parse(memstr);
  auxwild.parse(auxstr);

  req.encode(reqstr.c_str());
  mem.encode(memstr.c_str());
  nem.encode(nemstr.c_str());
  cmd.encode(cmdstr.c_str());
  out.encode(outstr.c_str());
  bux.encode(buxstr.c_str());
  reg1.encode(reg1str.c_str());
  reg2.encode(reg2str.c_str());

  return multiplicity;
}

void Rule::save(FILE *fp) const {
  size_t ret;

  req.save(fp);
  mem.save(fp);
  aux.save(fp);
  cmd.save(fp);
  out.save(fp);
  nem.save(fp);
  bux.save(fp);
  reg1.save(fp);
  reg2.save(fp);

  reqwild.save(fp);
  memwild.save(fp);
  auxwild.save(fp);

  uint32_t nm = htonl(multiplicity);
  ret = fwrite(&nm, 1, 4, fp);
  assert(ret == 4);
}

void Rule::load(FILE *fp) {
  size_t ret;

  req.load(fp);
  mem.load(fp);
  aux.load(fp);
  cmd.load(fp);
  out.load(fp);
  nem.load(fp);
  bux.load(fp);
  reg1.load(fp);
  reg2.load(fp);

  reqwild.load(fp);
  memwild.load(fp);
  auxwild.load(fp);

  uint32_t nm;
  ret = fread(&nm, 1, 4, fp);
  assert(ret == 4);
  multiplicity = (unsigned int)ntohl(nm);
}


}
