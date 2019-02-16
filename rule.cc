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
  string reqstr, memstr, outstr, nemstr, cmdstr, bufstr[4];

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
    assert(!*r);
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
    assert(!*r);
  } else {
    outstr = c;
  }
  
  for (unsigned int j = 1; j < 5 && j < bparts.size(); ++j) {
    bufstr[j - 1] = bparts[j];
  }

  reqstr = norm(reqstr);
  memstr = norm(memstr);
  nemstr = norm(nemstr);
  outstr = norm(outstr);
  cmdstr = norm(cmdstr);
  bufstr[0] = norm(bufstr[0]);
  bufstr[1] = norm(bufstr[1]);
  bufstr[2] = norm(bufstr[2]);
  bufstr[3] = norm(bufstr[3]);

#if 0
fprintf(stderr, "reqstr=[%s]\n", reqstr.c_str());
fprintf(stderr, "memstr=[%s]\n", memstr.c_str());
fprintf(stderr, "nemstr=[%s]\n", nemstr.c_str());
fprintf(stderr, "outstr=[%s]\n", outstr.c_str());
fprintf(stderr, "cmdstr=[%s]\n", cmdstr.c_str());
fprintf(stderr, "bufstr[0]=[%s]\n", bufstr[0].c_str());
fprintf(stderr, "bufstr[1]=[%s]\n", bufstr[1].c_str());
fprintf(stderr, "bufstr[2]=[%s]\n", bufstr[2].c_str());
fprintf(stderr, "bufstr[3]=[%s]\n", bufstr[3].c_str());
#endif

  reqwild.parse(reqstr);
  memwild.parse(memstr);

  req.encode(reqstr.c_str());
  mem.encode(memstr.c_str());
  nem.encode(nemstr.c_str());
  cmd.encode(cmdstr.c_str());
  out.encode(outstr.c_str());
  buf[0].encode(bufstr[0].c_str());
  buf[1].encode(bufstr[1].c_str());
  buf[2].encode(bufstr[2].c_str());
  buf[3].encode(bufstr[3].c_str());

  return multiplicity;
}

void Rule::save(FILE *fp) const {
  size_t ret;

  req.save(fp);
  mem.save(fp);
  cmd.save(fp);
  out.save(fp);
  nem.save(fp);
  buf[0].save(fp);
  buf[1].save(fp);
  buf[2].save(fp);
  buf[3].save(fp);

  reqwild.save(fp);
  memwild.save(fp);

  uint32_t nm = htonl(multiplicity);
  ret = fwrite(&nm, 1, 4, fp);
  assert(ret == 4);
}

void Rule::load(FILE *fp) {
  size_t ret;

  req.load(fp);
  mem.load(fp);
  cmd.load(fp);
  out.load(fp);
  nem.load(fp);
  buf[0].load(fp);
  buf[1].load(fp);
  buf[2].load(fp);
  buf[3].load(fp);

  reqwild.load(fp);
  memwild.load(fp);

  uint32_t nm;
  ret = fread(&nm, 1, 4, fp);
  assert(ret == 4);
  multiplicity = (unsigned int)ntohl(nm);
}


}
