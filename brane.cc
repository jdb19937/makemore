#define __MAKEMORE_BRANE_CC__ 1
#include "brane.hh"
#include "rule.hh"
#include "strutils.hh"

namespace makemore {

using namespace std;

Brane::Brane(Confab *_confab) {
  confab = _confab;
  _init_vocab();
}

Brane::~Brane() {

}

static const char *partdesc(char c) {
  switch (c) {
  case '^': return "head of";
  case '$': return "rear of";
  case '%': return "torso of";
  default: assert(0);
  }
}

static const char *bufdesc(char c) {
  switch (c) {
  case 'I': return "primary input";
  case 'J': return "primary output";
  case 'M': return "memory input";
  case 'N': return "memory output";
  case '1': return "register 1";
  case '2': return "register 2";
  case '3': return "register 3";
  case '4': return "register 4";
  default: assert(0);
  }
}

void Brane::_init_vocab() {
  char buf[256], desc[256];

  const char *rbuf = "IJMN1234";
  const char *wbuf = "JN1234";
  const char *parts = "^%$";

  vocab.clear();
  for (const char *b = rbuf; *b; ++b) {
    for (const char *c = wbuf; *c; ++c) {
      for (const char *p = parts; *p; ++p) {
        sprintf(buf, "p%c%c%c", *p, *b, *c);
        sprintf(desc, "prepend %s %s to %s", partdesc(*p), bufdesc(*b), bufdesc(*c));
        vocab.add(buf, desc);

        sprintf(buf, "a%c%c%c", *p, *b, *c);
        sprintf(desc, "append %s %s to %s", partdesc(*p), bufdesc(*b), bufdesc(*c));
        vocab.add(buf, desc);

        sprintf(buf, "c%c%c%c", *p, *b, *c);
        sprintf(desc, "copy %s %s to %s", partdesc(*p), bufdesc(*b), bufdesc(*c));
        vocab.add(buf, desc);
      }

      sprintf(buf, "p%c%c", *b, *c);
      sprintf(desc, "prepend %s to %s", bufdesc(*b), bufdesc(*c));
      vocab.add(buf, desc);

      sprintf(buf, "a%c%c", *b, *c);
      sprintf(desc, "append %s to %s", bufdesc(*b), bufdesc(*c));
      vocab.add(buf, desc);

      sprintf(buf, "c%c%c", *b, *c);
      sprintf(desc, "copy %s to %s", bufdesc(*b), bufdesc(*c));
      vocab.add(buf, desc);
    }
  }

  for (const char *c = wbuf; *c; ++c) {
    for (const char *p = parts; *p; ++p) {
      sprintf(buf, "n%c%c", *p, *c);
      sprintf(desc, "negate %s %s", partdesc(*p), bufdesc(*c));
      vocab.add(buf, desc);

      sprintf(buf, "z%c%c", *p, *c);
      sprintf(desc, "zero %s %s", partdesc(*p), bufdesc(*c));
      vocab.add(buf, desc);
    }

    sprintf(buf, "n%c", *c);
    sprintf(desc, "negate %s", bufdesc(*c));
    vocab.add(buf, desc);

    sprintf(buf, "z%c", *c);
    sprintf(desc, "zero %s", bufdesc(*c));
    vocab.add(buf, desc);
  }

  for (const char *b = wbuf; *b; ++b) {
    sprintf(buf, "e%c", *b);
    sprintf(desc, "evaluate %s", bufdesc(*b));
    vocab.add(buf, desc);

    sprintf(buf, "r%c", *b);
    sprintf(desc, "reverse %s", bufdesc(*b));
    vocab.add(buf, desc);

    sprintf(buf, "d%c", *b);
    sprintf(desc, "decode %s", bufdesc(*b));
    vocab.add(buf, desc);

    sprintf(buf, "s%c", *b);
    sprintf(desc, "spell %s", bufdesc(*b));
    vocab.add(buf, desc);

    sprintf(buf, "j%c", *b);
    sprintf(desc, "join %s", bufdesc(*b));
    vocab.add(buf, desc);
  }

  vocab.add("nop", "do nothing");
}

static const Shibboleth *rbufmap(char cbuf, const Shibboleth *req, const Shibboleth *rsp) {
  switch (cbuf) {
  case 'I': return req;
  case 'M': return req + 1;
  case 'J': return rsp + 1;
  case 'N': return rsp + 2;
  case '1': return rsp + 3;
  case '2': return rsp + 4;
  case '3': return rsp + 5;
  case '4': return rsp + 6;
  default: assert(0);
  }
}

static Shibboleth *wbufmap(char cbuf, Shibboleth *rsp) {
  switch (cbuf) {
  case 'J': return rsp + 1;
  case 'N': return rsp + 2;
  case '1': return rsp + 3;
  case '2': return rsp + 4;
  case '3': return rsp + 5;
  case '4': return rsp + 6;
  default: assert(0);
  }
}

void Brane::burn(const Rule *rule, unsigned int mbn, double pi) {
  Shibboleth req[2], rsp[7];

  assert(mbn == confab->mbn);
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    req[0] = rule[mbi].req;
    req[1] = rule[mbi].mem;

//fprintf(stderr, "burn req %lf %lf %lf (%lf)\n", req[0].head.size(), req[0].torso.size(), req[0].rear.size(), req[0].pairs.size());
//fprintf(stderr, "burn rsp %lf %lf %lf (%lf)\n", rsp[1].head.size(), rsp[1].torso.size(), rsp[1].rear.size(), rsp[1].pairs.size());
    rsp[0] = rule[mbi].cmd;
    rsp[1] = rule[mbi].out;
    rsp[2] = rule[mbi].nem;
    rsp[3] = rule[mbi].buf[0];
    rsp[4] = rule[mbi].buf[1];
    rsp[5] = rule[mbi].buf[2];
    rsp[6] = rule[mbi].buf[3];

    memcpy(confab->ctxbuf + mbi * confab->ctxlay->n, (double *)req, sizeof(Shibboleth) * 2);
    memcpy(confab->tgtbuf + mbi * confab->tgtlay->n, (double *)rsp, sizeof(Shibboleth) * 7);
  }

  confab->burn(pi, pi);
}

Shibboleth Brane::ask(const Shibboleth &in, Shibboleth *memp, unsigned int depth) {
  Shibboleth req[2];
  Shibboleth rsp[7];
  Shibboleth *nemp;

  if (depth > max_depth) {
    Shibboleth dots;
    dots.encode("...");
    return dots;
  }

  req[0] = in;
  req[1] = *memp;

//fprintf(stderr, "ask in  [%s]\n", in.decode(confab->vocab).c_str());
//fprintf(stderr, "ask mem [%s]\n", memp->decode(confab->vocab).c_str());

  unsigned int k = sizeof(Shibboleth) / sizeof(double);
  assert(confab->mbn == 1);
  assert(confab->ctxlay->n * sizeof(double) == 2 * sizeof(Shibboleth)); 
  assert(confab->tgtlay->n * sizeof(double) == 7 * sizeof(Shibboleth));

  memcpy(confab->ctxbuf, (double *)req, sizeof(Shibboleth) * 2);

  confab->generate();

  memcpy((double *)rsp, confab->outbuf, sizeof(Shibboleth) * 7);
  nemp = rsp + 2;
  nemp->clear();

  string cmdstr = rsp[0].decode(vocab);
  vector<string> cmds;
  split(cmdstr.c_str(), ' ', &cmds);
//fprintf(stderr, "ask cmd [%s]\n", cmdstr.c_str());

  for (auto cmdi = cmds.begin(); cmdi != cmds.end(); ++cmdi) { 
    const char *cmd = cmdi->c_str();
    if (!strcmp(cmd, "nop"))
      continue;

    switch (cmd[0]) {
    case 'e':
      {
        assert(cmdi->length() == 2);
        Shibboleth *eval = wbufmap(cmd[1], rsp);
        *eval = ask(*eval, nemp, depth + 1);
        break;
      }
    case 'd':
      {
        assert(cmdi->length() == 2);
        Shibboleth *dec = wbufmap(cmd[1], rsp);

        string decstr = dec->decode(confab->vocab);

        dec->encode(decstr.c_str());
        break;
      }
    case 's':
      {
        assert(cmdi->length() == 2);
        Shibboleth *dec = wbufmap(cmd[1], rsp);

        string decstr = dec->decode(confab->vocab);

        string spelling;
        char sbuf[4];

        for (auto i = decstr.begin(); i != decstr.end(); ++i) {
          if (*i == ' ')
            continue;
          sprintf(sbuf, "%s%c", spelling.length() ? " " : "", *i);
          spelling += sbuf;
        }

        dec->encode(spelling.c_str());
        break;
      }
    case 'j':
      {
        assert(cmdi->length() == 2);
        Shibboleth *dec = wbufmap(cmd[1], rsp);

        string decstr = dec->decode(confab->vocab);

        string joined;
        char sbuf[4];

        for (auto i = decstr.begin(); i != decstr.end(); ++i) {
          if (*i == ' ')
            continue;
          sprintf(sbuf, "%c", *i);
          joined += sbuf;
        }

        dec->encode(joined.c_str());
        break;
      }
    case 'r':
      {
        assert(cmdi->length() == 2);
        Shibboleth *rev = wbufmap(cmd[1], rsp);
        rev->reverse();
        break;
      }
    case 'z':
      {
        const Shibboleth *from;
        Shibboleth *to;
        if (cmdi->length() == 2) {
          to = wbufmap(cmd[1], rsp);
          to->clear();
        } else if (cmdi->length() == 3) {
          to = wbufmap(cmd[2], rsp);
  
          switch (cmd[1]) {
          case '^': to->head.clear(); break;
          case '%': to->torso.clear(); break;
          case '$': to->rear.clear(); break;
          default: assert(0);
          }
        } else {
          assert(0);
        }
      }
      break;
    case 'n':
      {
        const Shibboleth *from;
        Shibboleth *to;
        if (cmdi->length() == 2) {
          to = wbufmap(cmd[1], rsp);
          to->negate();
        } else if (cmdi->length() == 3) {
          to = wbufmap(cmd[2], rsp);
  
          switch (cmd[1]) {
          case '^': to->head.negate(); break;
          case '%': to->torso.negate(); break;
          case '$': to->rear.negate(); break;
          default: assert(0);
          }
        } else {
          assert(0);
        }
      }
      break;
    case 'c':
      {
        const Shibboleth *from;
        Shibboleth *to;
        if (cmdi->length() == 3) {
          from = rbufmap(cmd[1], req, rsp);
          to = wbufmap(cmd[2], rsp);
  
          *to = *from;

        } else if (cmdi->length() == 4) {
  
          from = rbufmap(cmd[2], req, rsp);
          to = wbufmap(cmd[3], rsp);
  
          switch (cmd[1]) {
          case '^': to->copy(from->head); break;
          case '%': to->copy(from->torso); break;
          case '$': to->copy(from->rear); break;
          default: assert(0);
          }
        } else {
          assert(0);
        }
      }
      break;
    case 'a':
      {
        const Shibboleth *from;
        Shibboleth *to;
        if (cmdi->length() == 3) {
          from = rbufmap(cmd[1], req, rsp);
          to = wbufmap(cmd[2], rsp);
  
//fprintf(stderr, "append from [%s]\n", from->decode(confab->vocab).c_str());
//fprintf(stderr, "append mem  [%s]\n", memp->decode(confab->vocab).c_str());
          to->append(*from);

        } else if (cmdi->length() == 4) {
  
          from = rbufmap(cmd[2], req, rsp);
          to = wbufmap(cmd[3], rsp);

          switch (cmd[1]) {
          case '^': to->append(from->head); break;
          case '%': to->append(from->torso); break;
          case '$': to->append(from->rear); break;
          default: assert(0);
          }
        } else {
          assert(0);
        }
      }
      break;
    case 'p':
      {
        const Shibboleth *from;
        Shibboleth *to;
        if (cmdi->length() == 3) {
          from = rbufmap(cmd[1], req, rsp);
          to = wbufmap(cmd[2], rsp);
  
          to->prepend(*from);

        } else if (cmdi->length() == 4) {
  
          from = rbufmap(cmd[2], req, rsp);
          to = wbufmap(cmd[3], rsp);

          switch (cmd[1]) {
          case '^': to->prepend(from->head); break;
          case '%': to->prepend(from->torso); break;
          case '$': to->prepend(from->rear); break;
          default: assert(0);
          }
        } else {
          assert(0);
        }
      }
      break;
    default:
      assert(0);
    }
  }

  memcpy(memp, nemp, sizeof(Shibboleth));
  return rsp[1];
}


}
