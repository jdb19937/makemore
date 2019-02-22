#define __MAKEMORE_PIPELINE_CC__ 1

#include <netinet/in.h>

#include "pipeline.hh"

#include "twiddle.hh"
#include "project.hh"
#include "layout.hh"

namespace makemore {

Pipeline::Pipeline(const char *_dir, unsigned int _mbn) : Project(_dir, _mbn) {
  mbn = _mbn;
  assert(mbn > 0);

  ctrlay = new Layout;
  adjlay = new Layout;

  assert(config["type"] == "pipeline");

  assert(config["n_stages"] != "");
  unsigned int n_stages = (unsigned int)atoi(config["n_stages"].c_str());
  assert(n_stages > 0);

  char stagedir[4096];
  assert(strlen(_dir) < 4000);
  for (unsigned int i = 1; i <= n_stages; ++i) {
    sprintf(stagedir, "%s/stage%u.proj", _dir, i);
    Stage *proj = new Stage(stagedir, mbn);
    _add_stage(proj);
  }

  _setup();

  ctxbuf = new double[ctxlay->n * mbn];
  ctrbuf = new double[ctrlay->n * mbn];
  adjbuf = new double[adjlay->n * mbn];
  outbuf = new double[outlay->n * mbn];

  tgtlock = 0;
  ctrlock = (unsigned)-1;
}

Stage *Pipeline::initial() {
  assert(stages.size());
  return stages[0];
}

Stage *Pipeline::final() {
  assert(stages.size());
  return stages[stages.size() - 1];
}

void Pipeline::_add_stage(Stage *proj) {
  assert(proj->mbn == mbn);

  if (stages.size()) {
    Stage *prev = final();
    assert(ctxlay);
    assert(ctxlay->n + prev->outlay->n == proj->ctxlay->n);
  }

  stages.push_back(proj);
  *ctrlay += *proj->ctrlay;
  *adjlay += *proj->outlay;

  ctxlay = initial()->ctxlay;
  outlay = final()->outlay;
}

Pipeline::~Pipeline() {
  delete ctrlay;
  delete adjlay;

  delete[] ctxbuf;
  delete[] ctrbuf;
  delete[] adjbuf;
  delete[] outbuf;

  for (auto i = stages.begin(); i != stages.end(); ++i) {
    delete *i;
  }
  stages.clear();
}
  

void Pipeline::generate(
) {
  assert(stages.size());


  unsigned int coff = 0;
  for (unsigned int i = 0; i < stages.size(); ++i) {
    assert(coff < ctrlay->n);
    memcpy(stages[i]->ctrbuf, ctrbuf + coff * mbn, mbn * stages[i]->ctrlay->n * sizeof(double));
    coff += stages[i]->ctrlay->n;
  }
  assert(coff == ctrlay->n);


#if 0
  unsigned int aoff = 0;
  for (unsigned int i = 0; i < stages.size(); ++i) {
    assert(aoff < adjlay->n);
    memcpy(stages[i]->adjbuf, adjbuf + aoff * mbn, mbn * stages[i]->adjlay->n * sizeof(double));
    aoff += stages[i]->adjlay->n;
  }
  assert(aoff == adjlay->n);
#endif



  Stage *proj;

  proj = initial();
  assert(ctxlay->n == proj->ctxlay->n);
  memcpy(proj->ctxbuf, ctxbuf, sizeof(double) * mbn * ctxlay->n);
  proj->generate();

  for (unsigned int i = 1; i < stages.size(); ++i) {
    Stage *lastproj = stages[i - 1];
    Stage *proj = stages[i];

    assert(proj->ctxlay->n == ctxlay->n + lastproj->outlay->n);
    assert(proj->mbn == lastproj->mbn);
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      memcpy(
        proj->ctxbuf + mbi * proj->ctxlay->n + 0,
        ctxbuf + mbi * ctxlay->n,
        sizeof(double) * ctxlay->n
      );
      memcpy(
        proj->ctxbuf + mbi * proj->ctxlay->n + ctxlay->n,
        lastproj->outbuf + mbi * lastproj->outlay->n,
        sizeof(double) * lastproj->outlay->n
      );
    }
    proj->generate();
  }

  proj = final();
  assert(outlay->n == proj->outlay->n);
  memcpy(outbuf, proj->outbuf, sizeof(double) * mbn * outlay->n);

#if 0
    coff = 0;
    for (unsigned int i = 0; i < stages.size(); ++i) {
      assert(coff < controlslay->n);
      memcpy(
        controlbuf + coff * mbn,
        stages[i]->controlbuf,
        mbn * stages[i]->controlslay->n * sizeof(double)
      );
      coff += stages[i]->controlslay->n;
    }
    assert(coff == controlslay->n);
    encodectrl();
  }
#endif

}




void Pipeline::retarget() {
  assert(stages.size());

  {
    unsigned int coff = 0;
    for (unsigned int i = 0; i < stages.size(); ++i) {
      assert(coff < ctrlay->n);
      memcpy(stages[i]->ctrbuf, ctrbuf + coff * mbn, mbn * stages[i]->ctrlay->n * sizeof(double));
      coff += stages[i]->ctrlay->n;
    }
    assert(coff == ctrlay->n);
  }


  {
    unsigned int aoff = 0;
    for (unsigned int i = 0; i < stages.size(); ++i) {
      assert(aoff < adjlay->n);
      if (!(tgtlock & (1 << i))) {
        memset(adjbuf + aoff * mbn, 0, mbn * stages[i]->outlay->n * sizeof(double));
      }
      memcpy(stages[i]->adjbuf, adjbuf + aoff * mbn, mbn * stages[i]->outlay->n * sizeof(double));
      aoff += stages[i]->outlay->n;
    }
    assert(aoff == adjlay->n);
  }



  Stage *proj = initial();
  assert(proj->ctxlay->n == ctxlay->n);
  memcpy(proj->ctxbuf, ctxbuf, mbn * ctxlay->n * sizeof(double));

  proj->generate();
  for (unsigned int j = 0, jn = mbn * proj->outlay->n; j < jn; ++j)
    proj->outbuf[j] += proj->adjbuf[j];

  for (unsigned int i = 1; i < stages.size(); ++i) {
    Stage *lastproj = stages[i - 1];
    Stage *proj = stages[i];

    assert(proj->ctxlay->n == ctxlay->n + lastproj->outlay->n);
    assert(proj->mbn == lastproj->mbn);
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      memcpy(
        proj->ctxbuf + mbi * proj->ctxlay->n + 0,
        ctxbuf + mbi * ctxlay->n,
        sizeof(double) * ctxlay->n
      );
      memcpy(
        proj->ctxbuf + mbi * proj->ctxlay->n + ctxlay->n,
        lastproj->outbuf + mbi * lastproj->outlay->n,
        sizeof(double) * lastproj->outlay->n
      );
    }

    proj->generate();
    for (unsigned int j = 0, jn = mbn * proj->outlay->n; j < jn; ++j) {
      proj->outbuf[j] += proj->adjbuf[j];
    }
  }

  proj = stages[stages.size() - 1];
  assert(proj->outlay->n == outlay->n);
  memcpy(outbuf, proj->outbuf, sizeof(double) * mbn * outlay->n);
}

void Pipeline::uptarget() {
  Stage *proj = final();
  assert(proj->outlay->n == outlay->n);
  memcpy(proj->adjbuf, outbuf, mbn * sizeof(double) * outlay->n);

  for (int i = stages.size() - 2; i >= 0; --i) {
    Stage *lastproj = proj;
    proj = stages[i];
    unsigned int dim = lround(sqrt(lastproj->outlay->n / 3));
    assert(dim * dim * 3 == lastproj->outlay->n);
    assert(dim * dim * 3 == proj->outlay->n * 4);

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      twiddle3(
        lastproj->adjbuf + mbi * lastproj->outlay->n,
        dim, dim,
        proj->adjbuf + mbi * proj->outlay->n, NULL
      );
    }
  }
  // adjbufs have old target

  {
    unsigned int coff = 0;
    for (unsigned int i = 0; i < stages.size(); ++i) {
      assert(coff < ctrlay->n);
      memcpy(stages[i]->ctrbuf, ctrbuf + coff * mbn, mbn * stages[i]->ctrlay->n * sizeof(double));
      coff += stages[i]->ctrlay->n;
    }
    assert(coff == ctrlay->n);
  }
  // controls are copied


  proj = initial();
  assert(proj->ctxlay->n == ctxlay->n);
  memcpy(proj->ctxbuf, ctxbuf, mbn * ctxlay->n * sizeof(double));


  for (unsigned int i = 0; i < stages.size(); ++i) {
    Stage *lastproj = i > 0 ? stages[i-1] : NULL;
    Stage *proj = stages[i];

    if (lastproj) {
      assert(proj->ctxlay->n == ctxlay->n + lastproj->outlay->n);
      assert(proj->mbn == lastproj->mbn);

      for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
        memcpy(
          proj->ctxbuf + mbi * proj->ctxlay->n,
          ctxbuf + mbi * ctxlay->n,
          ctxlay->n * sizeof(double)
        );
        memcpy(
          proj->ctxbuf + mbi * proj->ctxlay->n + ctxlay->n,
          lastproj->outbuf + mbi * lastproj->outlay->n,
          lastproj->outlay->n * sizeof(double)
        );
      }
    }

    proj->generate();

    // set outbuf to target if target locked

    if (tgtlock & (1 << i))
      for (unsigned int j = 0, jn = mbn * proj->outlay->n; j < jn; ++j) {
        proj->outbuf[j] = proj->adjbuf[j];
    }
  }

  proj = final();
  assert(proj->outlay->n == outlay->n);
  memcpy(outbuf, proj->outbuf, sizeof(double) * mbn * outlay->n);
  // copied new target
}


void Pipeline::readjust() {
  Stage *proj = final();
  assert(proj->outlay->n == outlay->n);
  memcpy(proj->adjbuf, outbuf, mbn * sizeof(double) * outlay->n);

  for (int i = stages.size() - 2; i >= 0; --i) {
    Stage *lastproj = proj;
    proj = stages[i];
    unsigned int dim = lround(sqrt(lastproj->outlay->n / 3));
    assert(dim * dim * 3 == lastproj->outlay->n);
    assert(dim * dim * 3 == proj->outlay->n * 4);

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      twiddle3(
        lastproj->adjbuf + mbi * lastproj->outlay->n,
        dim, dim,
        proj->adjbuf + mbi * proj->outlay->n, NULL
      );
    }
  }

  {
    unsigned int coff = 0;
    for (unsigned int i = 0; i < stages.size(); ++i) {
      assert(coff < ctrlay->n);
      memcpy(stages[i]->ctrbuf, ctrbuf + coff * mbn, mbn * stages[i]->ctrlay->n * sizeof(double));
      coff += stages[i]->ctrlay->n;
    }
    assert(coff == ctrlay->n);
  }


  proj = initial();
  assert(proj->ctxlay->n == ctxlay->n);
  memcpy(proj->ctxbuf, ctxbuf, mbn * ctxlay->n * sizeof(double));


  for (unsigned int i = 0; i < stages.size(); ++i) {
    Stage *lastproj = i > 0 ? stages[i-1] : NULL;
    Stage *proj = stages[i];

    if (lastproj) {
      assert(proj->ctxlay->n == ctxlay->n + lastproj->outlay->n);
      assert(proj->mbn == lastproj->mbn);

      for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
        memcpy(
          proj->ctxbuf + mbi * proj->ctxlay->n,
          ctxbuf + mbi * ctxlay->n,
          ctxlay->n * sizeof(double)
        );
        memcpy(
          proj->ctxbuf + mbi * proj->ctxlay->n + ctxlay->n,
          lastproj->outbuf + mbi * lastproj->outlay->n,
          lastproj->outlay->n * sizeof(double)
        );
      }
    }

    proj->generate();

//fprintf(stderr, "adjbuf: ");
    for (unsigned int j = 0, jn = mbn * proj->outlay->n; j < jn; ++j) {
if (tgtlock & (1 << i))  {
      double z = proj->adjbuf[j];
      proj->adjbuf[j] -= proj->outbuf[j];
      proj->outbuf[j] = z;
} else {
      proj->adjbuf[j] = 0;
}

//if (j < 20) { fprintf(stderr, "%lf ", proj->adjbuf[j]); }
    }
//fprintf(stderr, "\n");
  }






  {
    unsigned int aoff = 0;
    for (unsigned int i = 0; i < stages.size(); ++i) {
      assert(aoff < adjlay->n);
      memcpy(
        adjbuf + aoff * mbn,
        stages[i]->adjbuf,
        mbn * stages[i]->outlay->n * sizeof(double)
      );
      aoff += stages[i]->outlay->n;
    }
    assert(aoff == adjlay->n);
  }
}

void Pipeline::fix(unsigned int iters, double blend) {
  double *bak_ctr = new double[ctrlay->n * mbn];
  double *bak_out = new double[outlay->n * mbn];

  for (int i = 0; i < iters; ++i) {
    memcpy(bak_ctr, ctrbuf, sizeof(double) * ctrlay->n * mbn);
    reencode();

    for (unsigned int j = 0, jn = mbn * ctrlay->n; j < jn; ++j)
      ctrbuf[j] = blend * ctrbuf[j] + (1.0 - blend) * bak_ctr[j];

    memcpy(bak_out, outbuf, sizeof(double) * outlay->n * mbn);
    generate();

    for (unsigned int j = 0, jn = mbn * outlay->n; j < jn; ++j)
      outbuf[j] = blend * outbuf[j] + (1.0 - blend) * bak_out[j];
  }

  delete[] bak_ctr;
  delete[] bak_out;
}

void Pipeline::reencode() {
  assert(stages.size());
  Stage *proj = final();
  memcpy(proj->adjbuf, outbuf, mbn * sizeof(double) * outlay->n);

  for (int i = stages.size() - 2; i >= 0; --i) {
    Stage *lastproj = proj;
    proj = stages[i];
    unsigned int dim = lround(sqrt(lastproj->outlay->n / 3));
    assert(dim * dim * 3 == lastproj->outlay->n);
    assert(dim * dim * 3 == proj->outlay->n * 4);
    assert(lastproj->tgtlay->n == 3 * proj->outlay->n);
    assert(lastproj->ctxlay->n == proj->outlay->n + ctxlay->n);

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      twiddle3(
        lastproj->adjbuf + mbi * lastproj->outlay->n,
        dim, dim,
        lastproj->ctxbuf + mbi * lastproj->ctxlay->n + ctxlay->n,
        lastproj->tgtbuf + mbi * lastproj->tgtlay->n
      );
      memcpy(
        proj->adjbuf + mbi * proj->outlay->n,
        lastproj->ctxbuf + mbi * lastproj->ctxlay->n + ctxlay->n,
        sizeof(double) * proj->outlay->n
      );
    }
  }

  proj = initial();
  assert(proj->ctxlay->n == ctxlay->n);
  assert(proj->tgtlay->n == proj->outlay->n);
  memcpy(proj->tgtbuf, proj->adjbuf, sizeof(double) * mbn * proj->outlay->n);

  for (unsigned int i = 0; i < stages.size(); ++i) {
    Stage *proj = stages[i];

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      memcpy(
        proj->ctxbuf + mbi * proj->ctxlay->n + 0,
        ctxbuf + mbi * ctxlay->n,
        sizeof(double) * ctxlay->n
      );
      if (i > 0) {
        Stage *lastproj = stages[i - 1];
        assert(proj->ctxlay->n == ctxlay->n + lastproj->outlay->n);
        memcpy(
          proj->ctxbuf + mbi * proj->ctxlay->n + ctxlay->n,
          lastproj->outbuf + mbi * lastproj->outlay->n,
          sizeof(double) * lastproj->outlay->n
        );
      }
    }

    if (!(ctrlock & (1 << i)))
      proj->reencode(true);

    proj->generate();
  }

  proj = final();
  assert(outlay->n == proj->outlay->n);
  memcpy(outbuf, proj->outbuf, sizeof(double) * mbn * outlay->n);

  {
    unsigned int coff = 0;
    for (unsigned int i = 0; i < stages.size(); ++i) {
      assert(coff < ctrlay->n);
      memcpy(
        ctrbuf + coff * mbn,
        stages[i]->ctrbuf,
        mbn * stages[i]->ctrlay->n * sizeof(double)
      );
      coff += stages[i]->ctrlay->n;
    }
    assert(coff == ctrlay->n);
  }
}


static void _padzcut(double *src, int dx, int dy, double z, int w, int h, double *dst) {
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {

      double rx, ry;
//      if (z > 1.0) {
        rx = (((double)w/2.0) + (x - dx - ((double)w/2.0)) / z);
        ry = (((double)h/2.0) + (y - dy - ((double)h/2.0)) / z);
//      } else {
//        rx = (((double)w/2.0) + (x - ((double)w/2.0)) / z - dx);
//        ry = (((double)h/2.0) + (y - ((double)h/2.0)) / z - dy);
//      }

      int rx0 = floorl(rx), rx1 = rx0 + 1;
      int ry0 = floorl(ry), ry1 = ry0 + 1;
      double bx = rx - rx0;
      double by = ry - ry0;

//      {
//        bx = (bx > 0.5 ? 1.0 : 0.0);
//        by = (by > 0.5 ? 1.0 : 0.0);
//      }

      if (rx0 < 0) { rx0 = 0; } if (rx0 >= w) { rx0 = w - 1; }
      if (rx1 < 1) { rx1 = 1; } if (rx1 >= w) { rx1 = w - 1; }
      if (ry0 < 0) { ry0 = 0; } if (ry0 >= h) { ry0 = h - 1; }
      if (ry1 < 0) { ry1 = 0; } if (ry1 >= h) { ry1 = h - 1; }

      for (int c = 0; c < 3; ++c) {
        *dst++ = 
          (1.0-bx) * (1.0-by) * src[ry0 * w * 3 + rx0 * 3 + c] +
          (bx) * (1.0-by) * src[ry0 * w * 3 + rx1 * 3 + c] +
          (1.0-bx) * (by) * src[ry1 * w * 3 + rx0 * 3 + c] +
          (bx) * (by) * src[ry1 * w * 3 + rx1 * 3 + c];
      }
    }
  }
}


static double _err2(double *outbuf, double *tgtbuf, unsigned int n) {
  double e2 = 0;
  for (unsigned int j = 0; j < n; ++j)
    e2 += (outbuf[j] - tgtbuf[j]) * (outbuf[j] - tgtbuf[j]);
  e2 /= (double)n;
  return e2;
}

static double _shifterr(Pipeline *pipe, double *srcbuf, int dx, int dy, double z) {
  double *tgtbuf = new double[pipe->outlay->n];
  double *outbuf = new double[pipe->outlay->n];

  assert(pipe->outlay->n % 3 == 0);
  int dim = lround(sqrt((double)pipe->outlay->n / 3.0));
  assert(dim * dim * 3 == pipe->outlay->n);

  _padzcut(srcbuf, dx, dy, z, dim, dim, tgtbuf);
  memcpy(pipe->outbuf, tgtbuf, sizeof(double) * pipe->outlay->n);
  pipe->reencode();
  pipe->generate();
  memcpy(outbuf, pipe->outbuf, sizeof(double) * pipe->outlay->n);
  double e2 = _err2(outbuf, tgtbuf, pipe->outlay->n);

  delete[] tgtbuf;
  delete[] outbuf;

  return e2;
}

void Pipeline::autolign(unsigned int iters, int dzoom) {
  assert(mbn == 1);
  assert(outlay->n % 3 == 0);
  int dim = lround(sqrt((double)outlay->n / 3.0));
  assert(dim * dim * 3 == outlay->n);

  double *srcbuf = new double[outlay->n];

  ctrlock = 0;
  tgtlock = -1;
  memcpy(srcbuf, outbuf, sizeof(double) * outlay->n);

  struct Pos { int dx, dy, z;
    Pos() { dx = 0; dy = 0; z = 0; }
    Pos(int _dx, int _dy, int _z) { dx = _dx; dy = _dy; z = _z; }
    Pos(const Pos &p) { dx = p.dx; dy = p.dy; z = p.z; }
    bool operator < (const Pos &p) const {
      if (dx < p.dx) {
        return true;
      } else if (dx > p.dx) {
        return false;
      }

      if (dy < p.dy) {
        return true;
      } else if (dy > p.dy) {
        return false;
      }

      if (z < p.z) {
        return true;
      } else {
        return false;
      }
    }

    bool operator == (const Pos &p) const {
      return (dx == p.dx && dx == p.dy && z == p.z);
    }
  };
  std::multimap<double, Pos> errpos;
  std::set<Pos> seen;

  Pos best = {0, 0, 128};
  double orige = -1;
  Pos orig = best;
  std::multimap<double, Pos> q;
  q.insert(std::make_pair(0.0, best));

  for (int i = 0; i < iters && q.begin() != q.end(); ++i) {
    Pos x;
    double score = q.begin()->first;
    x = q.begin()->second;
    q.erase(q.begin());

    double ez = exp(((double)x.z - 128.0) / 64.0);
    double e = _shifterr(this, srcbuf, x.dx, x.dy, ez);
    if (i == 0)
      orige = e;
    errpos.insert(std::make_pair(e, x));

    score = e;

    { Pos tmp(x.dx - 1, x.dy + 0, x.z); if (!seen.count(tmp)) { q.insert(std::make_pair(score, tmp)); seen.insert(tmp); } }
    { Pos tmp(x.dx + 1, x.dy + 0, x.z); if (!seen.count(tmp)) { q.insert(std::make_pair(score, tmp)); seen.insert(tmp); } }
    { Pos tmp(x.dx + 0, x.dy - 1, x.z); if (!seen.count(tmp)) { q.insert(std::make_pair(score, tmp)); seen.insert(tmp); } }
    { Pos tmp(x.dx + 0, x.dy + 1, x.z); if (!seen.count(tmp)) { q.insert(std::make_pair(score, tmp)); seen.insert(tmp); } }
    if (dzoom) {
      { Pos tmp(x.dx + 0, x.dy + 0, x.z - dzoom); if (!seen.count(tmp)) { q.insert(std::make_pair(score, tmp)); seen.insert(tmp); } }
      { Pos tmp(x.dx + 0, x.dy + 0, x.z + dzoom); if (!seen.count(tmp)) { q.insert(std::make_pair(score, tmp)); seen.insert(tmp); } }
    }

    fprintf(stderr, "autolign dx=%d dy=%d z=%d e=%lf\n", x.dx, x.dy, x.z, e);
  }

  auto errposi = errpos.begin();
  double newe = errposi->first;
  if (1) { // newe < orige * 0.9) {
    Pos newp = errposi->second;
    double newez = exp(((double)newp.z - 128.0) / 64.0);
    fprintf(stderr, "autolign best dx=%d dy=%d z=%lf e=%lf\n", newp.dx, newp.dy, newez, newe);

    _padzcut(srcbuf, newp.dx, newp.dy, newez, dim, dim, outbuf);
  } else {
    memcpy(outbuf, srcbuf, dim * dim * 3 * sizeof(double));
  }

  delete[] srcbuf;
}

void Pipeline::burn(uint32_t which, double nu, double pi) {
  assert(stages.size());
  Stage *proj = final();
  memcpy(proj->adjbuf, outbuf, mbn * sizeof(double) * outlay->n);

  for (int i = stages.size() - 2; i >= 0; --i) {
    Stage *lastproj = proj;
    proj = stages[i];
    unsigned int dim = lround(sqrt(lastproj->outlay->n / 3));
    assert(dim * dim * 3 == lastproj->outlay->n);
    assert(dim * dim * 3 == proj->outlay->n * 4);
    assert(lastproj->tgtlay->n == 3 * proj->outlay->n);
    assert(lastproj->ctxlay->n == proj->outlay->n + ctxlay->n);

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      twiddle3(
        lastproj->adjbuf + mbi * lastproj->outlay->n,
        dim, dim,
        lastproj->ctxbuf + mbi * lastproj->ctxlay->n + ctxlay->n,
        lastproj->tgtbuf + mbi * lastproj->tgtlay->n
      );
      memcpy(
        proj->adjbuf + mbi * proj->outlay->n,
        lastproj->ctxbuf + mbi * lastproj->ctxlay->n + ctxlay->n,
        sizeof(double) * proj->outlay->n
      );
    }
  }

  proj = initial();
  assert(proj->ctxlay->n == ctxlay->n);
  assert(proj->tgtlay->n == proj->outlay->n);
  memcpy(proj->tgtbuf, proj->adjbuf, sizeof(double) * mbn * proj->outlay->n);

  for (unsigned int i = 0; i < stages.size(); ++i) {
    Stage *proj = stages[i];

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      memcpy(
        proj->ctxbuf + mbi * proj->ctxlay->n + 0,
        ctxbuf + mbi * ctxlay->n,
        sizeof(double) * ctxlay->n
      );
    }

    if (which & (1 << i)) {
      // fprintf(stderr, "burning stage %u nu=%lf\n", i, nu);
      proj->burn(nu, pi);
    }
  }
}


void Pipeline::recombine() {
  const unsigned int js = 8;

  assert(stages.size());
  assert(mbn % 2 == 0);
  assert(ctrlay->n % js == 0);

  for (unsigned int mbi = 0; mbi < mbn; mbi += 2) {
    for (unsigned int j = mbi * ctxlay->n, jn = j + ctxlay->n; j < jn; ++j) {
      if (randuint() % 2) {
        unsigned int k = j + ctxlay->n;
        std::swap(ctxbuf[j], ctxbuf[k]);
      }
    }
  }

  unsigned int coff = 0;
  for (unsigned int i = 0; i < stages.size(); ++i) {
    assert(coff < ctrlay->n);
    memcpy(stages[i]->ctrbuf, ctrbuf + coff * mbn, mbn * stages[i]->ctrlay->n * sizeof(double));
    coff += stages[i]->ctrlay->n;
  }
  assert(coff == ctrlay->n);

  for (unsigned int i = 0; i < stages.size(); ++i) {
    assert(mbn == stages[i]->mbn);
    const Layout *sctrlay = stages[i]->ctrlay;
    double *sctrbuf = stages[i]->ctrbuf;

    for (unsigned int mbi = 0; mbi < mbn; mbi += 2) {
      for (unsigned int j = mbi * sctrlay->n, jn = j + sctrlay->n; j < jn; j += js) {
        if (randuint() % 2) {
          for (unsigned int s = j, sn = j + js, t = s + sctrlay->n; s < sn; ++s, ++t) {
            std::swap(sctrbuf[s], sctrbuf[t]);
          }
        }
      }
    }
  }

  {
    unsigned int coff = 0;
    for (unsigned int i = 0; i < stages.size(); ++i) {
      assert(coff < ctrlay->n);
      memcpy(
        ctrbuf + coff * mbn,
        stages[i]->ctrbuf,
        mbn * stages[i]->ctrlay->n * sizeof(double)
      );
      coff += stages[i]->ctrlay->n;
    }
    assert(coff == ctrlay->n);
  }
}





void Pipeline::condition(uint32_t which, double yo, double wu) {
  assert(stages.size());
  Stage *proj = final();
  memcpy(proj->adjbuf, outbuf, mbn * sizeof(double) * outlay->n);

  for (int i = stages.size() - 2; i >= 0; --i) {
    Stage *lastproj = proj;
    proj = stages[i];
    unsigned int dim = lround(sqrt(lastproj->outlay->n / 3));
    assert(dim * dim * 3 == lastproj->outlay->n);
    assert(dim * dim * 3 == proj->outlay->n * 4);
    assert(lastproj->tgtlay->n == 3 * proj->outlay->n);
    assert(lastproj->ctxlay->n == proj->outlay->n + ctxlay->n);

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      twiddle3(
        lastproj->adjbuf + mbi * lastproj->outlay->n,
        dim, dim,
        lastproj->ctxbuf + mbi * lastproj->ctxlay->n + ctxlay->n,
        lastproj->tgtbuf + mbi * lastproj->tgtlay->n
      );
      memcpy(
        proj->adjbuf + mbi * proj->outlay->n,
        lastproj->ctxbuf + mbi * lastproj->ctxlay->n + ctxlay->n,
        sizeof(double) * proj->outlay->n
      );
    }
  }

  proj = initial();
  assert(proj->ctxlay->n == ctxlay->n);
  assert(proj->tgtlay->n == proj->outlay->n);
  memcpy(proj->tgtbuf, proj->adjbuf, sizeof(double) * mbn * proj->outlay->n);

  for (unsigned int i = 0; i < stages.size(); ++i) {
    Stage *proj = stages[i];

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      memcpy(
        proj->ctxbuf + mbi * proj->ctxlay->n + 0,
        ctxbuf + mbi * ctxlay->n,
        sizeof(double) * ctxlay->n
      );
    }

    if (which & (1 << i)) {
      // fprintf(stderr, "burning stage %u nu=%lf\n", i, nu);
      proj->condition(yo, wu);
    }
  }
}






#if 0
void Pipeline::burnin(
) {
  generate();


  for (unsigned int i = 0; i < stages.size(); ++i) {
    Stage *lastproj = i > 0 ? stages[i-1] : NULL;
    Stage *proj = stages[i];

    if (lastproj) {
      unsigned int n = proj->ctxlay->n;
      assert(n == lastproj->outlay->n);
      assert(proj->mbn == lastproj->mbn);
      memcpy(proj->ctxbuf, lastproj->outbuf, n * mbn * sizeof(double));
    }
    memset(proj->adjbuf, 0, tgtlay->n * mbn * sizeof(double));

    proj->generate();

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      unsigned int boff = mbi * proj->tgtlay->n;
      unsigned int ooff = mbi * proj->outlay->n + (proj->outlay->n - proj->tgtlay->n);
      for (unsigned int j = 0, jn = proj->tgtlay->n; j < jn; ++j, ++boff, ++ooff) {
        proj->adjbuf[boff] = proj->tgtbuf[boff] - proj->outbuf[ooff];
        proj->outbuf[ooff] = proj->tgtbuf[boff];
      }
    }
  }




  generate();
}
#endif


void Pipeline::scramble(double mean, double dev) {
  for (unsigned int j = 0, jn = ctrlay->n * mbn; j < jn; ++j) {
    ctrbuf[j] = sigmoid(mean + randgauss() * dev);
  }
}
 
void Pipeline::load() {
  for (auto pi = stages.begin(); pi != stages.end(); ++pi) {
    Stage *proj = *pi;
    proj->load();
  }
}

void Pipeline::save() {
  for (auto pi = stages.begin(); pi != stages.end(); ++pi) {
    Stage *proj = *pi;
    proj->save();
  }
}

void Pipeline::load_ctx_bytes(const uint8_t *bbuf) {
  for (unsigned int j = 0, jn = mbn * ctxlay->n; j < jn; ++j) {
    ctxbuf[j] = ((double)bbuf[j] + 0.5) / 256.0;
  }
}

void Pipeline::save_ctx_bytes(uint8_t *bbuf) {
  for (unsigned int j = 0, jn = mbn * ctxlay->n; j < jn; ++j) {
    long z = (long)(ctxbuf[j] * 256.0);
    if (z > 255) { z = 255; }
    if (z < 0) { z = 0; }
    bbuf[j] = z;
  }
}

void Pipeline::load_out_bytes(const uint8_t *bbuf) {
  for (unsigned int j = 0, jn = mbn * outlay->n; j < jn; ++j) {
    outbuf[j] = ((double)bbuf[j] + 0.5) / 256.0;
  }
}

bool Pipeline::load_ctx_bytes(FILE *infp) {
  int ret;

  uint8_t *bctxbuf = new uint8_t[mbn * ctxlay->n];

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    ret = fread(bctxbuf + mbi * ctxlay->n, 1, ctxlay->n, infp);
    if (ret != ctxlay->n) {
      delete[] bctxbuf;
      return false;
    }
  }

  load_ctx_bytes(bctxbuf);
  delete[] bctxbuf;

  return true;
}

bool Pipeline::load_out_bytes(FILE *infp) {
  int ret;

  uint8_t *boutbuf = new uint8_t[mbn * outlay->n];

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    ret = fread(boutbuf + mbi * outlay->n, 1, outlay->n, infp);
    if (ret != outlay->n) {
      delete[] boutbuf;
      return false;
    }
  }

  load_out_bytes(boutbuf);
  delete[] boutbuf;

  return true;
}

void Pipeline::report(const char *prog) {
  for (auto i = stages.begin(); i != stages.end(); ++i) 
    (*i)->report(prog);
}

}
