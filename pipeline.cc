#define __MAKEMORE_PIPELINE_CC__ 1
#include "pipeline.hh"

#include "twiddle.hh"
#include "project.hh"
#include "layout.hh"


Pipeline::Pipeline(unsigned int _mbn) {
  mbn = _mbn;
  assert(mbn > 0);

  ctxlay = NULL;
  outlay = NULL;
  ctrlay = new Layout;
  adjlay = new Layout;

  ctrbuf = NULL;
  adjbuf = NULL;
  outbuf = NULL;
  ctxbuf = NULL;

  bctrbuf = NULL;
  badjbuf = NULL;
  boutbuf = NULL;
  bctxbuf = NULL;

}

Project *Pipeline::initial() {
  assert(stages.size());
  return stages[0];
}

Project *Pipeline::final() {
  assert(stages.size());
  return stages[stages.size() - 1];
}

void Pipeline::add_stage(Project *proj) {
  assert(proj->mbn == mbn);

  if (stages.size()) {
    Project *prev = final();
    assert(ctxlay);
    assert(ctxlay->n + prev->outlay->n == proj->ctxlay->n);
  }

  stages.push_back(proj);
  *ctrlay += *proj->ctrlay;
  *adjlay += *proj->outlay;

  _setup();
}

void Pipeline::_setup() {
  assert(stages.size());

  unsigned int f = stages.size() - 1;
  ctxlay = initial()->ctxlay;
  outlay = final()->outlay;

  delete[] ctxbuf; ctxbuf = new double[ctxlay->n * mbn];
  delete[] ctrbuf; ctrbuf = new double[ctrlay->n * mbn];
  delete[] adjbuf; adjbuf = new double[adjlay->n * mbn];
  delete[] outbuf; outbuf = new double[outlay->n * mbn];

  delete[] bctxbuf; bctxbuf = new uint8_t[ctxlay->n * mbn];
  delete[] bctrbuf; bctrbuf = new uint8_t[ctrlay->n * mbn];
  delete[] badjbuf; badjbuf = new uint8_t[adjlay->n * mbn];
  delete[] boutbuf; boutbuf = new uint8_t[outlay->n * mbn];
}

Pipeline::~Pipeline() {
  delete ctrlay;
  delete adjlay;

  delete[] ctxbuf;
  delete[] ctrbuf;
  delete[] adjbuf;
  delete[] outbuf;

  delete[] bctxbuf;
  delete[] bctrbuf;
  delete[] badjbuf;
  delete[] boutbuf;
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



  Project *proj;

  proj = initial();
  assert(ctxlay->n == proj->ctxlay->n);
  memcpy(proj->ctxbuf, ctxbuf, sizeof(double) * mbn * ctxlay->n);
  proj->generate();

  for (unsigned int i = 1; i < stages.size(); ++i) {
    Project *lastproj = stages[i - 1];
    Project *proj = stages[i];

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
  encode_out();

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
      memcpy(stages[i]->adjbuf, adjbuf + aoff * mbn, mbn * stages[i]->outlay->n * sizeof(double));
      aoff += stages[i]->outlay->n;
    }
    assert(aoff == adjlay->n);
  }



  Project *proj = initial();
  assert(proj->ctxlay->n == ctxlay->n);
  memcpy(proj->ctxbuf, ctxbuf, mbn * ctxlay->n * sizeof(double));

  proj->generate();
  for (unsigned int j = 0, jn = mbn * proj->outlay->n; j < jn; ++j)
    proj->outbuf[j] += proj->adjbuf[j];

  for (unsigned int i = 1; i < stages.size(); ++i) {
    Project *lastproj = stages[i - 1];
    Project *proj = stages[i];

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
    for (unsigned int j = 0, jn = mbn * proj->outlay->n; j < jn; ++j)
      proj->outbuf[j] += proj->adjbuf[j];
  }

  proj = stages[stages.size() - 1];
  assert(proj->outlay->n == outlay->n);
  memcpy(outbuf, proj->outbuf, sizeof(double) * mbn * outlay->n);
  encode_out();
}



void Pipeline::readjust() {
  Project *proj = final();
  memcpy(proj->adjbuf, outbuf, mbn * sizeof(double) * outlay->n);

  for (int i = stages.size() - 2; i >= 0; --i) {
    Project *lastproj = proj;
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
    Project *lastproj = i > 0 ? stages[i-1] : NULL;
    Project *proj = stages[i];

    if (lastproj) {
      assert(proj->ctxlay->n == ctxlay->n + lastproj->outlay->n);
      assert(proj->mbn == lastproj->mbn);

      for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
        memcpy(
          proj->ctxbuf + mbi * proj->ctxlay->n,
          ctxbuf + mbi * ctxlay->n,
          ctxlay->n * mbn * sizeof(double)
        );
        memcpy(
          proj->ctxbuf + mbi * proj->ctxlay->n + ctxlay->n,
          lastproj->outbuf + mbi * lastproj->outlay->n,
          lastproj->outlay->n * mbn * sizeof(double)
        );
      }
    }

    proj->generate();

    for (unsigned int j = 0, jn = mbn * proj->outlay->n; j < jn; ++j) {
      double z = proj->adjbuf[j];
      proj->adjbuf[j] -= proj->outbuf[j];
      proj->outbuf[j] = z;
    }
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
    encode_adj();
  }
}


void Pipeline::reencode(uint32_t which) {
  assert(stages.size());
  Project *proj = final();
  memcpy(proj->adjbuf, outbuf, mbn * sizeof(double) * outlay->n);

  for (int i = stages.size() - 2; i >= 0; --i) {
    Project *lastproj = proj;
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
    Project *proj = stages[i];

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      memcpy(
        proj->ctxbuf + mbi * proj->ctxlay->n + 0,
        ctxbuf + mbi * ctxlay->n,
        sizeof(double) * ctxlay->n
      );
    }

    if (which & (1 << i))
      proj->reencode();
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
    encode_ctr();
  }
}

void Pipeline::burnmask(uint32_t which, double nu) {
  assert(stages.size());
  Project *proj = final();
  memcpy(proj->adjbuf, outbuf, mbn * sizeof(double) * outlay->n);

  for (int i = stages.size() - 2; i >= 0; --i) {
    Project *lastproj = proj;
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
  assert(proj->tgtlay->n == proj->outlay->n);
  memcpy(proj->tgtbuf, proj->adjbuf, sizeof(double) * mbn * proj->outlay->n);

  for (unsigned int i = 0; i < stages.size(); ++i) {
    Project *proj = stages[i];

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      memcpy(
        proj->ctxbuf + mbi * proj->ctxlay->n + 0,
        ctxbuf + mbi * ctxlay->n,
        sizeof(double) * ctxlay->n
      );
    }

    if (which & (1 << i))
      proj->burnmask(nu);
  }
}







#if 0
void Pipeline::burnin(
) {
  generate();


  for (unsigned int i = 0; i < stages.size(); ++i) {
    Project *lastproj = i > 0 ? stages[i-1] : NULL;
    Project *proj = stages[i];

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
    Project *proj = *pi;
    proj->load();
  }
}

void Pipeline::save() {
  for (auto pi = stages.begin(); pi != stages.end(); ++pi) {
    Project *proj = *pi;
    proj->save();
  }
}




void Pipeline::encode_ctx() {
  for (unsigned int j = 0, jn = mbn * ctxlay->n; j < jn; ++j) {
    int vl = (int)(ctxbuf[j] * 256.0);
    bctxbuf[j] = vl < 0 ? 0 : vl > 255 ? 255 : vl;
  }
}

void Pipeline::encode_out() {
  for (unsigned int j = 0, jn = mbn * outlay->n; j < jn; ++j) {
    int vl = (int)(outbuf[j] * 256.0);
    boutbuf[j] = vl < 0 ? 0 : vl > 255 ? 255 : vl;
  }
}

void Pipeline::encode_adj() {
  for (unsigned int j = 0, jn = mbn * adjlay->n; j < jn; ++j) {
    double v = adjbuf[j];
    v /= 2.0;
    v += 0.5;
    v *= 256.0;
    long vl = lround(v);
    badjbuf[j] = vl < 0 ? 0 : vl > 255 ? 255 : vl;
  }
}

void Pipeline::encode_ctr() {
  for (unsigned int j = 0, jn = mbn * ctrlay->n; j < jn; ++j) {
    int vl = (int)(ctrbuf[j] * 256.0);
    bctrbuf[j] = vl < 0 ? 0 : vl > 255 ? 255 : vl;
  }
}

bool Pipeline::load_ctx(const uint8_t *bbuf) {
  for (unsigned int j = 0, jn = mbn * ctxlay->n; j < jn; ++j) {
    ctxbuf[j] = ((double)bbuf[j] + 0.5) / 256.0;
  }
  return true;
}

bool Pipeline::load_ctx(FILE *infp) {
  int ret;

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    ret = fread(bctxbuf + mbi * ctxlay->n, 1, ctxlay->n, infp);
    if (ret != ctxlay->n)
      return false;
  }
  return load_ctx(bctxbuf);
}

bool Pipeline::load_ctr(const uint8_t *bbuf) {
  for (unsigned int j = 0, jn = mbn * ctrlay->n; j < jn; ++j) {
    ctrbuf[j] = ((double)bbuf[j] + 0.5) / 256.0;
  }
  return true;
}

bool Pipeline::load_ctr(FILE *infp) {
  int ret;

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    ret = fread(bctrbuf + mbi * ctrlay->n, 1, ctrlay->n, infp);
    if (ret != ctrlay->n)
      return false;
  }
  return load_ctr(bctrbuf);
}

bool Pipeline::load_out(const uint8_t *bbuf) {
  for (unsigned int j = 0, jn = mbn * outlay->n; j < jn; ++j)
    outbuf[j] = ((double)bbuf[j] + 0.5) / 256.0;
  return true;
}

bool Pipeline::load_out(FILE *infp) {
  int ret;

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    ret = fread(boutbuf + mbi * outlay->n, 1, outlay->n, infp);
    if (ret != outlay->n)
      return false;
  }
  return load_out(boutbuf);
}

bool Pipeline::load_adj(const uint8_t *bbuf) {
  for (unsigned int j = 0, jn = mbn * adjlay->n; j < jn; ++j) {
    double z = ((double)bbuf[j]) / 256.0;
    z -= 0.5;
    z *= 2.0;
    adjbuf[j] = z;
  }
  return true;
}

bool Pipeline::load_adj(FILE *infp) {
  int ret;

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    ret = fread(badjbuf + mbi * adjlay->n, 1, adjlay->n, infp);
    if (ret != adjlay->n)
      return false;
  }

  return load_adj(badjbuf);
}
