#define __MAKEMORE_MULTITRON_CC__ 1
#include "multitron.hh"
#include "cudamem.hh"

#include <vector>

Multitron::Multitron(const std::vector<Wiring*> wires, double *weightbuf, unsigned int _npass, unsigned int _mbn) : Tron(0, 0) {
  npass = _npass;
  mbn = _mbn;

  megatrons.clear();
  assert(wires.begin() != wires.end());

  double *wb = weightbuf;
  for (auto wi = wires.begin(); wi != wires.end(); ++wi) {
    Megatron *mt = new Megatron(*wi, wb, mbn);
    megatrons.push_back(mt);
    wb += (*wi)->wn;
  }
  inrn = (*wires.begin())->inn;
  outrn = (*wires.rbegin())->outn + npass;

  inn = inrn * mbn;
  outn = outrn * mbn;

  passbuf = NULL;
  fpassbuf = NULL;
  if (npass > 0) {
    cumake(&passbuf, outrn * mbn);
    cumake(&fpassbuf, outrn * mbn);
  }
}

void Multitron::randomize(double disp) {
  for (auto ti = megatrons.begin(); ti != megatrons.end(); ++ti)
    (*ti)->randomize(disp);
}

Multitron::~Multitron() {
  if (passbuf)
    cufree(passbuf);
  if (fpassbuf)
    cufree(fpassbuf);
}


const double *Multitron::feed(const double *_in, double *_fin) {
  in = _in;
  fin = _fin;

  auto mi = megatrons.begin();
  assert(mi != megatrons.end());
  out = (*mi)->feed(in, fin);
  fout = (*mi)->foutput();

  auto pmi = mi++;
  while (mi != megatrons.end()) {
    out = (*mi)->feed(out, fout);
    fout = (*mi)->foutput();
    ++mi;
  }

  if (npass == 0)
    return out;

  cuzero(fpassbuf, mbn * outrn); 
  cucutpaste(in, out, mbn, inrn, outrn - npass, outrn, passbuf);

  return passbuf;
}

void Multitron::train(double nu) {
  if (npass > 0) {
    assert(fpassbuf);
    cucutadd(fpassbuf, mbn, outrn, outrn - npass, fout);
  }

  for (auto mi = megatrons.rbegin(); mi != megatrons.rend(); ++mi) {
    (*mi)->train(nu);
  }
}

