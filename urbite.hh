#ifndef __MAKEMORE_URBITE_HH__
#define __MAKEMORE_URBITE_HH__ 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>
#include <string>

#include "urb.hh"
#include "parson.hh"

namespace makemore {

struct Urbite {
  std::string nom;
  Parson *_parson;
  Urb *urb;

  Urbite(const char *_nom, Urb *_urb) {
    nom = _nom;
    urb = _urb;
    _parson = NULL;
  }

  Parson *parson() {
    if (_parson) {
      if (nom == _parson->nom)
        return _parson;
      else
        _parson = NULL;
    }

    assert(!_parson);
    _parson = urb->find(nom);

    return _parson;
  }

#if 0
  Parson *resurrect() {
    Parson *p = parson();
    if (p)
      return p;

    _parson = urb->import(nom);
    assert(_parson);
    assert(nom == _parson->nom);
    return _parson;
  }
#endif

};

}
#endif
