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

  Urbite(Urb *_urb) {
    urb = _urb;
    nom = Parson::gen_nom();
    _parson = NULL;
  }

  Urbite(const std::string &_nom, Urb *_urb, Parson *__parson = NULL) {
    nom = _nom;
    urb = _urb;
    _parson = __parson;
  }

  void copy(const Urbite &urbite) {
    nom = urbite.nom;
    urb = urbite.urb;
    _parson = urbite._parson;
  }

  Urbite &operator =(const Urbite &urbite) {
    copy(urbite);
    return *this;
  }

  Urbite(const Urbite &urbite) {
    copy(urbite);
  }

  Parson *parson() {
    assert(Parson::valid_nom(nom.c_str()));

    if (_parson) {
      if (nom == _parson->nom)
        return _parson;
      else
        _parson = NULL;
    }

    assert(!_parson);
    _parson = urb->find(nom);

    if (!_parson) {
      _parson = urb->make(nom.c_str());
    }
    assert(_parson);

    return _parson;
  }

  void become(const std::string &_nom) {
    if (nom == _nom)
      return;
    nom = _nom;
    _parson = NULL;
  }
    

#if 0
  Parson *resurrect() {
    Parson *p = parson();
    if (p)
      return p;

    _parson = urb->make(nom);
    assert(_parson);
    assert(nom == _parson->nom);
    return _parson;
  }
#endif

};

}
#endif
