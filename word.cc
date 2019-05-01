#define __MAKEMORE_WORD_CC__ 1
#include "word.hh"
#include "strutils.hh"

namespace makemore {

void line_to_strvec(const Line &wv, strvec *svp) {
  unsigned int sn = wv.size();
  strvec &sv = *svp;
  sv.resize(sn);

  for (unsigned int si = 0; si < sn; ++si) {
    const Word &w = wv[si];

    if (w.cuda) {
fprintf(stderr, "DECUDING LINE len=%d\n", w.len);
      sv[si].resize(w.len);
      decude((const uint8_t *)w.ptr, w.len, (uint8_t *)sv[si].data());
    } else {
      sv[si].assign((const char *)w.ptr, w.len);
    }
  }
}

void strvec_to_line(const strvec &sv, Line *wvp) {
  unsigned int wn = sv.size();
  Line &wv = *wvp;
  wv.resize(wn);

  for (unsigned int wi = 0; wi < wn; ++wi)
    wv[wi].copy(sv[wi]);
}

}
