{
  switch (head->type) {
  case Supertron::Layer::TYPE_FULL:
    if (i == 0) {
      *inip = wi / head->outn;
      if (*inip >= head->inn)
        *inip = -1;
      *outip = wi % head->outn;
      return true;
    } else {
      *inip = -1;
      *outip = -1;
      return false;
    }

  case Supertron::Layer::TYPE_LOCAL:
    {
      if (i != 0) {
        *inip = -1;
        *outip = -1;
        return false;
      }

      int outi = wi % head->outn;
      *outip = outi;
      // int oz = (outi % head->oc);
      outi /= head->oc;
      int ox = outi % head->ow;
      int oy = outi / head->ow;

      int d21 = (head->d * 2 + 1);
      int m = (d21 * d21 * head->ic + 1);
      int di = wi / head->outn;
      if (di >= m - 1) {
        *inip = -1;
        return true;
      }

      int iz = di % head->ic;
      di /= head->ic;
      int dx0 = di % d21;
      int dy0 = di / d21;
      int dx = dx0 - head->d;
      int dy = dy0 - head->d;

      int ix = shr(ox, head->s) - dx;
      int iy = shr(oy, head->s) - dy;
  
      if (ix < 0 || ix >= head->iw || iy < 0 || iy >= head->ih) {
        *inip = -2;
        return true;
      }

      *inip = iz + head->ic * (ix + head->iw * iy);
      return true;
    }
  case Supertron::Layer::TYPE_CONV:
    {
      int s2 = (head->s > 0) ? (1 << head->s) : 1;
      int n = head->ow * head->oh / s2 / s2;

      if (i < 0 || i >= n) {
        *inip = -1;
        *outip = -1;
        return false;
      }

      int d21 = (head->d * 2 + 1);
      int m = (d21 * d21 * head->ic + 1);

      int tmp = wi;
      int rxy = tmp % (s2 * s2); tmp /= (s2 * s2);
      int ry = rxy / s2;
      int rx = rxy % s2;
      int oz = tmp % head->oc; tmp /= head->oc;
      int di = tmp;

      int how = head->ow / s2;
      int ox = rx + s2 * (i % how); i /= how;
      int oy = ry + s2 * i;

      int outi = oz + head->oc * (ox + head->ow * oy);
      *outip = outi;

      if (di >= m - 1) {
        *inip = -1;
        return true;
      }

      int iz = di % head->ic;
      di /= head->ic;
      int dx0 = di % d21;
      int dy0 = di / d21;
      int dx = dx0 - head->d;
      int dy = dy0 - head->d;

      int ix = shr(ox, head->s) - dx;
      int iy = shr(oy, head->s) - dy;
  
      if (ix < 0 || ix >= head->iw || iy < 0 || iy >= head->ih) {
        *inip = -2;
        return true;
      }

      *inip = iz + head->ic * (ix + head->iw * iy);
      return true;
    }
  case Supertron::Layer::TYPE_YCONV:
    {
      int s2 = (head->s > 0) ? (1 << head->s) : 1;
      int n = head->oh / s2;

      if (i < 0 || i >= n) {
        *inip = -1;
        *outip = -1;
        return false;
      }

      int d21 = (head->d * 2 + 1);
      int m = (d21 * d21 * head->ic + 1);

      int tmp = wi;
      int ry = tmp % s2; tmp /= s2;
      int oz = tmp % head->oc; tmp /= head->oc;
      int ox = tmp % head->ow; tmp /= head->ow;
      int oy = ry + s2 * i;
      int di = tmp;

      int outi = oz + head->oc * (ox + head->ow * oy);
      *outip = outi;

      if (di >= m - 1) {
        *inip = -1;
        return true;
      }

      int dx0 = (int)(di / head->ic) % d21;
      int dy0 = (int)(di / head->ic) / d21;
      int dx = dx0 - head->d;
      int dy = dy0 - head->d;

      int iz = di % head->ic;
      int ix = shr(ox, head->s) - dx;
      int iy = shr(oy, head->s) - dy;
  
      if (ix < 0 || ix >= head->iw || iy < 0 || iy >= head->ih) {
        *inip = -2;
        return true;
      }

      *inip = iz + head->ic * (ix + head->iw * iy);
      return true;
    }
  default:
    *inip = -1;
    *outip = -1;
    return false;
  }

  // return false;
}
