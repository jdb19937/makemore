{
  switch (head->type) {
  case Supertron::Layer::TYPE_FULL:
    if (i < 0 || i >= head->inn + 1) {
      *inip = -1;
      *wip = -1;
      return false;
    }

    if (i == head->inn) {
      *inip = -1;
    } else {
      *inip = i;
    }

    *wip = i * head->outn + outi;
    return true;

  case Supertron::Layer::TYPE_LOCAL:
    {
      int d21 = (head->d * 2 + 1);
      int m = (d21 * d21 * head->ic + 1);
      if (i >= m) {
        *inip = -1;
        *wip = -1;
        return false;
      }

      *wip = outi + head->outn * i;

      if (i >= m - 1) {
        *inip = -1;
        return true;
      }

      int iz = i % head->ic;
      i /= head->ic;
      int dx0 = i % d21;
      int dy0 = i / d21;
      int dx = dx0 - head->d;
      int dy = dy0 - head->d;

      outi /= head->oc;
      int ox = outi % head->ow;
      int oy = outi / head->ow;

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
      int d21 = (head->d * 2 + 1);
      int m = (d21 * d21 * head->ic + 1);
      if (i >= m) {
        *inip = -1;
        *wip = -1;
        return false;
      }

      int oz = (outi % head->oc);
      outi /= head->oc;
      int ox = outi % head->ow;
      int oy = outi / head->ow;

      int s2 = (head->s > 0) ? (1 << head->s) : 1;
      int rxy = s2 * (oy % s2) + (ox % s2);
      *wip = rxy + s2 * s2 * (oz + head->oc * i);

      if (i >= m - 1) {
        *inip = -1;
        return true;
      }

      int iz = i % head->ic;
      i /= head->ic;
      int dx0 = i % d21;
      int dy0 = i / d21;
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
      int d21 = (head->d * 2 + 1);
      int m = (d21 * d21 * head->ic + 1);
      if (i >= m) {
        *inip = -1;
        *wip = -1;
        return false;
      }

      int oz = (outi % head->oc);
      int ox = (int)(outi / head->oc) % head->ow;
      int oy = (int)(outi / head->oc) / head->ow;

      int s2 = (head->s > 0) ? (1 << head->s) : 1;
      int ry = (oy % s2);

      *wip = ry + s2 * (oz + head->oc * (ox + head->ow * (i)));

      if (i >= m - 1) {
        *inip = -1;
        return true;
      }

      int dx0 = (int)(i / head->ic) % d21;
      int dy0 = (int)(i / head->ic) / d21;
      int dx = dx0 - head->d;
      int dy = dy0 - head->d;

      int iz = i % head->ic;
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
    *wip = -1;
    return false;
  }

  // return false;
}
