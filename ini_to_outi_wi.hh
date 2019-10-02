{
  switch (head->type) {
  case Supertron::Layer::TYPE_FULL:
    if (i < 0 || i >= head->outn) {
      *outip = -1;
      *wip = -1;
      return false;
    }

    *outip = i;
    *wip = ini * head->outn + *outip;
    return true;

  case Supertron::Layer::TYPE_LOCAL:
    {
      int iz = (ini % head->ic);
      ini /= head->ic;
      int ix = ini % head->iw;
      int iy = ini / head->iw;

      int d21 = (head->d * 2 + 1);
      int s2 = (head->s > 0) ? (1 << head->s) : 1;
      int m = (d21 * d21 * head->oc);
      if (i >= m * s2 * s2) {
        *outip = -1;
        *wip = -1;
        return false;
      }

      int rxy = i / m;
      int rx = rxy % s2;
      int ry = rxy / s2;
      i %= m;

      int oz = i % head->oc;
      i /= head->oc;
      int dx0 = i % d21;
      int dy0 = i / d21;
      int dx = dx0 - head->d;
      int dy = dy0 - head->d;

      int ox = shl((ix + dx), head->s) + rx;
      int oy = shl((iy + dy), head->s) + ry;

      if (ox < 0 || ox >= head->ow || oy < 0 || oy >= head->oh) {
        *outip = -1;
        *wip = -1;
        return true;
      }
      
      *outip = oz + head->oc * (ox + head->ow * oy);
      *wip = *outip + head->outn * (iz + head->ic * (dx0 + d21 * (dy0)));
      return true;
    }

  case Supertron::Layer::TYPE_CONV:
    {
      int iz = (ini % head->ic);
      ini /= head->ic;
      int ix = ini % head->iw;
      int iy = ini / head->iw;

      int d21 = (head->d * 2 + 1);
      int s2 = (head->s > 0) ? (1 << head->s) : 1;
      int m = (d21 * d21 * head->oc);
      if (i >= m * s2 * s2) {
        *outip = -1;
        *wip = -1;
        return false;
      }

      int rxy = i / m;
      int rx = rxy % s2;
      int ry = rxy / s2;
      i %= m;

      int oz = i % head->oc;
      i /= head->oc;
      int dx0 = i % d21;
      int dy0 = i / d21;
      int dx = dx0 - head->d;
      int dy = dy0 - head->d;

      ix += dx;
      iy += dy;
      while (ix < 0) ix += head->iw;
      ix %= head->iw;
      while (iy < 0) iy += head->ih;
      iy %= head->ih;

      int ox = shl(ix, head->s) + rx;
      int oy = shl(iy, head->s) + ry;

//      if (ox < 0 || ox >= head->ow || oy < 0 || oy >= head->oh) {
//        *outip = -1;
//        *wip = -1;
//        return true;
//      }

      *outip = oz + head->oc * (ox + head->ow * oy);
      *wip = rxy + s2 * s2 * (oz + head->oc * (iz + head->ic * (dx0 + d21 * (dy0))));
      return true;
    }

  case Supertron::Layer::TYPE_YCONV:
    {
      int iz = (ini % head->ic);
      int ix = (int)(ini / head->ic) % head->iw;
      int iy = (int)(ini / head->ic) / head->iw;

      int d21 = (head->d * 2 + 1);
      int s2 = (head->s > 0) ? (1 << head->s) : 1;
      int m = (d21 * d21 * head->oc);
      if (i >= m * s2 * s2) {
        *outip = -1;
        *wip = -1;
        return false;
      }

      int rxy = i / m;
      int rx = rxy % s2;
      int ry = rxy / s2;
      i %= m;

      int dx0 = (int)(i / head->oc) % d21;
      int dy0 = (int)(i / head->oc) / d21;
      int dx = dx0 - head->d;
      int dy = dy0 - head->d;

      int oz = i % head->oc;
      int ox = shl((ix + dx), head->s) + rx;
      int oy = shl((iy + dy), head->s) + ry;

      if (ox < 0 || ox >= head->ow || oy < 0 || oy >= head->oh) {
        *outip = -1;
        *wip = -1;
        return true;
      }

      *outip = oz + head->oc * (ox + head->ow * oy);

      *wip = ry + s2 *
        (oz + (head->oc * (ox + head->ow *
          (iz + head->ic * (dx0 + d21 * (dy0)))
        )));

      return true;
    }

  default:
    *outip = -1;
    *wip = -1;
    return false;
  }

  //return false;
}
