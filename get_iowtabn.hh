  {
    switch (head->type) {
    case Supertron::Layer::TYPE_FULL:
      return head->outn;
    case Supertron::Layer::TYPE_LOCAL:
    case Supertron::Layer::TYPE_CONV:
    case Supertron::Layer::TYPE_YCONV:
      {
        int s2 = (head->s > 0) ? (1 << head->s) : 1;
        return s2 * s2 * ((head->d * 2 + 1) * (head->d * 2 + 1) * head->oc);
      }
    }

    return 0;
  }
