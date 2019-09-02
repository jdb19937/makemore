  {
    switch (head->type) {
    case Supertron::Layer::TYPE_FULL:
      return (head->inn + 1);
    case Supertron::Layer::TYPE_LOCAL:
    case Supertron::Layer::TYPE_CONV:
    case Supertron::Layer::TYPE_YCONV:
      return ((head->d * 2 + 1) * (head->d * 2 + 1) * head->ic + 1);
    }

    return 0;
  }
