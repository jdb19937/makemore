  {
    switch (head->type) {
    case Supertron::Layer::TYPE_FULL:
    case Supertron::Layer::TYPE_LOCAL:
      return 1;

    case Supertron::Layer::TYPE_CONV:
      {
        int s2 = (head->s > 0) ? (1 << head->s) : 1;
        return head->ow * head->oh / s2 / s2;
      }
    case Supertron::Layer::TYPE_YCONV:
      {
        int s2 = (head->s > 0) ? (1 << head->s) : 1;
        return head->oh / s2;
      }
    }

    return 0;
  }
