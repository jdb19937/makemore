#!/bin/bash

LAYOUTS = context.lay control.lay target.lay

TOPOLOGIES =

MAPFILES = gen.map

BINDIR = ..

all: $(DATASETS) $(LAYOUTS) $(WIRINGS) $(TOPOLOGIES) $(MAPFILES)


geninput.lay: context.lay control.lay
	$(BINDIR)/catlay $^ > $@
genoutput.lay: target.lay
	$(BINDIR)/catlay $^ > $@

gen.map:
	$(BINDIR)/maketron --new $@

	$(BINDIR)/maketron --add $@ 2 32 32  3   32 32 16   1 0 3

	$(BINDIR)/maketron --add $@ 6 32 32 16   32 32 20   0 0 0
	$(BINDIR)/maketron --add $@ 7 32 32 20   32 32 24   0 0 0
	$(BINDIR)/maketron --add $@ 2 32 32 24   32 32 16   1 0 3

	$(BINDIR)/maketron --add $@ 6 32 32 16   32 32 20   0 0 0
	$(BINDIR)/maketron --add $@ 7 32 32 20   32 32 24   0 0 0
	$(BINDIR)/maketron --add $@ 2 32 32 24   32 32 16   1 0 3

	$(BINDIR)/maketron --add $@ 6 32 32 16   32 32 20   0 0 0
	$(BINDIR)/maketron --add $@ 7 32 32 20   32 32 24   0 0 0
	$(BINDIR)/maketron --add $@ 2 32 32 24   32 32 16   1 0 3

	$(BINDIR)/maketron --add $@ 6 32 32 16   32 32 20   0 0 0
	$(BINDIR)/maketron --add $@ 7 32 32 20   32 32 24   0 0 0
	$(BINDIR)/maketron --add $@ 2 32 32 24   32 32 16   1 0 3

	$(BINDIR)/maketron --add $@ 6 32 32 16   32 32 20   0 0 0
	$(BINDIR)/maketron --add $@ 7 32 32 20   32 32 24   0 0 0
	$(BINDIR)/maketron --add $@ 2 32 32 24   32 32 16   1 0 3

	$(BINDIR)/maketron --add $@ 6 32 32 16   32 32 20   0 0 0
	$(BINDIR)/maketron --add $@ 7 32 32 20   32 32 24   0 0 0
	$(BINDIR)/maketron --add $@ 2 32 32 24   32 32 16   1 0 3

	$(BINDIR)/maketron --add $@ 6 32 32 16   32 32 20   0 0 0
	$(BINDIR)/maketron --add $@ 7 32 32 20   32 32 24   0 0 0
	$(BINDIR)/maketron --add $@ 2 32 32 24   32 32 16   1 0 3

	$(BINDIR)/maketron --add $@ 6 32 32 16   32 32 20   0 0 0
	$(BINDIR)/maketron --add $@ 7 32 32 20   32 32 24   0 0 0
	$(BINDIR)/maketron --add $@ 2 32 32 24   32 32 16   1 0 3

	$(BINDIR)/maketron --add $@ 6 32 32 16   32 32 20   0 0 0
	$(BINDIR)/maketron --add $@ 7 32 32 20   32 32 24   0 0 0
	$(BINDIR)/maketron --add $@ 2 32 32 24   32 32 16   1 0 3

	$(BINDIR)/maketron --add $@ 6 32 32 16   32 32 20   0 0 0
	$(BINDIR)/maketron --add $@ 7 32 32 20   32 32 24   0 0 0
	$(BINDIR)/maketron --add $@ 2 32 32 24   32 32 16   1 0 3

	$(BINDIR)/maketron --add $@ 6 32 32 16   32 32 20   0 0 0
	$(BINDIR)/maketron --add $@ 7 32 32 20   32 32 24   0 0 0
	$(BINDIR)/maketron --add $@ 2 32 32 24   32 32 16   1 0 3

	$(BINDIR)/maketron --add $@ 6 32 32 16   32 32 20   0 0 0
	$(BINDIR)/maketron --add $@ 7 32 32 20   32 32 24   0 0 0
	$(BINDIR)/maketron --add $@ 2 32 32 24   32 32 16   1 0 3

	$(BINDIR)/maketron --add $@ 2 32 32 16   32 32  3   1 0 0

	$(BINDIR)/maketron --randomize $@ 1.0


control.lay:
	$(BINDIR)/makelay -grid 32 1 3 > $@
context.lay:
	$(BINDIR)/makelay -center 0 16 > $@
target.lay:
	$(BINDIR)/makelay -grid 32 1 3 > $@

clean:
	rm -f *.lay *.wire *.top *.map
