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
	$(BINDIR)/maketron --add $@ 0  16  16  4    32  32  4   0 0 1
	$(BINDIR)/maketron --add $@ 1  32  32  4    64  64  4   3 1 1
	$(BINDIR)/maketron --add $@ 1  64  64  4   128 128  4   2 1 1
	$(BINDIR)/maketron --add $@ 1 128 128  4   256 256  8   1 1 1
	$(BINDIR)/maketron --add $@ 2 256 256  8   256 256  8   1 0 1
	$(BINDIR)/maketron --add $@ 2 256 256  8   256 256  8   1 0 1
	$(BINDIR)/maketron --add $@ 2 256 256  8   256 256  3   1 0 0
	$(BINDIR)/maketron --randomize $@ 3


control.lay:
	$(BINDIR)/makelay -grid 16 1 4 > $@
context.lay:
	$(BINDIR)/makelay -center 0 16 > $@
target.lay:
	$(BINDIR)/makelay -grid 256 1 3 > $@

clean:
	rm -f *.lay *.wire *.top *.map
