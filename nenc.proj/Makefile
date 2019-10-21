#!/bin/bash

LAYOUTS = input.lay control.lay

MAPFILES = enc.map

BINDIR = ..

all: $(LAYOUTS) $(MAPFILES)

encinput.lay: input.lay
	$(BINDIR)/catlay $^ > $@
encoutput.lay: control.lay
	$(BINDIR)/catlay $^ > $@

enc.map:
	$(BINDIR)/maketron --new $@
	$(BINDIR)/maketron --add $@ 1 256 256  3   256 256  4  1  0 1
	$(BINDIR)/maketron --add $@ 1 256 256  4   128 128  4  1 -1 1
	$(BINDIR)/maketron --add $@ 1 128 128  4   64  64   4  2 -1 1
	$(BINDIR)/maketron --add $@ 1  64  64  4   32  32   4  3 -1 1
	$(BINDIR)/maketron --add $@ 0  32  32  4   16  16   4  0  0 1
	$(BINDIR)/maketron --randomize $@ 3

control.lay:
	$(BINDIR)/makelay -grid 16 1 4 > $@
input.lay:
	$(BINDIR)/makelay -grid 256 1 3 > $@

clean:
	rm -f *.lay *.wire *.top *.map
