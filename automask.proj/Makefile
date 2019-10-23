#!/bin/bash

LAYOUTS = target.lay

TOPOLOGIES = msk.top

MAPFILES = msk.map

BINDIR = ..

all: $(DATASETS) $(LAYOUTS) $(WIRINGS) $(TOPOLOGIES) $(MAPFILES)


mskinput.lay: target.lay
	$(BINDIR)/catlay $^ > $@
mskhidden1.lay:
	$(BINDIR)/makelay -rand 65536 16 > $@
mskhidden2.lay:
	$(BINDIR)/makelay -rand 8192 32 > $@
mskhidden3.lay:
	$(BINDIR)/makelay -rand 65536 16 > $@
mskoutput.lay: target.lay
	$(BINDIR)/catlay $^ > $@

mskhidden1.wire: mskinput.lay mskhidden1.lay
	$(BINDIR)/wireup --minv 4 $^ > $@
mskhidden2.wire: mskhidden1.lay mskhidden2.lay
	$(BINDIR)/wireup --minv 4 $^ > $@
mskhidden3.wire: mskhidden2.lay mskhidden3.lay
	$(BINDIR)/wireup --minv 4 $^ > $@
mskoutput.wire: mskhidden3.lay mskoutput.lay
	$(BINDIR)/wireup --minv 4 $^ > $@
msk.top: mskhidden1.wire mskhidden2.wire mskhidden3.wire mskoutput.wire
	$(BINDIR)/maketop $^ > $@
msk.map: msk.top
	$(BINDIR)/makemap --dispersion 3 msk.top msk.map

target.lay:
	$(BINDIR)/makelay -grid 256 1 3 > $@

clean:
	rm -f *.lay *.wire *.top *.map
