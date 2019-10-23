#!/bin/bash

LAYOUTS = seginput.lay segoutput.lay

TOPOLOGIES = seg.top

MAPFILES = seg.map

BINDIR = ..

all: $(DATASETS) $(LAYOUTS) $(TOPOLOGIES) $(MAPFILES)

seginput.lay:
	$(BINDIR)/makelay -grid 256 1 3 > $@
seghidden1.lay:
	$(BINDIR)/makelay -rand 32768 2 > $@
seghidden2.lay:
	$(BINDIR)/makelay -rand 4096 16 > $@
seghidden3.lay:
	$(BINDIR)/makelay -center 256 2 > $@
segoutput.lay:
	$(BINDIR)/makelay -center 6 2 > $@

seghidden1.wire: seginput.lay seghidden1.lay
	$(BINDIR)/wireup --minv 16 $^ > $@
seghidden2.wire: seghidden1.lay seghidden2.lay
	$(BINDIR)/wireup --minv 16 $^ > $@
seghidden3.wire: seghidden2.lay seghidden3.lay
	$(BINDIR)/wireup --minv 16 $^ > $@
segoutput.wire: seghidden3.lay segoutput.lay
	$(BINDIR)/wireup --minv 16 $^ > $@
seg.top: seghidden1.wire seghidden2.wire seghidden3.wire segoutput.wire
	$(BINDIR)/maketop $^ > $@
seg.map: seg.top
	$(BINDIR)/makemap --dispersion 3 seg.top seg.map

clean:
	rm -f *.lay *.wire *.top *.map
