#!/bin/bash

LAYOUTS = input.lay

MAPFILES = dis.map

BINDIR = ..

all: $(LAYOUTS) $(MAPFILES)

dis.map:
	$(BINDIR)/maketron --new $@
	$(BINDIR)/maketron --add $@ 2 256 256  3   256 256 8 1  0 3
	$(BINDIR)/maketron --add $@ 2 256 256  8  256 256 8  1  0 3
	$(BINDIR)/maketron --add $@ 4 256 256  8  128 128  8  0 -1 0
	$(BINDIR)/maketron --add $@ 2 128 128  8  128 128 16  1  0 3
	$(BINDIR)/maketron --add $@ 4 128 128  16  64 64 16  0 -1 0
	$(BINDIR)/maketron --add $@ 2 64   64  16  64 64 16  1  0 3
	$(BINDIR)/maketron --add $@ 4 64   64  16  32 32 16  0 -1 0
	$(BINDIR)/maketron --add $@ 2 32   32  16  32 32 16  1  0 3
	$(BINDIR)/maketron --add $@ 4 32   32  16  16 16 16  0 -1 0
	$(BINDIR)/maketron --add $@ 2 16   16  16  16 16 16  1  0 3
	$(BINDIR)/maketron --add $@ 4 16   16  16   8  8 16  0 -1 0
	$(BINDIR)/maketron --add $@ 1  8    8  16   8  8 16  1  0 3

	$(BINDIR)/maketron --add $@ 0  8    8  16   1  1 256   0 0 3
	$(BINDIR)/maketron --add $@ 0  1    1  256   1  1 128   0 0 3
	$(BINDIR)/maketron --add $@ 0  1    1  128   1  1 1   0 0 0
  

	$(BINDIR)/maketron --randomize $@ 2

input.lay:
	$(BINDIR)/makelay -grid 256 1 3 > $@

clean:
	rm -f *.lay *.wire *.top *.map
