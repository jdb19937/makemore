#!/bin/bash

MAPFILES = dis.map

BINDIR = ..

all: $(MAPFILES)

dis.map:
	$(BINDIR)/maketron --new $@
	$(BINDIR)/maketron --add $@ 2 32   32  3   32 32 16  1  0 3

	$(BINDIR)/maketron --add $@ 2 32   32  16  32 32 16  1  0 3
	$(BINDIR)/maketron --add $@ 2 32   32  16  32 32 16  1  0 3
	$(BINDIR)/maketron --add $@ 2 32   32  16  32 32 16  1  0 3
	$(BINDIR)/maketron --add $@ 2 32   32  16  32 32 16  1  0 3
	$(BINDIR)/maketron --add $@ 2 32   32  16  32 32 16  1  0 3
	$(BINDIR)/maketron --add $@ 2 32   32  16  32 32 16  1  0 3
	$(BINDIR)/maketron --add $@ 4 32   32  16  16 16 16  0 -1 0

	$(BINDIR)/maketron --add $@ 2 16   16  16  16 16 16  1  0 3
	$(BINDIR)/maketron --add $@ 2 16   16  16  16 16 16  1  0 3
	$(BINDIR)/maketron --add $@ 2 16   16  16  16 16 16  1  0 3
	$(BINDIR)/maketron --add $@ 2 16   16  16  16 16 16  1  0 3
	$(BINDIR)/maketron --add $@ 2 16   16  16  16 16 16  1  0 3
	$(BINDIR)/maketron --add $@ 2 16   16  16  16 16 16  1  0 3
	$(BINDIR)/maketron --add $@ 4 16   16  16   8  8 16  0 -1 0

	$(BINDIR)/maketron --add $@ 0  8    8  16   1 1 256  0  0 3
	$(BINDIR)/maketron --add $@ 0  1    1 256   1 1  64  0  0 3
	$(BINDIR)/maketron --add $@ 0  1    1  64   1 1   1  0  0 0

	$(BINDIR)/maketron --randomize $@ 1.5

clean:
	rm -f *.lay *.wire *.top *.map
