#!/bin/bash

MAPFILES = gen.map dis.map

BINDIR = ..

all: $(MAPFILES)

gen.map:
	$(BINDIR)/maketron --new $@

	$(BINDIR)/maketron --add $@ 6 32 32  6   32 32 24   0 0 0
	$(BINDIR)/maketron --add $@ 7 32 32 24   32 32 32   0 0 0

	$(BINDIR)/maketron --add $@ 2 32 32 32   32 32 32   1 0 2
	$(BINDIR)/maketron --add $@ 2 32 32 32   32 32 32   1 0 2
	$(BINDIR)/maketron --add $@ 2 32 32 32   32 32 32   1 0 2
	$(BINDIR)/maketron --add $@ 2 32 32 32   32 32 32   1 0 2
	$(BINDIR)/maketron --add $@ 2 32 32 32   32 32 32   1 0 2
	$(BINDIR)/maketron --add $@ 2 32 32 32   32 32 32   1 0 2
	$(BINDIR)/maketron --add $@ 2 32 32 32   32 32 32   1 0 2
	$(BINDIR)/maketron --add $@ 2 32 32 32   32 32 32   1 0 2
	$(BINDIR)/maketron --add $@ 2 32 32 32   32 32 32   1 0 2
	$(BINDIR)/maketron --add $@ 2 32 32 32   32 32 32   1 0 2
	$(BINDIR)/maketron --add $@ 2 32 32 32   32 32 32   1 0 2
	$(BINDIR)/maketron --add $@ 2 32 32 32   32 32 32   1 0 2
	$(BINDIR)/maketron --add $@ 2 32 32 32   32 32 32   1 0 2
	$(BINDIR)/maketron --add $@ 2 32 32 32   32 32 32   1 0 2
	$(BINDIR)/maketron --add $@ 2 32 32 32   32 32 32   1 0 2
	$(BINDIR)/maketron --add $@ 2 32 32 32   32 32 32   1 0 2

	$(BINDIR)/maketron --add $@ 2 32 32 32   32 32  3   1 0 0

	$(BINDIR)/maketron --randomize $@ 1.0


dis.map:
	$(BINDIR)/maketron --new $@

#	$(BINDIR)/maketron --add $@ 5 32 32  9   32 32  9   0 0 0

	$(BINDIR)/maketron --add $@ 2 32 32  9   32 32 32   0 0 0

	$(BINDIR)/maketron --add $@ 2 32 32 32   32 32 32   1 0 2
	$(BINDIR)/maketron --add $@ 2 32 32 32   32 32 32   1 0 2
	$(BINDIR)/maketron --add $@ 2 32 32 32   32 32 32   1 0 2
	$(BINDIR)/maketron --add $@ 2 32 32 32   32 32 32   1 0 2
	$(BINDIR)/maketron --add $@ 2 32 32 32   32 32 32   1 0 2
	$(BINDIR)/maketron --add $@ 2 32 32 32   32 32 32   1 0 2

	$(BINDIR)/maketron --add $@ 2 32 32 32   32 32  1   0 0 0

	$(BINDIR)/maketron --randomize $@ 1.0

clean:
	rm -f $(MAPFILES) out
