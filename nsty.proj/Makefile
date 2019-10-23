DATFILE = /spin/dan/cand.dat
BINDIR = ..

CHOLOS = base.cholo


.PHONY: all
all: $(CHOLOS)

.PHONY: clean
clean:
	rm -f $(CHOLOS)

base.cholo:
	$(BINDIR)/makecholo $(DATFILE) > $@
cat.cholo:
	$(BINDIR)/makecholo $(DATFILE) cat > $@
dog.cholo:
	$(BINDIR)/makecholo $(DATFILE) dog > $@
%.cholo:
	$(BINDIR)/makecholo $(DATFILE) `echo $@ |perl -pe 's/.cholo//'` > $@
