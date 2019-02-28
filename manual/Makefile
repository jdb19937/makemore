PDF = collaborative_synthesis.pdf moregasm.pdf moretp.pdf

.PHONY: all
all: $(PDF)

%.pdf: %.lyx
	lyx --export pdf3 $^

.PHONY: clean
clean:
	rm -f $(PDF)
