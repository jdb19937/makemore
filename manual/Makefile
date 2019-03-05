PDF = collaborative_synthesis.pdf moretran.pdf moretp.pdf peaple.pdf

.PHONY: all
all: $(PDF)

%.pdf: %.lyx
	lyx --export pdf3 $^

.PHONY: clean
clean:
	rm -f $(PDF) *~
