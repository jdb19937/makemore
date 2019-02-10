CXX = g++
NVCC = nvcc
NVCCFLAGS = -O3
CXXFLAGS = -O3
LDFLAGS = -lm
CULDFLAGS = -lcuda -lcudart

LIBHDR = cudamem.hh random.hh tron.hh ppm.hh layout.hh megatron.hh wiring.hh persist.hh dataset.hh topology.hh multitron.hh project.hh twiddle.hh sampler.hh closest.hh pipeline.hh parson.hh ipdb.hh warp.hh tagbag.hh vocab.hh confab.hh
LIBOBJ = cudamem.o random.o tron.o ppm.o layout.o megatron.o wiring.o persist.o dataset.o topology.o multitron.o project.o twiddle.o sampler.o closest.o pipeline.o parson.o ipdb.o warp.o tagbag.o vocab.o confab.o
LIB = libmakemore.a

DATASETS = face-attrs.dat face8.dat face16.dat face32.dat face64.dat face128.dat

LAYOUTS = face-attrs.lay \
  face-8x8-lab-full.lay face-8x8-gray-full.lay \
  face-16x16-lab-full.lay face-16x16-gray-full.lay \
  face-16x16-lab-hifreq.lay face-16x16-gray-hifreq.lay \
  face-32x32-lab-full.lay face-32x32-gray-full.lay \
  face-32x32-lab-hifreq.lay face-32x32-gray-hifreq.lay \
  face-64x64-lab-full.lay face-64x64-gray-full.lay \
  face-64x64-lab-hifreq.lay face-64x64-gray-hifreq.lay \
  face-128x128-lab-full.lay face-128x128-gray-full.lay \
  face-128x128-lab-hifreq.lay face-128x128-gray-hifreq.lay


LABTOOLS = \
  ppmtolab labtoppm \
  labshrink labhifreq labtogray grayhifreq \
  reconlab recongray 

MORETOOLS = makemore makelay catlay wireup maketop makemap learnmore genicon servemore makeparsonsdat sampler makeipdat errstats fillparsonsdat autolign

.PHONY: all
all: $(LIB) $(MORETOOLS)


.PHONY: datasets
datasets: $(DATASETS)

.PHONY: layouts
layouts: $(LAYOUTS)

.PHONY: labtools
labtools: $(LABTOOLS)

.PHONY: moretools
moretools: $(MORETOOLS)


$(LIBOBJ): $(LIBHDR)

%.o: %.cu
	$(NVCC) -o $@ $(NVCCFLAGS) -c $<

%.o: %.cc
	$(CXX) -o $@ $(CXXFLAGS) -c $<

libmakemore.a: $(LIBOBJ)
	ar cr $@ $^


# $(DATASETS): $(LABTOOLS) celeba-dataset/unzipped
$(DATASETS): celeba-dataset/unzipped

face-attrs.dat:
	./mkfacedat.pl -attrs > $@
face-attrs.lay:
	./mkattrslay.pl > $@

newface8.dat:
	./mkfacedat.pl -dim=8  > $@
face8.dat:
	./mkfacedat.pl -dim=8  > $@
face-8x8-lab-full.lay:
	./mkfacelay.pl 8 3 > $@
face-8x8-gray-full.dat:
	./mkfacedat.pl -dim=8 -gray > $@
face-8x8-gray-full.lay:
	./mkfacelay.pl 8 1 > $@


newface16.dat:
	./mkfacedat.pl -dim=16  > $@
face16.dat:
	./mkfacedat.pl -dim=16  > $@
face-16x16-lab-full.lay:
	./mkfacelay.pl 16 3 > $@
face-16x16-gray-full.lay:
	./mkfacelay.pl 16 1 > $@
face-16x16-lab-hifreq.lay:
	./mkfacelay.pl 8 9 > $@
face-16x16-gray-hifreq.lay:
	./mkfacelay.pl 8 3 > $@

newface32.dat:
	./mkfacedat.pl -dim=32  > $@
face32.dat:
	./mkfacedat.pl -dim=32  > $@
face-32x32-lab-full.lay:
	./mkfacelay.pl 32 3 > $@
face-32x32-gray-full.lay:
	./mkfacelay.pl 32 1 > $@
face-32x32-lab-hifreq.lay:
	./mkfacelay.pl 16 9 > $@
face-32x32-gray-hifreq.lay:
	./mkfacelay.pl 16 3 > $@

newface64.dat:
	./mkfacedat.pl -dim=64 > $@
face64.dat:
	./mkfacedat.pl -dim=64 > $@
face-64x64-lab-full.lay:
	./mkfacelay.pl 64 3 > $@
face-64x64-gray-full.lay:
	./mkfacelay.pl 64 1 > $@
face-64x64-lab-hifreq.lay:
	./mkfacelay.pl 32 9 > $@
face-64x64-gray-hifreq.lay:
	./mkfacelay.pl 32 3 > $@

face128.dat:
	./mkfacedat.pl -dim=128 > $@
face-128x128-lab-full.lay:
	./mkfacelay.pl 128 3 > $@
face-128x128-gray-full.lay:
	./mkfacelay.pl 128 1 > $@
face-128x128-lab-hifreq.lay:
	./mkfacelay.pl 64 9 > $@
face-128x128-gray-hifreq.lay:
	./mkfacelay.pl 64 3 > $@

ppmtolab.o: ppm.hh
ppmtolab: ppmtolab.o ppm.o
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS)

labtoppm.o: ppm.hh
labtoppm: labtoppm.o ppm.o
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS)

graytoppm.o: ppm.hh
graytoppm: graytoppm.o ppm.o
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS)

labshrink: labshrink.o
	$(CXX) -o $@ $(CXXFLAGS) $< $(LDFLAGS)
labtogray: labtogray.o
	$(CXX) -o $@ $(CXXFLAGS) $< $(LDFLAGS)
graytolab: graytolab.o
	$(CXX) -o $@ $(CXXFLAGS) $< $(LDFLAGS)
labhifreq: labhifreq.o
	$(CXX) -o $@ $(CXXFLAGS) $< $(LDFLAGS)
grayhifreq: grayhifreq.o
	$(CXX) -o $@ $(CXXFLAGS) $< $(LDFLAGS)
reconlab: reconlab.o
	$(CXX) -o $@ $(CXXFLAGS) $< $(LDFLAGS)
recongray: recongray.o
	$(CXX) -o $@ $(CXXFLAGS) $< $(LDFLAGS)

catlay.o: $(LIBHDR)
catlay: catlay.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS)
makelay.o: $(LIBHDR)
makelay: makelay.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS)

genicon.o: $(LIBHDR)
genicon: genicon.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS)
spread.ppm: genicon
	./genicon spread > $@
spread.png: spread.ppm
	convert $^ $@
fade.ppm: genicon
	./genicon fade > $@
fade.png: fade.ppm
	convert $^ $@
darken.ppm: genicon
	./genicon darken > $@
darken.png: darken.ppm
	convert $^ $@
brighten.ppm: genicon
	./genicon brighten > $@
brighten.png: brighten.ppm
	convert $^ $@
sharp.ppm: genicon
	./genicon sharp > $@
sharp.png: sharp.ppm
	convert $^ $@
blur.ppm: genicon
	./genicon blur > $@
blur.png: blur.ppm
	convert $^ $@
fuzz.ppm: genicon
	./genicon fuzz > $@
fuzz.png: fuzz.ppm
	convert $^ $@
spectrum.ppm: genicon
	./genicon spectrum > $@
spectrum.png: spectrum.ppm
	convert $^ $@
random.ppm: genicon
	./genicon random > $@
random.png: random.ppm
	convert $^ $@
static.ppm: genicon
	./genicon static > $@
static.png: static.ppm
	convert $^ $@
burn.ppm: genicon
	./genicon burn > $@
burn.png: burn.ppm
	convert $^ $@

wireup.o: $(LIBHDR)
wireup: wireup.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS)
maketop.o: $(LIBHDR)
maketop: maketop.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS)
makemap.o: $(LIBHDR)
makemap: makemap.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)

parpulate.o: $(LIBHDR)
parpulate: parpulate.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
makemore.o: $(LIBHDR)
makemore: makemore.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
servemore.o: $(LIBHDR)
servemore: servemore.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
autolign.o: $(LIBHDR)
autolign: autolign.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
errstats.o: $(LIBHDR)
errstats: errstats.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
learnmore.o: $(LIBHDR)
learnmore: learnmore.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
makeipdat.o: $(LIBHDR)
makeipdat: makeipdat.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
makeparsonsdat.o: $(LIBHDR)
makeparsonsdat: makeparsonsdat.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
fillparsonsdat.o: $(LIBHDR)
fillparsonsdat: fillparsonsdat.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
sampler-main.o: sampler.cc
	$(CXX) -c -o $@ -DSAMPLER_MAIN $(CXXFLAGS) $^
sampler: sampler-main.o random.o
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
traindis.o: $(LIBHDR)
traindis: traindis.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
trainfaith.o: $(LIBHDR)
trainfaith: trainfaith.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
seefaith.o: $(LIBHDR)
seefaith: seefaith.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
gencenter.o: $(LIBHDR)
gencenter: gencenter.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
pipecenter.o: $(LIBHDR)
pipecenter: pipecenter.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
seecenter.o: $(LIBHDR)
seecenter: seecenter.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
seerand.o: $(LIBHDR)
seerand: seerand.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
seezoom.o: $(LIBHDR)
seezoom: seezoom.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)


celeba-dataset/unzipped: celeba-dataset.zip
	rm -rf celeba-dataset
	unzip celeba-dataset.zip -d celeba-dataset
	unzip celeba-dataset/img_align_celeba.zip -d celeba-dataset
	touch celeba-dataset/unzipped

#
# .PHONY: celeba-dataset.zip
# celeba-dataset.zip:
# 	test -s $@ || { echo 'get celeba-dataset.zip from https://www.kaggle.com/jessicali9530/celeba-dataset/home'; false; }
# 

.PHONY: clean
clean:
	rm -f *.o
	rm -f makemore
	rm -f libmakemore.a
	rm -f $(LABTOOLS)
	rm -f $(MORETOOLS)
	rm -f *.tmp

.PHONY: dataclean
dataclean: clean
	rm -rf celeba-dataset
	rm -f $(DATASETS)
	rm -f $(LAYOUTS)
