CXX = g++
NVCC = nvcc
NVCCFLAGS = -O3 -Xcompiler -fPIC
CXXFLAGS = -O3 -g -fPIC
LDFLAGS = -L. -Wl,-rpath,'$$ORIGIN' -lmakemore -lm -lpng -lssl -lcrypto -lcudart -ldl
CULDFLAGS = -lcuda -lcudart

LIBHDR = cudamem.hh random.hh tron.hh ppm.hh layout.hh megatron.hh wiring.hh persist.hh dataset.hh topology.hh multitron.hh project.hh twiddle.hh sampler.hh closest.hh pipeline.hh parson.hh ipdb.hh warp.hh vocab.hh script.hh strutils.hh hashbag.hh shibboleth.hh brane.hh rule.hh wildleth.hh stage.hh zone.hh bus.hh numutils.hh server.hh urb.hh org.hh urbite.hh convo.hh shibbomore.hh wildmore.hh agent.hh markov.hh imgutils.hh judge.hh process.hh command.hh system.hh io.hh session.hh improver.hh lineref.hh word.hh tmutils.hh cholo.hh encgendis.hh mapfile.hh normatron.hh camera.hh partrait.hh triangle.hh point.hh pose.hh autoposer.hh catalog.hh
LIBOBJ = cudamem.o random.o tron.o ppm.o layout.o megatron.o wiring.o persist.o dataset.o topology.o multitron.o project.o twiddle.o sampler.o closest.o pipeline.o parson.o ipdb.o warp.o vocab.o script.o strutils.o hashbag.o shibboleth.o brane.o rule.o wildleth.o stage.o zone.o bus.o numutils.o server.o urb.o org.o urbite.o convo.o shibbomore.o wildmore.o agent.o markov.o imgutils.o judge.o process.o command.o system.o io.o session.o improver.o lineref.o word.o cholo.o encgendis.o mapfile.o normatron.o camera.o partrait.o triangle.o point.o pose.o autoposer.o catalog.o
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

MORETOOLS = makemore makelay catlay wireup maketop makemap chatfab learnfab learnmore genicon servemore makezone sampler makeipdat errstats fillparsonsdat autolign dumpbranetxt moretran learnmark makemark burnbig burnimp learnimp showbig showimp impcholo cholo64 showcholo64  burnpre learnpre restock stest makesamp inburn incholo makeknobs inconc makeconc burnseg tmptag eyewarp aleyen makesamp makecholo 

MANUAL = moretran.pdf moretp.pdf

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

.PHONY: manual
manual: $(MANUAL)


$(LIBOBJ): $(LIBHDR)

%.pdf: %.lyx
	lyx --export pdf3 $^

%.o: %.cu
	$(NVCC) -o $@ $(NVCCFLAGS) -c $<

%.o: %.cc
	$(CXX) -o $@ $(CXXFLAGS) -c $<

libmakemore.a: $(LIBOBJ)
	ar cr $@ $^

libmakemore.so: $(LIBOBJ)
	$(CXX) -o $@ -shared $^

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
chatfab.o: $(LIBHDR)
chatfab: chatfab.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
talkfab.o: $(LIBHDR)
talkfab: talkfab.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
learnfab.o: $(LIBHDR)
learnfab: learnfab.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
makemark.o: $(LIBHDR)
makemark: makemark.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
learnmark.o: $(LIBHDR)
learnmark: learnmark.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
cholo64.o: $(LIBHDR)
cholo64: cholo64.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
makecholo.o: $(LIBHDR)
makecholo: makecholo.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
impcholo.o: $(LIBHDR)
impcholo: impcholo.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
incholo.o: $(LIBHDR)
incholo: incholo.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
clones.o: $(LIBHDR)
clones: clones.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
showbig.o: $(LIBHDR)
showbig: showbig.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
showimp.o: $(LIBHDR)
showimp: showimp.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
showcholo64.o: $(LIBHDR)
showcholo64: showcholo64.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
learnpre.o: $(LIBHDR)
learnpre: learnpre.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
burnpre.o: $(LIBHDR)
burnpre: burnpre.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
restock.o: $(LIBHDR)
restock: restock.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
stest.o: $(LIBHDR)
stest: stest.o libmakemore.so
	$(CXX) -o $@ $(CXXFLAGS) stest.o $(LDFLAGS) $(CULDFLAGS)
makesamp.o: $(LIBHDR)
makesamp: makesamp.o libmakemore.so
	$(CXX) -o $@ $(CXXFLAGS) makesamp.o $(LDFLAGS) $(CULDFLAGS)
makeconc.o: $(LIBHDR)
makeconc: makeconc.o libmakemore.so
	$(CXX) -o $@ $(CXXFLAGS) makeconc.o $(LDFLAGS) $(CULDFLAGS)
makeknobs.o: $(LIBHDR)
makeknobs: makeknobs.o libmakemore.so
	$(CXX) -o $@ $(CXXFLAGS) makeknobs.o $(LDFLAGS) $(CULDFLAGS)
testchol.o: $(LIBHDR)
testchol: testchol.o libmakemore.so
	$(CXX) -o $@ $(CXXFLAGS) testchol.o $(LDFLAGS) $(CULDFLAGS)
burnreseg.o: $(LIBHDR)
burnreseg: burnreseg.o libmakemore.so
	$(CXX) -o $@ $(CXXFLAGS) burnreseg.o $(LDFLAGS) $(CULDFLAGS)
burnseg.o: $(LIBHDR)
burnseg: burnseg.o libmakemore.so
	$(CXX) -o $@ $(CXXFLAGS) burnseg.o $(LDFLAGS) $(CULDFLAGS)
tmptag.o: $(LIBHDR)
tmptag: tmptag.o libmakemore.so
	$(CXX) -o $@ $(CXXFLAGS) tmptag.o $(LDFLAGS) $(CULDFLAGS)
aleyen.o: $(LIBHDR)
aleyen: aleyen.o libmakemore.so
	$(CXX) -o $@ $(CXXFLAGS) aleyen.o $(LDFLAGS) $(CULDFLAGS)
eyewarp.o: $(LIBHDR)
eyewarp: eyewarp.o libmakemore.so
	$(CXX) -o $@ $(CXXFLAGS) eyewarp.o $(LDFLAGS) $(CULDFLAGS)
burnbig.o: $(LIBHDR)
burnbig: burnbig.o libmakemore.so
	$(CXX) -o $@ $(CXXFLAGS) burnbig.o $(LDFLAGS) $(CULDFLAGS)
addposetags.o: $(LIBHDR)
addposetags: addposetags.o libmakemore.so
	$(CXX) -o $@ $(CXXFLAGS) addposetags.o $(LDFLAGS) $(CULDFLAGS)
showtag.o: $(LIBHDR)
showtag: showtag.o libmakemore.so
	$(CXX) -o $@ $(CXXFLAGS) showtag.o $(LDFLAGS) $(CULDFLAGS)
posestats.o: $(LIBHDR)
posestats: posestats.o libmakemore.so
	$(CXX) -o $@ $(CXXFLAGS) posestats.o $(LDFLAGS) $(CULDFLAGS)
mf.o: $(LIBHDR)
mf: mf.o libmakemore.so
	$(CXX) -o $@ $(CXXFLAGS) mf.o $(LDFLAGS) $(CULDFLAGS)
burnimp.o: $(LIBHDR)
burnimp: burnimp.o libmakemore.so
	$(CXX) -o $@ $(CXXFLAGS) burnimp.o $(LDFLAGS) $(CULDFLAGS)
sdltest.o: $(LIBHDR)
sdltest: sdltest.o libmakemore.so
	$(CXX) -o $@ $(CXXFLAGS) sdltest.o $(LDFLAGS) $(CULDFLAGS) -lSDL2
inconc.o: $(LIBHDR)
inconc: inconc.o libmakemore.so
	$(CXX) -o $@ $(CXXFLAGS) inconc.o $(LDFLAGS) $(CULDFLAGS)
inburn.o: $(LIBHDR)
inburn: inburn.o libmakemore.so
	$(CXX) -o $@ $(CXXFLAGS) inburn.o $(LDFLAGS) $(CULDFLAGS)
showimpdis.o: $(LIBHDR)
showimpdis: showimpdis.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
learnimpdis.o: $(LIBHDR)
learnimpdis: learnimpdis.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
learnimp.o: $(LIBHDR)
learnimp: learnimp.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
dumpbranetxt.o: $(LIBHDR)
dumpbranetxt: dumpbranetxt.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
moretran.o: $(LIBHDR)
moretran: moretran.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
learnmore.o: $(LIBHDR)
learnmore: learnmore.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
scriptmore.o: $(LIBHDR)
scriptmore: scriptmore.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
makeipdat.o: $(LIBHDR)
makeipdat: makeipdat.o $(LIB)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)
makezone.o: $(LIBHDR)
makezone: makezone.o $(LIB)
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


moregasm.tsv: dumpbranetxt
	./dumpbranetxt |sort > $@

mork.png: mork.pl
	./mork.pl | pnmtopng > $@
