CXX = g++
NVCC = nvcc
NVCCFLAGS = -O6
CXXFLAGS = -O6
LDFLAGS = -lm
CULDFLAGS = -lcuda -lcudart

HDR = cudamem.hh random.hh tron.hh ppm.hh layout.hh megatron.hh wiring.hh persist.hh dataset.hh topology.hh network.hh
OBJ = cudamem.o random.o tron.o ppm.o layout.o megatron.o wiring.o persist.o dataset.o topology.o network.o

DATASETS = face-attrs.dat \
  face-8x8-lab-full.dat face-8x8-gray-full.dat \
  face-16x16-lab-full.dat face-16x16-gray-full.dat \
  face-16x16-lab-hifreq.dat face-16x16-gray-hifreq.dat \
  face-32x32-lab-full.dat face-32x32-gray-full.dat \
  face-32x32-lab-hifreq.dat face-32x32-gray-hifreq.dat

LAYOUTS = face-attrs.lay \
  face-8x8-lab-full.lay face-8x8-gray-full.lay \
  face-16x16-lab-full.lay face-16x16-gray-full.lay \
  face-16x16-lab-hifreq.lay face-16x16-gray-hifreq.lay \
  face-32x32-lab-full.lay face-32x32-gray-full.lay \
  face-32x32-lab-hifreq.lay face-32x32-gray-hifreq.lay


LABTOOLS = \
  ppmtolab labtoppm \
  labshrink labhifreq labtogray grayhifreq \
  reconlab recongray

MORETOOLS = catlay wireup maketop

.PHONY: all
all: makemore


.PHONY: datasets
datasets: $(DATASETS)

.PHONY: layouts
layouts: $(LAYOUTS)

.PHONY: labtools
labtools: $(LABTOOLS)

.PHONY: moretools
moretools: $(MORETOOLS)


$(OBJ): $(HDR)

makemore.o: $(HDR)

%.o: %.cu
	$(NVCC) -o $@ $(NVCCFLAGS) -c $<

%.o: %.cc
	$(CXX) -o $@ $(CXXFLAGS) -c $<


makemore: $(OBJ) makemore.o
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)

# $(DATASETS): $(LABTOOLS) celeba-dataset/unzipped
$(DATASETS): celeba-dataset/unzipped

face-attrs.dat:
	./mkattrsdat.pl > $@
face-attrs.lay:
	./mkattrslay.pl > $@

face-8x8-lab-full.dat:
	./mkfacedat.pl -dim=8  > $@
face-8x8-lab-full.lay:
	./mkfacelay.pl 8 3 > $@
face-8x8-gray-full.dat:
	./mkfacedat.pl -dim=8 -gray > $@
face-8x8-gray-full.lay:
	./mkfacelay.pl 8 1 > $@


face-16x16-lab-full.dat:
	./mkfacedat.pl -dim=16  > $@
face-16x16-lab-full.lay:
	./mkfacelay.pl 16 3 > $@
face-16x16-gray-full.dat:
	./mkfacedat.pl -dim=16 -gray > $@
face-16x16-gray-full.lay:
	./mkfacelay.pl 16 1 > $@
face-16x16-lab-hifreq.dat:
	./mkfacedat.pl -dim=16 -hifreq  > $@
face-16x16-lab-hifreq.lay:
	./mkfacelay.pl 8 9 > $@
face-16x16-gray-hifreq.dat:
	./mkfacedat.pl -dim=16 -gray -hifreq > $@
face-16x16-gray-hifreq.lay:
	./mkfacelay.pl 8 3 > $@

face-32x32-lab-full.dat:
	./mkfacedat.pl -dim=32  > $@
face-32x32-lab-full.lay:
	./mkfacelay.pl 32 3 > $@
face-32x32-gray-full.dat:
	./mkfacedat.pl -dim=32 -gray > $@
face-32x32-gray-full.lay:
	./mkfacelay.pl 32 1 > $@
face-32x32-lab-hifreq.dat:
	./mkfacedat.pl -dim=32 -hifreq  > $@
face-32x32-lab-hifreq.lay:
	./mkfacelay.pl 16 9 > $@
face-32x32-gray-hifreq.dat:
	./mkfacedat.pl -dim=32 -gray -hifreq > $@
face-32x32-gray-hifreq.lay:
	./mkfacelay.pl 16 3 > $@


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

catlay: catlay.o layout.o persist.o
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS)

wireup: wireup.o wiring.o layout.o persist.o
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS)
maketop: maketop.o topology.o wiring.o layout.o persist.o
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS)


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
	rm -f $(LABTOOLS)
	rm -f *.tmp

.PHONY: dataclean
dataclean: clean
	rm -rf celeba-dataset
	rm -f $(DATASETS)
	rm -f $(LAYOUTS)
