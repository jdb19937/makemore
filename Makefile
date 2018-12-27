CXX = g++
NVCC = nvcc
NVCCFLAGS = -O6
CXXFLAGS = -O6
LDFLAGS = -lm
CULDFLAGS = -lcuda -lcudart

HDR = cudamem.hh random.hh tron.hh ppm.hh layout.hh megatron.hh wiring.hh persist.hh dataset.hh
OBJ = cudamem.o random.o tron.o ppm.o layout.o megatron.o wiring.o persist.o dataset.o

DATASETS = faceattrs.dat \
  face8l.dat face8g.dat \
  face16l.dat face16g.dat face16lh.dat face16gh.dat \
  face32l.dat face32g.dat face32lh.dat face32gh.dat

LABTOOLS = \
  ppmtolab labtoppm \
  labshrink labhi labtogray grayhi \
  reconlab recongray

.PHONY: all
all: makemore


.PHONY: datasets
datasets: $(DATASETS)

.PHONY: labtools
labtools: $(LABTOOLS)


$(OBJ): $(HDR)

makemore.o: $(HDR)

%.o: %.cu
	$(NVCC) -o $@ $(NVCCFLAGS) -c $<

%.o: %.cc
	$(CXX) -o $@ $(CXXFLAGS) -c $<


makemore: $(OBJ) makemore.o
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)

$(DATASETS): $(LABTOOLS) celeba-dataset/unzipped

faceattrs.dat:
	./mkattrs.pl > $@

face8l.dat:
	./mkdata.pl './labshrink 128 128 4'  > $@
face8g.dat:
	./mkdata.pl './labshrink 128 128 4 |./labtogray 8 8' > $@

face16l.dat:
	./mkdata.pl './labshrink 128 128 3'  > $@
face16g.dat:
	./mkdata.pl './labshrink 128 128 3 |./labtogray 16 16' > $@
face16lh.dat:
	./mkdata.pl './labshrink 128 128 3 |./labhi 16 16'  > $@
face16gh.dat:
	./mkdata.pl './labshrink 128 128 3 |./labtogray 16 16 |./grayhi 16 16' > $@

face32l.dat:
	./mkdata.pl './labshrink 128 128 2'  > $@
face32g.dat:
	./mkdata.pl './labshrink 128 128 2 |./labtogray 32 32' > $@
face32lh.dat:
	./mkdata.pl './labshrink 128 128 2 |./labhi 32 32'  > $@
face32gh.dat:
	./mkdata.pl './labshrink 128 128 2 |./labtogray 32 32 |./grayhi 32 32' > $@


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
labhi: labhi.o
	$(CXX) -o $@ $(CXXFLAGS) $< $(LDFLAGS)
grayhi: grayhi.o
	$(CXX) -o $@ $(CXXFLAGS) $< $(LDFLAGS)
reconlab: reconlab.o
	$(CXX) -o $@ $(CXXFLAGS) $< $(LDFLAGS)
recongray: recongray.o
	$(CXX) -o $@ $(CXXFLAGS) $< $(LDFLAGS)

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
