CXX = g++
NVCC = nvcc
NVCCFLAGS = -O6
CXXFLAGS = -O6
LDFLAGS = -lm
CULDFLAGS = -lcuda -lcudart

HDR = cudamem.hh random.hh tron.hh ppm.hh layout.hh megatron.hh wiring.hh persist.hh
OBJ = cudamem.o random.o tron.o ppm.o layout.o megatron.o wiring.o persist.o

DATASETS = faceattrs.dat \
  face8lab.dat face8l.dat \
  face16lab.dat face16l.dat face16labhi.dat face16lhi.dat \
  face32lab.dat face32l.dat face32labhi.dat face32lhi.dat

LABTOOLS = labhi labl lablhi labshrink ppmtolab

.PHONY: all
all: makemore


.PHONY: datasets
datasets: $(DATASETS)


$(OBJ): $(HDR)

makemore.o: $(HDR)

%.o: %.cu
	$(NVCC) -o $@ $(NVCCFLAGS) -c $<

%.o: %.cc
	$(CXX) -o $@ $(CXXFLAGS) -c $<


makemore: $(OBJ) makemore.o
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(CULDFLAGS)

$(DATASETS): $(LABTOOLS) celeba-dataset

faceattrs.dat:
	./mkattrs.pl > $@

face8lab.dat:
	./mkdata.pl './labshrink 128 128 4'  > $@
face8l.dat:
	./mkdata.pl './labshrink 128 128 4 |./labl 8 8' > $@

face16lab.dat:
	./mkdata.pl './labshrink 128 128 3'  > $@
face16l.dat:
	./mkdata.pl './labshrink 128 128 3 |./labl 16 16' > $@
face16labhi.dat:
	./mkdata.pl './labshrink 128 128 3 |./labhi 16 16'  > $@
face16lhi.dat:
	./mkdata.pl './labshrink 128 128 3 |./lablhi 16 16' > $@

face32lab.dat:
	./mkdata.pl './labshrink 128 128 2'  > $@
face32l.dat:
	./mkdata.pl './labshrink 128 128 2 |./labl 32 32' > $@
face32labhi.dat:
	./mkdata.pl './labshrink 128 128 2 |./labhi 32 32'  > $@
face32lhi.dat:
	./mkdata.pl './labshrink 128 128 2 |./lablhi 32 32' > $@



ppmtolab: ppm.cc ppm.hh
	$(CXX) -o $@ $(CXXFLAGS) -DPPMTOLAB_MAIN $< $(LDFLAGS)
ppmtol: ppm.cc ppm.hh
	$(CXX) -o $@ $(CXXFLAGS) -DPPMTOL_MAIN $< $(LDFLAGS)
ppmtolabtwid: ppm.cc ppm.hh
	$(CXX) -o $@ $(CXXFLAGS) -DPPMTOLABTWID_MAIN $< $(LDFLAGS)
ppmtolabhi: ppm.cc ppm.hh
	$(CXX) -o $@ $(CXXFLAGS) -DPPMTOLABTWID_MAIN $< $(LDFLAGS)
ppmtoltwid: ppm.cc ppm.hh
	$(CXX) -o $@ $(CXXFLAGS) -DPPMTOLTWID_MAIN $< $(LDFLAGS)
ppmtolhi: ppm.cc ppm.hh
	$(CXX) -o $@ $(CXXFLAGS) -DPPMTOLTWID_MAIN $< $(LDFLAGS)
labtwidtoppm: ppm.cc ppm.hh
	$(CXX) -o $@ $(CXXFLAGS) -DLABTWIDTOPPM_MAIN $< $(LDFLAGS)
labtoppm: ppm.cc ppm.hh
	$(CXX) -o $@ $(CXXFLAGS) -DLABTOPPM_MAIN $< $(LDFLAGS)
ltoppm: ppm.cc ppm.hh
	$(CXX) -o $@ $(CXXFLAGS) -DLTOPPM_MAIN $< $(LDFLAGS)

labshrink: labshrink.o
	$(CXX) -o $@ $(CXXFLAGS) $< $(LDFLAGS)
labhi: labhi.o
	$(CXX) -o $@ $(CXXFLAGS) $< $(LDFLAGS)
lablhi: lablhi.o
	$(CXX) -o $@ $(CXXFLAGS) $< $(LDFLAGS)
labl: labl.o
	$(CXX) -o $@ $(CXXFLAGS) $< $(LDFLAGS)

unzipped: celeba-dataset.zip
	rm -rf celeba-dataset
	unzip celeba-dataset.zip -d celeba-dataset
	unzip celeba-dataset/img_align_celeba.zip -d celeba-dataset
	touch unzipped

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
	rm -f unzipped
	rm -f $(DATASETS)
