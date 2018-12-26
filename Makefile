CXX = g++
NVCC = nvcc
NVCCFLAGS = -O6
CXXFLAGS = -O6
LDFLAGS = -lm
CULDFLAGS = -lcuda -lcudart

HDR = cudamem.hh random.hh tron.hh ppm.hh layout.hh megatron.hh wiring.hh persist.hh
OBJ = cudamem.o random.o tron.o ppm.o layout.o megatron.o wiring.o persist.o

PPMTOOLS = ppmtolab ppmtol ppmtolabtwid ppmtoltwid labtwidtoppm labtoppm ltoppm
DATAREQS = $(PPMTOOLS) unzipped

DATASETS = \
  faces8lab.dat faces8l.dat \
  faces8to16lab.dat faces8to16l.dat faces16lab.dat faces16l.dat \
  faces16to32lab.dat faces16to32l.dat faces32lab.dat faces32l.dat


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

$(DATASETS): $(DATAREQS)

faces8lab.dat:
	./mkdata.pl -dim=8 -twid=0 -gray=0 > $@.tmp
	mv -f $@.tmp $@
faces8l.dat:
	./mkdata.pl -dim=8 -twid=0 -gray=1 > $@.tmp
	mv -f $@.tmp $@


faces8to16lab.dat:
	./mkdata.pl -dim=16 -twid=1 -gray=0 > $@.tmp
	mv -f $@.tmp $@
faces8to16l.dat:
	./mkdata.pl -dim=16 -twid=1 -gray=1 > $@.tmp
	mv -f $@.tmp $@
faces16lab.dat:
	./mkdata.pl -dim=16 -twid=0 -gray=0 > $@.tmp
	mv -f $@.tmp $@
faces16l.dat:
	./mkdata.pl -dim=16 -twid=0 -gray=1 > $@.tmp
	mv -f $@.tmp $@


faces16to32lab.dat:
	./mkdata.pl -dim=32 -twid=1 -gray=0 > $@.tmp
	mv -f $@.tmp $@
faces16to32l.dat:
	./mkdata.pl -dim=32 -twid=1 -gray=1 > $@.tmp
	mv -f $@.tmp $@
faces32lab.dat:
	./mkdata.pl -dim=32 -twid=0 -gray=0 > $@.tmp
	mv -f $@.tmp $@
faces32l.dat:
	./mkdata.pl -dim=32 -twid=0 -gray=1 > $@.tmp
	mv -f $@.tmp $@


ppmtolab: ppm.cc ppm.hh
	$(CXX) -o $@ $(CXXFLAGS) -DPPMTOLAB_MAIN $< $(LDFLAGS)
ppmtol: ppm.cc ppm.hh
	$(CXX) -o $@ $(CXXFLAGS) -DPPMTOL_MAIN $< $(LDFLAGS)
ppmtolabtwid: ppm.cc ppm.hh
	$(CXX) -o $@ $(CXXFLAGS) -DPPMTOLABTWID_MAIN $< $(LDFLAGS)
ppmtoltwid: ppm.cc ppm.hh
	$(CXX) -o $@ $(CXXFLAGS) -DPPMTOLTWID_MAIN $< $(LDFLAGS)
labtwidtoppm: ppm.cc ppm.hh
	$(CXX) -o $@ $(CXXFLAGS) -DLABTWIDTOPPM_MAIN $< $(LDFLAGS)
labtoppm: ppm.cc ppm.hh
	$(CXX) -o $@ $(CXXFLAGS) -DLABTOPPM_MAIN $< $(LDFLAGS)
ltoppm: ppm.cc ppm.hh
	$(CXX) -o $@ $(CXXFLAGS) -DLTOPPM_MAIN $< $(LDFLAGS)

unzipped: celeba-dataset.zip
	rm -rf celeba-dataset
	unzip celeba-dataset.zip -d celeba-dataset
	unzip celeba-dataset/img_align_celeba.zip -d celeba-dataset
	touch unzipped

.PHONY: celeba-dataset.zip
celeba-dataset.zip:
	test -s $@ || { echo 'get celeba-dataset.zip from https://www.kaggle.com/jessicali9530/celeba-dataset/home'; false; }

.PHONY: clean
clean:
	rm -f $(OBJ)
	rm -f makemore makemore.o
	rm -f $(PPMTOOLS)
	rm -f *.tmp

.PHONY: dataclean
dataclean: clean
	rm -rf celeba-dataset
	rm -f unzipped
	rm -f $(DATASETS)
