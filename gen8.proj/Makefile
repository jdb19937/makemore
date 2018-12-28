#!/bin/bash

DATASETS = context.dat samples.dat
LAYOUTS = \
  context.lay samples.lay controls.lay \
  encinput.lay enchidden1.lay enchidden2.lay encoutput.lay \
  geninput.lay genhidden1.lay genhidden2.lay genoutput.lay \
  disinput.lay dishidden1.lay dishidden2.lay disoutput.lay

WIRINGS = \
  enchidden1.wire enchidden2.wire encoutput.wire \
  genhidden1.wire genhidden2.wire genoutput.wire \
  dishidden1.wire dishidden2.wire disoutput.wire

TOPOLOGIES = \
  enc.top gen.top dis.top

all: $(DATASETS) $(LAYOUTS) $(WIRINGS) $(TOPOLOGIES)

context.dat:
	ln -sf ../face-attrs.dat $@
context.lay:
	ln -sf ../face-attrs.lay $@
samples.dat:
	ln -sf ../face-8x8-lab-full.dat $@
samples.lay:
	ln -sf ../face-8x8-lab-full.lay $@

global-controls.lay:
	../makelay -center 16 > $@
local-controls.lay:
	../makelay -grid 4 > $@
controls.lay: global-controls.lay local-controls.lay
	../catlay $^ > $@


encinput.lay: samples.lay context.lay
	../catlay samples.lay context.lay > $@
enchidden1.lay:
	../makelay -rand 256 16 > $@
enchidden1.wire: encinput.lay enchidden1.lay
	../wireup $^ > $@
enchidden2.lay:
	../makelay -rand 256 16 > $@
enchidden2.wire: enchidden1.lay enchidden2.lay
	../wireup $^ > $@
encoutput.lay: controls.lay
	cp -f controls.lay $@
encoutput.wire: enchidden2.lay encoutput.lay
	../wireup $^ > $@
enc.top: enchidden1.wire enchidden2.wire encoutput.wire
	../maketop $^ > $@


geninput.lay: controls.lay context.lay
	../catlay controls.lay context.lay > $@
genhidden1.lay:
	../makelay -rand 256 16 > $@
genhidden1.wire: geninput.lay genhidden1.lay
	../wireup $^ > $@
genhidden2.lay:
	../makelay -rand 256 16 > $@
genhidden2.wire: genhidden1.lay genhidden2.lay
	../wireup $^ > $@
genoutput.lay: samples.lay
	cp -f samples.lay $@
genoutput.wire: genhidden2.lay genoutput.lay
	../wireup $^ > $@
gen.top: genhidden1.wire genhidden2.wire genoutput.wire
	../maketop $^ > $@


disinput.lay: samples.lay context.lay
	../catlay controls.lay context.lay > $@
dishidden1.lay:
	../makelay -rand 256 16 > $@
dishidden1.wire: disinput.lay dishidden1.lay
	../wireup $^ > $@
dishidden2.lay:
	../makelay -rand 64 16 > $@
dishidden2.wire: dishidden1.lay dishidden2.lay
	../wireup $^ > $@
disoutput.lay:
	../makelay -center 1 > $@
disoutput.wire: dishidden2.lay disoutput.lay
	../wireup $^ > $@
dis.top: dishidden1.wire dishidden2.wire disoutput.wire
	../maketop $^ > $@

clean:
	rm -f *.lay *.dat *.wire *.top
