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

MAPFILES = \
  enc.map gen.map dis.map

all: $(DATASETS) $(LAYOUTS) $(WIRINGS) $(TOPOLOGIES) $(MAPFILES)

enchidden1.lay:
	../makelay -rand 256 16 > $@
enchidden2.lay:
	../makelay -rand 256 32 > $@
genhidden1.lay:
	../makelay -rand 256 32 > $@
genhidden2.lay:
	../makelay -rand 256 16 > $@
dishidden1.lay:
	../makelay -rand 128 16 > $@
dishidden2.lay:
	../makelay -rand 64 16 > $@
global-controls.lay:
	../makelay -center 16 > $@
local-controls.lay:
	../makelay -grid 8 > $@


context.dat:
	ln -sf ../face-attrs.dat $@
context.lay:
	ln -sf ../face-attrs.lay $@
samples.dat:
	ln -sf ../face-8x8-lab-full.dat $@
samples.lay:
	ln -sf ../face-8x8-lab-full.lay $@

controls.lay: global-controls.lay local-controls.lay
	../catlay $^ > $@


encinput.lay: context.lay samples.lay
	../catlay context.lay samples.lay > $@
enchidden1.wire: encinput.lay enchidden1.lay
	../wireup $^ > $@
enchidden2.wire: enchidden1.lay enchidden2.lay
	../wireup $^ > $@
encoutput.lay: controls.lay
	cp -f controls.lay $@
encoutput.wire: enchidden2.lay encoutput.lay
	../wireup $^ > $@
enc.top: enchidden1.wire enchidden2.wire encoutput.wire
	../maketop $^ > $@
enc.map: enc.top
	../makemap enc.top enc.map


geninput.lay: context.lay controls.lay
	../catlay context.lay controls.lay > $@
genhidden1.wire: geninput.lay genhidden1.lay
	../wireup $^ > $@
genhidden2.wire: genhidden1.lay genhidden2.lay
	../wireup $^ > $@
genoutput.lay: samples.lay
	cp -f samples.lay $@
genoutput.wire: genhidden2.lay genoutput.lay
	../wireup $^ > $@
gen.top: genhidden1.wire genhidden2.wire genoutput.wire
	../maketop $^ > $@
gen.map: gen.top
	../makemap gen.top gen.map


disinput.lay: context.lay samples.lay
	../catlay context.lay samples.lay > $@
dishidden1.wire: disinput.lay dishidden1.lay
	../wireup $^ > $@
dishidden2.wire: dishidden1.lay dishidden2.lay
	../wireup $^ > $@
disoutput.lay:
	../makelay -center 1 > $@
disoutput.wire: dishidden2.lay disoutput.lay
	../wireup $^ > $@
dis.top: dishidden1.wire dishidden2.wire disoutput.wire
	../maketop $^ > $@
dis.map: dis.top
	../makemap dis.top dis.map

clean:
	rm -f *.lay *.dat *.wire *.top *.map
