#!/bin/bash

LAYOUTS = \
  context.lay control.lay target.lay output.lay \
  encinput.lay enchidden1.lay enchidden2.lay encoutput.lay \
  geninput.lay genhidden1.lay genhidden2.lay genoutput.lay \

TOPOLOGIES = \
  enc.top gen.top dis.top

MAPFILES = \
  enc.map gen.map dis.map

all: $(DATASETS) $(LAYOUTS) $(TOPOLOGIES) $(MAPFILES)


dishidden1.lay:
	../makelay -rand 1024 64 > $@
dishidden2.lay:
	../makelay -rand 1024 64 > $@
enchidden1.lay:
	../makelay -rand 1024 64 > $@
enchidden2.lay:
	../makelay -rand 1024 64 > $@
genhidden1.lay:
	../makelay -rand 1024 64 > $@
genhidden2.lay:
	../makelay -rand 1024 64 > $@
control.lay:
	../makelay -rand 128 64 > $@


context.lay:
	cp -f ../face-attrs.lay $@
target.lay:
	cp -f ../face-8x8-lab-full.lay $@
output.lay:
	cp -f ../face-8x8-lab-full.lay $@

disinput.lay: context.lay target.lay
	../catlay $^ > $@
dishidden1.wire: disinput.lay dishidden1.lay
	../wireup $^ > $@
dishidden2.wire: dishidden1.lay dishidden2.lay
	../wireup $^ > $@
disoutput.lay:
	../makelay -center 1 2 > $@
disoutput.wire: dishidden2.lay disoutput.lay
	../wireup $^ > $@
dis.top: dishidden1.wire dishidden2.wire disoutput.wire
	../maketop $^ > $@
dis.map: dis.top
	../makemap dis.top dis.map 4


encinput.lay: context.lay target.lay
	../catlay $^ > $@
enchidden1.wire: encinput.lay enchidden1.lay
	../wireup $^ > $@
enchidden2.wire: enchidden1.lay enchidden2.lay
	../wireup $^ > $@
encoutput.lay: control.lay
	../catlay $^ > $@
encoutput.wire: enchidden2.lay encoutput.lay
	../wireup $^ > $@
enc.top: enchidden1.wire enchidden2.wire encoutput.wire
	../maketop $^ > $@
enc.map: enc.top
	../makemap enc.top enc.map 4


geninput.lay: context.lay control.lay
	../catlay $^ > $@
genhidden1.wire: geninput.lay genhidden1.lay
	../wireup $^ > $@
genhidden2.wire: genhidden1.lay genhidden2.lay
	../wireup $^ > $@
genoutput.lay: target.lay
	../catlay $^ > $@
genoutput.wire: genhidden2.lay genoutput.lay
	../wireup $^ > $@
gen.top: genhidden1.wire genhidden2.wire genoutput.wire
	../maketop $^ > $@
gen.map: gen.top
	../makemap gen.top gen.map 0.2

clean:
	rm -f *.lay *.wire *.top *.map
