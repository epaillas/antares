# compiler choice
CC    = gcc

all: fastmodules c

.PHONY : fastmodules

c:
	make -C antares/voids/c all

fastmodules:
	python antares/voids/setup.py build_ext --inplace
	mv fastmodules*.so antares/voids/

clean:
	rm -f antares/voids/*.*o
	rm -f antares/voids/fastmodules.c
	rm -f antares/voids/fastmodules*.so
	rm -f antares/voids/*.pyc
	rm -f antares/voids/c/*.o
	rm -f antares/voids/c/*.exe
