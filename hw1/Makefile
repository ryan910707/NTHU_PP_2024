CC = mpicc
CXX = mpicxx
CXXFLAGS = -O3 -lm
CFLAGS = -O3 -lm
TARGETS = hw1

.PHONY: all
all: $(TARGETS)

$(TARGETS): hw1.cc
	$(CXX)	$(CXXFLAGS)	-o	$@	$^

.PHONY: clean
clean:
	rm -f $(TARGETS)
