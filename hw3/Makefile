CC = clang
CXX = clang++
FLAGS = -fopenmp -pthread -Wall -Wextra -march=native -Ofast
CXXFLAGS = $(FLAGS)
CFLAGS = -lm $(FLAGS)

NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
NVFLAGS += -Xcompiler "-fopenmp -pthread -Wall -Wextra -march=native"
LDFLAGS  := -lm

EXES     := hw3-1 hw3-2 hw3-3

alls: $(EXES)

clean:
	rm -f $(EXES)

hw3-1: hw3-1.cc
	mpicxx  -O3 -fopenmp -o $@ $?

hw3-2: hw3-2.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-3: hw3-3.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?