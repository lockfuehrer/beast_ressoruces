CXX=clang++
FLAGS=-std=c++17 -stdlib=libc++ -lstdc++ -fvectorize -lrt -lm -lpthread -fopenmp -target x86_64-pc-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906
FLAG=-fopenmp -target x86_64-pc-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906

OBJS = main.cpp

all: run 

multigrid: clean
	$(CXX) $(FLAG) $(OBJS) -o multigrid
clean: run 
	rm -f *~ *.o *.vtk multigrid
run: multigrid
	./multigrid 4 6 14 8

