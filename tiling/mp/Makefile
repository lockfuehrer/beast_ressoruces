CXX=clang++
FLAGS=-std=c++17 -stdlib=libc++ -lstdc++ -fvectorize -lrt -lm -lpthread -fopenmp -target x86_64-pc-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906

MAIN=main.cpp
BIN=bin
EXE=mp
OUT=ouput

DATA=input/randomlist128.txt
LEN=2048
SUBLEN=10

all: prepare compile run 

prepare:
	bash prepare.sh

compile:
	$(CXX) $(FLAGS) $(MAIN) -o $(BIN)/$(EXE) -lm

compile_debug:
	$(CXX) $(FLAGS) $(MAIN) -o $(BIN)/$(EXE) -lm -DDEB

run: compile
	$(BIN)/$(EXE) $(DATA) 1024 $(SUBLEN)
	$(BIN)/$(EXE) $(DATA) 4096 $(SUBLEN)
	$(BIN)/$(EXE) $(DATA) 16384 $(SUBLEN)

clean: run
	rm -rf $(BIN)/$(EXE)
	rm -rf $(BIN) $(OUT)
