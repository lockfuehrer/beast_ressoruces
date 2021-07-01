#!/bin/sh
thx=(-fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target=nvptx64 -march=sm_70)
rome=(-fopenmp -target x86_64-pc-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906)
FLAG=${1%.*}
std=( -std=c++17 -stdlib=libc++)
omp=( -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target=nvptx64 -march=sm_70)
rest=( -lstdc++ -fvectorize -lrt -lm -lpthread )
clang++ ${std[@]} ${rome[@]} ${rest[@]} -o "$FLAG" "$FLAG".cpp



