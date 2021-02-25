CXXFLAGS:= -Wall -std=c++17 -O3 -DARMA_DONT_USE_WRAPPER -DARMA_NO_DEBUG

all:
	clang++ main.cc -o main $(CXXFLAGS) -lopenblas -llapack -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5 -lhdf5_cpp -march=native -larmadillo -fopenmp `pkg-config --libs openssl`

