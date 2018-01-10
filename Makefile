OMP= -Xcompiler -fopenmp

Release: main.o simjoin.o inverted_index.o utils.o
	nvcc -arch=sm_20 -O3 -lgomp  main.o simjoin.o inverted_index.o  utils.o -o sim

main.o: main.cu simjoin.cuh inverted_index.cuh utils.cuh structs.cuh
	nvcc -arch=sm_20 -O3 $(OMP)  -c main.cu -o main.o
	
simjoin.o: simjoin.cu simjoin.cuh inverted_index.cuh utils.cuh structs.cuh 
	nvcc -arch=sm_20 -O3  $(OMP) -c simjoin.cu -o simjoin.o

inverted_index.o: inverted_index.cu inverted_index.cuh utils.cuh 
	nvcc -arch=sm_20 -O3  $(OMP) -c inverted_index.cu -o inverted_index.o

utils.o: utils.cu utils.cuh 
	nvcc -arch=sm_20 -O3 $(OMP)  -c utils.cu -o utils.o

clean:
	rm *.o sim
