/*********************************************************************
11
12	 Copyright (C) 2016 by Sidney Ribeiro Junior
13
14	 This program is free software; you can redistribute it and/or modify
15	 it under the terms of the GNU General Public License as published by
16	 the Free Software Foundation; either version 2 of the License, or
17	 (at your option) any later version.
18
19	 This program is distributed in the hope that it will be useful,
20	 but WITHOUT ANY WARRANTY; without even the implied warranty of
21	 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
22	 GNU General Public License for more details.
23
24	 You should have received a copy of the GNU General Public License
25	 along with this program; if not, write to the Free Software
26	 Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
27
28	 ********************************************************************/

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <vector>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <string>
#include <sstream>
#include <cuda.h>
#include <map>

#include "structs.cuh"
#include "utils.cuh"
#include "inverted_index.cuh"
#include "simjoin.cuh"


#define OUTPUT 1
#define NUM_STREAMS 1


using namespace std;

struct FileStats {
	int num_sets;
	int num_terms;

	vector<int> sizes; // set sizes
	vector<float> weighted_sizes; // weighted set sizes
	vector<int> start; // beginning of each entry
	vector<float> token_weights; // weights of each token

	FileStats() : num_sets(0), num_terms(0) {}
};

FileStats readInputFiles(string &sets_filename, string &weights_filename, vector<Entry> &entries, vector<string> &ids);
void processTestFile(InvertedIndex &index, FileStats &stats, string &file, vector<string> &ids, float threshold, int topk, bool topk_is_strict, stringstream &fileout);


/**
 * Receives as parameters the training file name and the test file name
 */

static int num_tests = 0;
int biggestQuerySize = -1;


int main(int argc, char **argv) {

	if (argc != 8) {
		cerr << "Wrong parameters. Correct usage: <executable> <input_token_file> <input_weights_file> <threshold> <topk> <topk_mode> <output_file> <number_of_gpus>" << endl;
		exit(1);
	}

	bool topk_is_strict;
	string topk_mode(argv[5]);
	if (topk_mode == "strict") {
		topk_is_strict = true;
	} else if (topk_mode == "soft") {
		topk_is_strict = false;
	} else {
		cerr << "Wrong parameter 'topk_mode'. Must be 'strict' or 'soft'" << endl;
		exit(1);
	}

	int gpuNum;
	cudaGetDeviceCount(&gpuNum);

	if (gpuNum > atoi(argv[7])) {
		gpuNum = atoi(argv[7]);
		if (gpuNum < 1)
			gpuNum = 1;
	}
	//cerr << "Using " << gpuNum << "GPUs" << endl;

	// we use 2 streams per GPU
	int numThreads = gpuNum*NUM_STREAMS;

	omp_set_num_threads(numThreads);

#if OUTPUT
	//truncate output files
	ofstream ofsf(argv[6], ofstream::trunc);
	ofsf.close();

	ofstream ofsfileoutput(argv[6], ofstream::out | ofstream::app);
#endif
	vector<string> inputs;// to read the whole test file in memory
	vector<InvertedIndex> indexes;
	indexes.resize(gpuNum);

	double starts, ends;

	string inputSetsFileName(argv[1]);
	string inputWeightsFileName(argv[2]);

	printf("Reading files...\n");
	vector<Entry> entries;
	vector<string> ids;

	starts = gettime();
	FileStats stats = readInputFiles(inputSetsFileName, inputWeightsFileName, entries, ids);
	ends = gettime();

	printf("Time taken: %lf seconds\n", ends - starts);

	vector<stringstream*> outputString;
	//Each thread builds an output string, so it can be flushed at once at the end of the program
	for (int i = 0; i < numThreads; i++) {
		outputString.push_back(new stringstream);
	}

	//create an inverted index for all streams in each GPU
	#pragma omp parallel num_threads(gpuNum)
	{
		int cpuid = omp_get_thread_num();
		cudaSetDevice(cpuid);
		double start, end;

		start = gettime();
		indexes[cpuid] = make_inverted_index(stats.num_sets, stats.num_terms, entries);
		end = gettime();

		#pragma omp single nowait
		printf("Total time taken for insertion: %lf seconds\n", end - start);
	}


	#pragma omp parallel
	{
		int cpuid = omp_get_thread_num();
		cudaSetDevice(cpuid / NUM_STREAMS);

		float threshold = atof(argv[3]);
		float topk = atof(argv[4]);

		FileStats lstats = stats;

		processTestFile(indexes[cpuid / NUM_STREAMS], lstats, inputSetsFileName, ids, threshold, topk, topk_is_strict, *outputString[cpuid]);
		if (cpuid %  NUM_STREAMS == 0)
			gpuAssert(cudaDeviceReset());

	}

#if OUTPUT
		starts = gettime();
		for (int i = 0; i < numThreads; i++) {
			ofsfileoutput << outputString[i]->str();
		}
		ends = gettime();

		printf("Time taken to write output: %lf seconds\n", ends - starts);

		ofsfileoutput.close();
#endif
		return 0;
}

FileStats readInputFiles(string &sets_filename, string &weights_filename, vector<Entry> &entries, vector<string> &ids) {
	string line;
	FileStats stats;

	// get number of terms and check weights file
	ifstream input_weights(weights_filename.c_str());
	stats.num_terms = 1; // must start at 1 for zero-based array access

	while (!input_weights.eof()) {
		getline(input_weights, line);
		if (line == "") continue;

		vector<string> line_spl = split(line, ' ');
		int token = atoi(line_spl[0].c_str());
		float weight = atof(line_spl[1].c_str());
		if (stats.num_terms == 1 && token != 1) {
			cerr << "Error in " << weights_filename << ": First token id must be 1 in " << endl;
			exit(1);
		}
		if (stats.num_terms != token) {
			cerr << "Error in " << weights_filename << ": Token " << stats.num_terms << " is missing" << endl;
			exit(1);
		}
		if (weight < 0) {
			cerr << "Error in " << weights_filename << ": Token weight may not be smaller than 0" << endl;
			exit(1);
		}
		stats.num_terms++;
	}

	// read weights
	input_weights.clear();
	input_weights.seekg(0, ios::beg);
	float token_weights[stats.num_terms];

	while (!input_weights.eof()) {
		getline(input_weights, line);
		if (line == "") continue;

		vector<string> line_spl = split(line, ' ');
		int token = atoi(line_spl[0].c_str());
		float weight = atof(line_spl[1].c_str());
		token_weights[token] = weight;
	}

	input_weights.close();

	vector<float> vec(token_weights, token_weights + stats.num_terms);
	stats.token_weights = vec;

	// read sets
	ifstream input_sets(sets_filename.c_str());
	int accumulatedsize = 0;
	int set_id = 0;

	while (!input_sets.eof()) {
		getline(input_sets, line);
		if (line == "") continue;

		vector<string> line_spl = split(line, ' ');
		vector<string> tokens(line_spl.begin() + 1, line_spl.begin() + (int)line_spl.size());
		ids.push_back(line_spl[0]);

		biggestQuerySize = max((int)tokens.size(), biggestQuerySize);

		int size = tokens.size();
		float weighted_size = 0;
		stats.sizes.push_back(size);
		stats.start.push_back(accumulatedsize);
		accumulatedsize += size;

		for (int i = 0; i < size; i++) {
			int term_id = atoi(tokens[i].c_str());
			entries.push_back(Entry(set_id, term_id));
			weighted_size += stats.token_weights.at(term_id);
		}
		stats.weighted_sizes.push_back(weighted_size);
		set_id++;
	}

	stats.num_sets = stats.start.size();

	input_sets.close();

	return stats;
}

void allocVariables(DeviceVariables *dev_vars, float threshold, int num_sets, int num_terms, Similarity** distances) {
	dim3 grid, threads;

	get_grid_config(grid, threads);

	gpuAssert(cudaMalloc(&dev_vars->d_dist, num_sets * sizeof(Similarity))); // distance between all the sets and the query doc
	gpuAssert(cudaMalloc(&dev_vars->d_result, num_sets * sizeof(Similarity))); // compacted similarities between all the sets and the query doc
	gpuAssert(cudaMalloc(&dev_vars->d_sim, num_sets * sizeof(float))); // count of elements in common
	gpuAssert(cudaMalloc(&dev_vars->d_wsizes, num_sets * sizeof(float))); // weighted size of all sets
	gpuAssert(cudaMalloc(&dev_vars->d_tokweights, num_terms * sizeof(float))); // weights of each token
	gpuAssert(cudaMalloc(&dev_vars->d_query, biggestQuerySize * sizeof(Entry))); // query
	gpuAssert(cudaMalloc(&dev_vars->d_index, biggestQuerySize * sizeof(int)));
	gpuAssert(cudaMalloc(&dev_vars->d_count, biggestQuerySize * sizeof(int)));

	*distances = (Similarity*)malloc(num_sets * sizeof(Similarity));

	int blocksize = 1024;
	int numBlocks = num_sets / blocksize + (num_sets % blocksize ? 1 : 0);

	gpuAssert(cudaMalloc(&dev_vars->d_bC,sizeof(int)*(numBlocks + 1)));
	gpuAssert(cudaMalloc(&dev_vars->d_bO,sizeof(int)*numBlocks));

}

void freeVariables(DeviceVariables *dev_vars, InvertedIndex &index, Similarity** distances) {
	cudaFree(dev_vars->d_dist);
	cudaFree(dev_vars->d_result);
	cudaFree(dev_vars->d_sim);
	cudaFree(dev_vars->d_wsizes);
	cudaFree(dev_vars->d_tokweights);
	cudaFree(dev_vars->d_query);
	cudaFree(dev_vars->d_index);
	cudaFree(dev_vars->d_count);
	cudaFree(dev_vars->d_bC);
	cudaFree(dev_vars->d_bO);

	free(*distances);

	if (omp_get_thread_num() % NUM_STREAMS == 0) {
		cudaFree(index.d_count);
		cudaFree(index.d_index);
		cudaFree(index.d_inverted_index);
	}
}

void processTestFile(InvertedIndex &index, FileStats &stats, string &filename, vector<string> &ids, float threshold, int topk, bool topk_is_strict, stringstream &outputfile) {

	int num_test_local = 0, setid;

	//#pragma omp single nowait
	printf("Processing input file %s...\n", filename.c_str());

	DeviceVariables dev_vars;
	Similarity* distances;

	allocVariables(&dev_vars, threshold, index.num_sets, stats.num_terms, &distances);

	cudaMemcpyAsync(dev_vars.d_wsizes, &stats.weighted_sizes[0], index.num_sets * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(dev_vars.d_tokweights, &stats.token_weights[0], stats.num_terms * sizeof(float), cudaMemcpyHostToDevice);

	double start = gettime();

#pragma omp for
	for (setid = 0; setid < index.num_sets - 1; setid++) {

		num_test_local++;

		int totalSimilars = findSimilars(index, threshold, topk, topk_is_strict, &dev_vars, distances, setid, stats.start[setid], stats.sizes[setid], stats.weighted_sizes[setid]);

#if OUTPUT
		for (int i = 0; i < totalSimilars; i++) {
			outputfile << ids[setid] << "\t" << ids[distances[i].set_id] << "\t" << distances[i].similarity << endl;
		}
#endif

	}

	freeVariables(&dev_vars, index, &distances);
	int threadid = omp_get_thread_num();

	printf("Entries in device %d stream %d: %d\n", threadid / NUM_STREAMS, threadid %  NUM_STREAMS, num_test_local);

	#pragma omp barrier

	double end = gettime();

	#pragma omp master
	{
		printf("Time taken for %d queries: %lf seconds\n\n", num_tests, end - start);
	}
}
