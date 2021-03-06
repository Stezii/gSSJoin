/*********************************************************************
11
12	 Copyright (C) 2015 by Wisllay Vitrio
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

/* *
 * knn.cu
 */

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <queue>
#include <vector>
#include <set>
#include <functional>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "simjoin.cuh"
#include "structs.cuh"
#include "utils.cuh"
#include "inverted_index.cuh"
#include "cuCompactor.cuh"

struct is_bigger_than_threshold
{
	float threshold;
	is_bigger_than_threshold(float thr) : threshold(thr) {};
	__host__ __device__
	bool operator()(const Similarity reg)
	{
		return (reg.similarity > threshold);
	}
};

struct is_smaller_than_threshold
{
	float threshold;
	is_smaller_than_threshold(float thr) : threshold(thr) {};
	__host__ __device__
	bool operator()(const Similarity reg)
	{
		return (reg.similarity < threshold);
	}
};

struct printSimilarity
{
	int docid;
	printSimilarity(int di) : docid(di) {};
  __host__ __device__
  void operator()(const Similarity &reg)
	{
		if (reg.similarity > 0) {
			printf("%i %i: %.10f\n", docid, reg.set_id, reg.similarity);
		}
	}
};

__host__ int findSimilars(InvertedIndex inverted_index, float threshold, int topk, bool topk_is_strict, struct DeviceVariables *dev_vars, Similarity* distances,
		int docid, int querystart, int querysize, float weighted_querysize) {

	dim3 grid, threads;
	get_grid_config(grid, threads);

	int num_sets = inverted_index.num_sets - docid - 1;
	int *d_count = dev_vars->d_count, *d_index = dev_vars->d_index;
	float *d_sim = dev_vars->d_sim, *weighted_size_doc = dev_vars->d_wsizes, *token_weights = dev_vars->d_tokweights;
	int *d_BlocksCount = dev_vars->d_bC, *d_BlocksOffset = dev_vars->d_bO;
	Entry *d_query = inverted_index.d_entries + querystart;
	Similarity *d_similarity = dev_vars->d_dist, *d_result = dev_vars->d_result;

	gpuAssert(cudaMemset(d_sim + docid + 1, 0, num_sets*sizeof(float)));

	get_term_count_and_tf_idf<<<grid, threads>>>(inverted_index, d_query, d_count, querysize);

	thrust::device_ptr<int> thrust_d_count(d_count);
	thrust::device_ptr<int> thrust_d_index(d_index);
	thrust::inclusive_scan(thrust_d_count, thrust_d_count + querysize, thrust_d_index);

	calculateJaccardSimilarity<<<grid, threads>>>(inverted_index, d_query, d_index, d_sim, querysize, docid, token_weights);

	filter_registers<<<grid, threads>>>(d_sim, threshold, weighted_querysize, docid, inverted_index.num_sets, weighted_size_doc, d_similarity);

	Similarity *sim_ptr = d_similarity + docid + 1;
	thrust::device_vector<Similarity> thrust_d_similarity(sim_ptr, sim_ptr + num_sets);
	thrust::sort(thrust_d_similarity.begin(), thrust_d_similarity.end(), thrust::greater<Similarity>());
	if (thrust_d_similarity.size() > topk) {
		if (topk_is_strict) {
			thrust::fill(thrust_d_similarity.begin() + topk, thrust_d_similarity.end(), -1);
		} else {
			Similarity topk_sim = thrust_d_similarity[topk - 1];
			thrust::replace_if(thrust_d_similarity.begin() + topk, thrust_d_similarity.end(), is_smaller_than_threshold(topk_sim.similarity), -1);
		}
	}
	//thrust::for_each(thrust_d_similarity.begin(), thrust_d_similarity.end(), printSimilarity(docid));
	thrust::copy(thrust_d_similarity.begin(), thrust_d_similarity.end(), sim_ptr);

	int blocksize = 1024;
	int numBlocks = cuCompactor::divup(num_sets, blocksize);

	int totalSimilars = cuCompactor::compact2<Similarity>(sim_ptr, d_result, num_sets, is_bigger_than_threshold(threshold), blocksize, numBlocks, d_BlocksCount, d_BlocksOffset);

	if (totalSimilars) cudaMemcpyAsync(distances, d_result, sizeof(Similarity)*totalSimilars, cudaMemcpyDeviceToHost);

	return totalSimilars;
}

__global__ void calculateJaccardSimilarity(InvertedIndex inverted_index, Entry *d_query, int *index, float *dist, int D, int docid, float *token_weights) {
	__shared__ int N;

	if (threadIdx.x == 0) {
		N = index[D - 1];	//Total number of items to be queried
	}
	__syncthreads();

	int block_size = N / gridDim.x + (N % gridDim.x == 0 ? 0 : 1);		//Partition size
	int lo = block_size * (blockIdx.x); 								//Beginning of the block
	int hi = min(lo + block_size, N); 								//End of the block
	int size = hi - lo;											// Real partition size (the last one can be smaller)

	int idx = 0;
	int end;

	for (int i = threadIdx.x; i < size; i += blockDim.x) {
		int pos = i + lo;

		while (true) {
			end = index[idx];

			if (end <= pos) {
				idx++;
			}
			else {
				break;
			}
		}

		Entry entry = d_query[idx]; 		//finds out the term
		int offset = end - pos;

		int idx2 = inverted_index.d_index[entry.term_id] - offset;
		Entry index_entry = inverted_index.d_inverted_index[idx2];

		if (index_entry.set_id > docid) {
			atomicAdd(&dist[index_entry.set_id], token_weights[entry.term_id]);
		}
	}
}


__global__ void get_term_count_and_tf_idf(InvertedIndex inverted_index, Entry *query, int *count, int N) {
	int block_size = N / gridDim.x + (N % gridDim.x == 0 ? 0 : 1);		//Partition size
	int offset = block_size * (blockIdx.x); 				//Beginning of the block
	int lim = min(offset + block_size, N); 					//End of the block
	int size = lim - offset; 						//Block size

	query += offset;
	count += offset;

	for (int i = threadIdx.x; i < size; i += blockDim.x) {
		Entry entry = query[i];

		int idf = inverted_index.d_count[entry.term_id];
		//query[i].tf_idf = entry.tf * log(inverted_index.num_sets / float(max(1, idf)));
		count[i] = idf;
		//atomicAdd(d_qnorm, query[i].tf_idf * query[i].tf_idf);
		//atomicAdd(d_qnorml1, query[i].tf_idf);
	}
}

__global__ void filter_registers(float *sim, float threshold, float querysize, int docid, int N, float *doc_size, Similarity *similars) { // similars + id_doc
	N -= (docid + 1);
	int block_size = N / gridDim.x + (N % gridDim.x == 0 ? 0 : 1);		//Partition size
	int offset = block_size * (blockIdx.x) + docid + 1; 				//Beginning of the block
	int lim = min(offset + block_size, N + docid + 1); 					//End of the block
	int size = lim - offset;

	similars += offset;
	sim += offset;
	doc_size += offset;

	for (int i = threadIdx.x; i < size; i += blockDim.x) {
		float jac = sim[i]/ (float) (querysize + doc_size[i] - sim[i]);

		similars[i].set_id = offset + i;
		similars[i].similarity = jac;
	}
}
