
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sha256.cuh"

#include <stdio.h>
#include <time.h>


__global__ void initSolutionMemory(int* blockContainsSolution) {
	*blockContainsSolution = 0;
}

__device__ int longLatValCalc(int64_t val,BYTE *data) {
	int idx = 0;
	if (val / 100000 % 9) {
		data[idx] = val / 100000 % 9 + 48;
		idx += 1;
	}

	data[idx] = val / 10000 % 10 + 48;
	idx += 1;
	data[idx] = 46; //First Decimal
	idx += 1;
	data[idx] = val / 1000 % 10 + 48;
	idx += 1;
	data[idx] = val/ 100 % 10 + 48;
	idx += 1;
	data[idx] = val / 10 % 10 + 48;
	idx += 1;
	data[idx] = val % 10 + 48;
	idx += 1;

	if (val / 900000000000 % 2) {
		data[idx] = val/ 900000000000 % 2 + 48;
		idx += 1;
	}
	if (val / 90000000000 % 10) {
		data[idx] = val / 90000000000 % 10 + 48;
		idx += 1;
	}

	data[idx] = val / 9000000000 % 10 + 48;
	idx += 1;
	data[idx] = 46; //First Decimal
	idx += 1;
	data[idx] = val / 900000000 % 10 + 48;
	idx += 1;
	data[idx] = val / 90000000 % 10 + 48;
	idx += 1;
	data[idx] = val / 9000000 % 10 + 48;
	idx += 1;
	data[idx] = val / 900000 % 10 + 48;
	idx += 1;

	return idx;
}

__global__ void sha256Cuda(BYTE *hash, BYTE* solution, int* blockContainsSolution, int64_t currVal) {
	//Handle the code on the GPU
	int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) + currVal;
	//atomicMax(blockContainsSolution, i);
	BYTE data[16];
	int len = longLatValCalc(i,data);

	SHA256_CTX ctx;
	BYTE digest[32];
	sha256_init(&ctx);
	sha256_update(&ctx, data, len);
	sha256_final(&ctx, digest);

	BYTE cmpHash[] = { 127,134,6,247,148,9,245,79,168,25,134,244,122,74,36,49,117,100,197,43,37,216,53,123,203,82,26,168,140,133,181,163 };
	for (size_t j = 0; j < 32; j++)
	{
		if (digest[j] != cmpHash[j]) { return; }
	}
	for (int j = 0; j < 33; j++)
		solution[j] = data[j];
	for (int j = 0; j < 33; j++)
		hash[j] = digest[j];
	*blockContainsSolution = 1;
}

void pre_sha256() {
	// compy symbols
	checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}

void launchGPUHandler(unsigned long *hashesProcessed) {
	//Handle launching the threads and blocks for computation
	BYTE* blockSolution = (BYTE*)malloc(sizeof(BYTE) * 33);
	BYTE* d_solution;
	cudaMalloc(&d_solution, sizeof(BYTE) * 33);

	BYTE* blockHash = (BYTE*)malloc(sizeof(BYTE) * 32);
	BYTE* d_hash;
	cudaMalloc(&d_hash, sizeof(BYTE) * 32);

	int* blockContainsSolution = (int*)malloc(sizeof(int));
	int* d_foundSolution;
	cudaMalloc(&d_foundSolution, sizeof(int));

	initSolutionMemory<<<1,1>>>(d_foundSolution);
	int64_t currVal = 0;

	while (true) {
		int threads = 2048;
		int blocks = 512;
		sha256Cuda<<<threads,blocks>>>(d_hash,d_solution,d_foundSolution, currVal);
		currVal += threads * blocks;
		cudaDeviceSynchronize();
		cudaMemcpy(blockContainsSolution, d_foundSolution, sizeof(int), cudaMemcpyDeviceToHost);
		//cudaMemcpy(blockHash, d_hash, sizeof(int), cudaMemcpyDeviceToHost);
		if (*blockContainsSolution) {
			printf("Solution found!\n%d\n",*blockContainsSolution);
			break;
		}
		else {
			printf("%I64d   ", currVal);
			printf("%02X:%02X:%02X:%02X\n", blockHash[0], blockHash[1], blockHash[2], blockHash[3]);
		}
	}
	cudaMemcpy(blockSolution, d_solution, sizeof(BYTE) * 33, cudaMemcpyDeviceToHost);
	cudaMemcpy(blockHash, d_hash, sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceReset();
	printf("%02X:%02X:%02X:%02X\n", blockHash[0], blockHash[1], blockHash[2], blockHash[3]);
	printf("%s\n", blockSolution);
}

int main()
{
	clock_t start = clock();
	pre_sha256();
	unsigned long hashesProcessed;
	launchGPUHandler(&hashesProcessed);
	//while (true) {
		//Handle hashesProcessed update loop and killing of loop if solution is found
	//}
	clock_t end = clock();
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("%.6f", seconds);
}
