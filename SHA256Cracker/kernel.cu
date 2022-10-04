
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sha256.cuh"

#include <stdio.h>
#include <time.h>

__global__ void sha256Cuda(BYTE *solution) {
	//Handle the code on the GPU
	SHA256_CTX ctx;
	BYTE digest[32];
	sha256_init(&ctx);
	sha256_update(&ctx, "TestTestTest", 12);
	sha256_final(&ctx, digest);
	for (int j = 0; j < 32; j++)
		solution[j] = digest[j];
}

void launchGPUHandler(unsigned long *hashesProcessed) {
	//Handle launching the threads and blocks for computation
	BYTE* blockSolution = (BYTE*)malloc(sizeof(BYTE) * 32);
	BYTE* d_solution;
	cudaMalloc(&d_solution, sizeof(BYTE) * 32);
	sha256Cuda<<<1,1>>>(d_solution);
	cudaDeviceSynchronize();
	cudaMemcpy(blockSolution, d_solution, sizeof(BYTE) * 32, cudaMemcpyDeviceToHost);
	printf("Solution: %.32s\n", blockSolution);
	cudaDeviceReset();
}

int main()
{
	clock_t start = clock();
	unsigned long hashesProcessed;
	launchGPUHandler(&hashesProcessed);
	//while (true) {
		//Handle hashesProcessed update loop and killing of loop if solution is found
	//}
	clock_t end = clock();
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("%.6f", seconds);
}
