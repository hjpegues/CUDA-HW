// Name: Hayden Pegues
// Device query
// nvcc HW5_Hayden.cu -o HW5
/*
 What to do:
 This code prints out useful information about the GPU(s) in your machine, 
 but there is much more data available in the cudaDeviceProp structure.

 Extend this code so that it prints out all the information about the GPU(s) in your system. 
 Also, and this is the fun part, be prepared to explain what each piece of information means. 
*/

/*
 Purpose:
 To learn how to find out what is on the GPU(s) in your machine and if you even have a GPU.
*/

/*
 Explain what you did to fix the code:
 Line 80: Added a line to print the memory bus width, which tells you the number of bits
 that can be moved in a single cycle
 Line 81: Added a line to print the clock rate of the memory on the GPU
 Line 84: Added a line that tells us shared mem per mp
 Line 85: Changed "mp" to "Block" in the print statement
 Line 86: Added a line that tells us how much memory the computer reserves that our
 CUDA code can't read
 Line 87: Added a line to tell us how many registers per mp we have
 Line 88: Changed "mp" to "Block" in the print statement
*/

// Include files
#include <stdio.h>

// Defines

// Global variables

// Function prototypes
void cudaErrorCheck(const char*, int);

void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

int main()
{
	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount(&count);
	cudaErrorCheck(__FILE__, __LINE__);
	printf(" You have %d GPUs in this machine\n", count);
	
	for (int i=0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		cudaErrorCheck(__FILE__, __LINE__);
		printf(" ---General Information for device %d ---\n", i);
		printf("Name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Clock rate: %d\n", prop.clockRate);
		printf("Device copy overlap: ");
		if (prop.deviceOverlap) printf("Enabled\n");
		else printf("Disabled\n");
		printf("Kernel execution timeout : ");
		if (prop.kernelExecTimeoutEnabled) printf("Enabled\n");
		else printf("Disabled\n");
		printf(" ---Memory Information for device %d ---\n", i);
		printf("Total global mem: %ld\n", prop.totalGlobalMem);
		printf("Total constant Mem: %ld\n", prop.totalConstMem);
		printf("Max mem pitch: %ld\n", prop.memPitch);
		printf("Texture Alignment: %ld\n", prop.textureAlignment);
		printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
		printf("Memory Clock Rate (kHz): %d\n", prop.memoryClockRate);
		printf(" ---MP Information for device %d ---\n", i);
		printf("Multiprocessor count : %d\n", prop.multiProcessorCount); //Number of SMs on the GPU
		printf("Shared mem per MP: %ld\n", prop.sharedMemPerMultiprocessor);
		printf("Shared mem per Block: %ld\n", prop.sharedMemPerBlock);
		printf("Reserved Shared Mem per Block (bytes): %ld\n", prop.reservedSharedMemPerBlock);
		printf("Registers per MP: %d\n", prop.regsPerMultiprocessor);
		printf("Registers per Block: %d\n", prop.regsPerBlock);
		printf("Threads in warp: %d\n", prop.warpSize);
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("\n");
	}	
	return(0);
}

