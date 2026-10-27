// Name: Hayden Pegues
// GPU random walk. 
// nvcc HW18_Hayden.cu -o HW18

/*
 What to do:
 This code runs a random walk for 10,000 steps on the CPU.

 1. Use cuRAND to run 20 random walks simultaneously on the GPU, each with a different seed.
    Print out all 20 final positions.

 2. Use cudaMallocManaged(&variable, amount_of_memory_needed);
    This allocates unified memory, which is automatically managed between the CPU and GPU.
    You lose some control over placement, but it saves you from having to manually copy data
    to and from the GPU.
*/

/*
 Purpose:
 To learn how to use cuRAND and unified memory.
*/

/*
 Explain what you did to fix the code:
 
*/

/*
 Note:
 The maximum signed int value is 2,147,483,647, so the maximum unsigned int value is 4,294,967,295.

 RAND_MAX is guaranteed to be at least 32,767. When I checked it on my laptop (10/6/2025), it was 2,147,483,647.
 rand() returns a value in [0, RAND_MAX]. It actually generates a list of pseudo-random numbers that depends on the seed.
 This list eventually repeats (this is called its period). The period is usually 2³¹ = 2,147,483,648,
 but it may vary by implementation.

 Because RAND_MAX is odd on this machine and 0 is included, there is no exact middle integer.
 Casting to float as in (float)RAND_MAX / 2.0 divides the range evenly.
 Using integer division (RAND_MAX / 2) would bias results slightly toward the positive side by one value out of 2,147,483,647.

 I know this is splitting hares (sorry, rabbits), but I'm just trying to be as accurate as possible.
 You might do this faster with a clever integer approach, but I’m using floats here for clarity.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Defines
#define walk_number 20

// Globals
int NumberOfRandomSteps = 10000;
float MidPoint = (float)RAND_MAX/2.0f;

// Function prototypes
int getRandomDirection();
int main(int, char**);
__global__ void randomWalk(int *positionX, int *positionY, int numberOfSteps, unsigned long long baseSeed);

int getRandomDirection()
{	
	int randomNumber = rand();
	
	if(randomNumber < MidPoint) return(-1);
	else return(1);
}

__global__ void randomWalk(int *positionX, int *positionY, int numberOfSteps, unsigned long long baseSeed)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < walk_number)
    {
        // Each walk gets a different seed so each sequence differs.
        curandState state;
        curand_init(baseSeed + id, id, 0, &state);

        int x = 0;
        int y = 0;
        for(int i = 0; i < numberOfSteps; i++)
        {
            // curand_uniform returns a float in (0, 1].
            float rX = curand_uniform(&state);
            float rY = curand_uniform(&state);

            if(rX < 0.5f) x += -1; else x += 1;
            if(rY < 0.5f) y += -1; else y += 1;
        }

        positionX[id] = x;
        positionY[id] = y;
    }
}

int main(int argc, char** argv)
{
	/*
	srand(time(NULL));
	
	printf(" RAND_MAX for this implementation is = %d \n", RAND_MAX);
	
	int positionX = 0;
	int positionY = 0;
	for(int i = 0; i < NumberOfRandomSteps; i++)
	{
		positionX += getRandomDirection();
		positionY += getRandomDirection();
	}
	*/
	 // Unified memory arrays shared between CPU and GPU.
    int *positionX;
    int *positionY;

    cudaMallocManaged(&positionX, walk_number * sizeof(int));
    cudaMallocManaged(&positionY, walk_number * sizeof(int));

    // Use the current time as a base seed; each walk offsets this by its id.
    unsigned long long baseSeed = (unsigned long long)time(NULL);

    // Launch one thread per walk (20 total).
    int threadsPerBlock = walk_number;
    int blocks = 1;
    randomWalk<<<blocks, threadsPerBlock>>>(positionX, positionY, NumberOfRandomSteps, baseSeed);

    // Wait for the GPU to finish before reading the unified memory.
    cudaDeviceSynchronize();

    printf("\n Final positions for %d random walks (%d steps each):\n", walk_number, NumberOfRandomSteps);
    for(int i = 0; i < walk_number; i++)
    {
        printf(" Walk %2d: (%d, %d)\n", i, positionX[i], positionY[i]);
    }
    printf("\n");

    // Free unified memory.
    cudaFree(positionX);
    cudaFree(positionY);
	
	//printf("\n Final position = (%d,%d) \n", positionX, positionY);
	
	return 0;
}

