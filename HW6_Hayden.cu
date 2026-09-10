// Name: Hayden Pegues
// Simple Julia CPU.
// nvcc HW6_Hayden.cu -o HW6 -lglut -lGL
// glut and GL are openGL libraries.
/*
 What to do:
 This code displays a simple Julia fractal using the CPU.
 Rewrite the code so that it uses the GPU to create the fractal. 
 Keep the window at 1024 by 1024.
 Use __device__ for the escapeOrNotColor function
*/

/*
 Purpose:
 To apply your new GPU skills to do  something cool!
*/

/*
 Explain what you did to fix the code:
 Lines 65-77: Turned the while into a for loop and deleted unnecessary variables
 Lines 81-98: Added this escapeKernel because the display loop can't call a device function
 Lines 110-116: Added device buffer to pass pixel information to the CPU to display it (also added cudaMemcpy line to help with this)
 Lines 121-123: Added our dimensions in display function
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  -0.824	//Real part of C
#define B  -0.1711	//Imaginary part of C

// Global variables
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

// Function prototypes
void cudaErrorCheck(const char*, int);
__device__ float escapeOrNotColor(float, float);

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

__device__ float escapeOrNotColor (float x, float y) 
{
	float tempX;

	for (int count = 0; count < MAXITERATIONS; count++) 
	{	
		if(x*x + y*y > MAXMAG*MAXMAG)
		{
			return 0.0f;
		}
		
		tempX = x; //We will be changing the x but we need its old value to find y.
		x = x*x - y*y + A;
		y = (2.0f * tempX * y) + B;
	}
	
	return(1.0f);

}

__global__ void escapeKernel(float *pixels, int Width, int Height, float xMin, float yMin, float stepSizeX, float stepSizeY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < Width && y < Height)
	{
		float X = xMin + x * stepSizeX;
		float Y = yMin + y * stepSizeY;
		
		float color = escapeOrNotColor(X, Y);

		int k = (y * Width + x) * 3;
		pixels[k] = 0.0f;
		pixels[k + 1] = color; //Sadie made me do it                
        pixels[k + 2] = 0.0f;
	}
}

void display(void) 
{ 
	float *pixels; 
	float *devicePixels;
	float stepSizeX, stepSizeY;
	
	int numPixels = WindowWidth * WindowHeight;
    size_t bufferSize = numPixels * 3 * sizeof(float);

    // Host buffer (each pixel has red, green, and blue).
    pixels = (float *)malloc(bufferSize);

    // Device buffer.
    cudaMalloc(&devicePixels, bufferSize);
    cudaErrorCheck(__FILE__, __LINE__);
	cudaMemset(devicePixels, 0, bufferSize);   // clear buffer to black
    cudaErrorCheck(__FILE__, __LINE__);
	
	stepSizeX = (XMax - XMin)/((float)WindowWidth);
	stepSizeY = (YMax - YMin)/((float)WindowHeight);

	dim3 BlockSize(16, 16);
	dim3 GridSize((WindowWidth  + BlockSize.x - 1) / BlockSize.x,
			      (WindowHeight + BlockSize.y - 1) / BlockSize.y);

	escapeKernel<<<GridSize, BlockSize>>>(devicePixels, WindowWidth, WindowHeight, XMin, YMin, stepSizeX, stepSizeY);

	cudaMemcpy(pixels, devicePixels, bufferSize, cudaMemcpyDeviceToHost);
    cudaErrorCheck(__FILE__, __LINE__);

	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixels); 
	glFlush(); 
}

int main(int argc, char** argv)
{ 

   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);
   	glutMainLoop();
}

