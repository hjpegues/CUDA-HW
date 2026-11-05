// Name: Hayden Pegues
// Creating a GPU nBody simulation from an nBody CPU simulation. 
// nvcc HW21_Hayden.cu -o HW21 -lglut -lm -lGLU -lGL

/*
 What to do:
 This is some lean nBody code that runs on the CPU. Rewrite it, keeping the same general format, 
 but offload the compute-intensive parts of the code to the GPU for acceleration.
 Note: The code takes two arguments as inputs:
 1. The number of bodies to simulate, (We will keep the number of bodies under 1024 for this HW so it can be run on one block.)
 2. Whether to draw sub-arrangements of the bodies during the simulation (1), or only the first and last arrangements (0).
*/

/*
 Purpose:
 To learn how to move an Nbody CPU simulation to an Nbody GPU simulation.
*/

/*
 Explain what you did to fix the code:
 In Defines: Defined maxinum numer of bodies so we only use one block
 Under globals: added ned global variables for our GPU
 Added new function prototypes; cudaErrorCheck, getForces, and moveBodies
 Inside setup: I cudamalloc memory into our GPU variables, declare block and grid size, and then cudamemcpy 
			   data from our initial P, V, F, and M variables into the cuda versions of each
 Created the force function which finds the forces on each body and onlt writes to f[i] so nothing is overwritten
 Created the moveBodies function, which uses the leapfrog method discussed in class
 Completely rewrote the nbody function so it uses all GPU functions
 Inside main: Checks to make sure you haven't called more bodies than threads allowed
*/

// Include files
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Defines
#define PI 3.14159265359
#define DRAW_RATE 10

// This is to create a Lennard-Jones type function G/(r^p) - H(r^q). (p < q) p has to be less than q.
// In this code we will keep it a p = 2 and q = 4 problem. The diameter of a body is found using the general
// case so it will be more robust but in the code leaving it as a set 2, 4 problem make the coding much easier.

// And always remember: little bottoms make big tops
#define G 10.0
#define H 10.0
#define LJP  2.0
#define LJQ  4.0

#define MAX_BODIES 1024

#define DT 0.0001
#define RUN_TIME 1.0

// Globals
int N, DrawFlag;
float3 *P, *V, *F;
float *M; 
float GlobeRadius, Diameter, Radius;
float Damp;
float3 *PGPU, *VGPU, *FGPU;
float *MGPU;
dim3 BlockSize, GridSize;

// Function prototypes
void cudaErrorCheck(const char*, int);
void keyPressed(unsigned char, int, int);
long elaspedTime(struct timeval, struct timeval);
void drawPicture();
void timer();
void setup();
__global__ void getForces(float3*, float3*, float*, int);
__global__ void moveBodies(float3*, float3*, float3*, float*, float, float, int, int);
void nBody();
int main(int, char**);

void cudaErrorCheck(const char *file, int line)
{
    cudaError_t error;
    error = cudaGetLastError();

    if(error != cudaSuccess)
    {
        printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
        exit(0);
    }
}

void keyPressed(unsigned char key, int x, int y)
{
	if(key == 's')
	{
		timer();
	}
	
	if(key == 'q')
	{
		exit(0);
	}
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	// Returning the total time elasped in microseconds
	return endTime - startTime;
}

void drawPicture()
{
	int i;
	
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	glColor3d(1.0,1.0,0.5);
	for(i=0; i<N; i++)
	{
		glPushMatrix();
		glTranslatef(P[i].x, P[i].y, P[i].z);
		glutSolidSphere(Radius,20,20);
		glPopMatrix();
	}
	
	glutSwapBuffers();
}

void timer()
{	
	timeval start, end;
	long computeTime;
	
	drawPicture();
	gettimeofday(&start, NULL);
    		nBody();
    	gettimeofday(&end, NULL);
    	drawPicture();
    	
	computeTime = elaspedTime(start, end);
	printf("\n The compute time was %ld microseconds.\n\n", computeTime);
}

void setup()
{
    	float randomAngle1, randomAngle2, randomRadius;
    	float d, dx, dy, dz;
    	int test;
    	
    	Damp = 0.5;
    	
    	M = (float*)malloc(N*sizeof(float));
    	P = (float3*)malloc(N*sizeof(float3));
    	V = (float3*)malloc(N*sizeof(float3));
    	F = (float3*)malloc(N*sizeof(float3));

		cudaMalloc(&MGPU, N*sizeof(float));
        cudaErrorCheck(__FILE__, __LINE__);
        cudaMalloc(&PGPU, N*sizeof(float3));
        cudaErrorCheck(__FILE__, __LINE__);
        cudaMalloc(&VGPU, N*sizeof(float3));
        cudaErrorCheck(__FILE__, __LINE__);
        cudaMalloc(&FGPU, N*sizeof(float3));
        cudaErrorCheck(__FILE__, __LINE__);
        
        // N is under 1024 for this HW so the whole simulation fits in one block.
        // Rounding the block up to a full warp so we never launch a partial warp.
        BlockSize.x = ((N - 1)/32 + 1)*32;
        BlockSize.y = 1;
        BlockSize.z = 1;
        
        GridSize.x = 1;
        GridSize.y = 1;
        GridSize.z = 1;
	
	Diameter = pow(H/G, 1.0/(LJQ - LJP)); // This is the value where the force is zero for the L-J type force.
	Radius = Diameter/2.0;
	
	// Using the radius of a body and a 68% packing ratio to find the radius of a global sphere that should hold all the bodies.
	// Then we double this radius just so we can get all the bodies setup with no problems. 
	float totalVolume = float(N)*(4.0/3.0)*PI*Radius*Radius*Radius;
	totalVolume /= 0.68;
	float totalRadius = pow(3.0*totalVolume/(4.0*PI), 1.0/3.0);
	GlobeRadius = 2.0*totalRadius;
	
	// Randomly setting these bodies in the glaobal sphere and setting the initial velosity, initial force, and mass.
	for(int i = 0; i < N; i++)
	{
		test = 0;
		while(test == 0)
		{
			// Get random position.
			randomAngle1 = ((float)rand()/(float)RAND_MAX)*2.0*PI;
			randomAngle2 = ((float)rand()/(float)RAND_MAX)*PI;
			randomRadius = ((float)rand()/(float)RAND_MAX)*GlobeRadius;
			P[i].x = randomRadius*cos(randomAngle1)*sin(randomAngle2);
			P[i].y = randomRadius*sin(randomAngle1)*sin(randomAngle2);
			P[i].z = randomRadius*cos(randomAngle2);
			
			// Making sure the balls centers are at least a diameter apart.
			// If they are not throw these positions away and try again.
			test = 1;
			for(int j = 0; j < i; j++)
			{
				dx = P[i].x-P[j].x;
				dy = P[i].y-P[j].y;
				dz = P[i].z-P[j].z;
				d = sqrt(dx*dx + dy*dy + dz*dz);
				if(d < Diameter)
				{
					test = 0;
					break;
				}
			}
		}
	
		V[i].x = 0.0;
		V[i].y = 0.0;
		V[i].z = 0.0;
		
		F[i].x = 0.0;
		F[i].y = 0.0;
		F[i].z = 0.0;
		
		M[i] = 1.0;
	}

	// Pushing the initial arrangement up to the GPU one time.
    cudaMemcpy(PGPU, P, N*sizeof(float3), cudaMemcpyHostToDevice);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMemcpy(VGPU, V, N*sizeof(float3), cudaMemcpyHostToDevice);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMemcpy(FGPU, F, N*sizeof(float3), cudaMemcpyHostToDevice);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMemcpy(MGPU, M, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaErrorCheck(__FILE__, __LINE__);

	printf("\n To start timing type s.\n");
}

__global__ void getForces(float3 *p, float3 *f, float *m, int n)
{
    // Tiling the positions and masses into shared memory since the inner loop reads them n times.
    __shared__ float3 shPos[MAX_BODIES];
    __shared__ float  shMass[MAX_BODIES];
    
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    
    if(i < n)
    {
        shPos[i]  = p[i];
        shMass[i] = m[i];
    }
    __syncthreads();
    
    if(i < n)
    {
        float3 force;
        force.x = 0.0f;
        force.y = 0.0f;
        force.z = 0.0f;
        
        float3 posMe = shPos[i];
        float massMe = shMass[i];
		float dx, dy, dz, d, d2, forceMag, inverseD;
        
        // Full loop now instead of the CPU's j = i+1 triangular loop. Thread i only
        // touches its own accumulator, so Newton's third law shortcut is given up on
        // purpose to avoid two threads writing the same F element at once.
        for(int j = 0; j < n; j++)
        {
            if(j != i)
            {
                dx = shPos[j].x - posMe.x;
                dy = shPos[j].y - posMe.y;
                dz = shPos[j].z - posMe.z;
                
                d2 = dx*dx + dy*dy + dz*dz;
                d  = sqrtf(d2);
                inverseD = 1.0f/d; // One divide, then three multiplies, instead of three divides.
                
                forceMag = (G*massMe*shMass[j])/(d2) - (H*massMe*shMass[j])/(d2*d2);
                
                force.x += forceMag*dx*inverseD;
                force.y += forceMag*dy*inverseD;
                force.z += forceMag*dz*inverseD;
            }
        }
        
        // One single global store at the end instead of 2N read-modify-writes.
        f[i] = force;
    }
}

__global__ void moveBodies(float3 *p, float3 *v, float3 *f, float *m, float damp, float dt, int firstStep, int n)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    
    if(i < n)
    {
        if(firstStep == 1)
        {
            // Half a kick to get the leap-frog started.
            v[i].x += (f[i].x/m[i])*0.5f*dt;
            v[i].y += (f[i].y/m[i])*0.5f*dt;
            v[i].z += (f[i].z/m[i])*0.5f*dt;
        }
        else
        {
            v[i].x += ((f[i].x - damp*v[i].x)/m[i])*dt;
            v[i].y += ((f[i].y - damp*v[i].y)/m[i])*dt;
            v[i].z += ((f[i].z - damp*v[i].z)/m[i])*dt;
        }

        p[i].x += v[i].x*dt;
        p[i].y += v[i].y*dt;
        p[i].z += v[i].z*dt;
    }
}

void nBody()
{
	int   drawCount = 0; 
    float time = 0.0;
    float dt = DT;
    int   firstStep = 1;

    while(time < RUN_TIME)
    {
        getForces<<<GridSize, BlockSize>>>(PGPU, FGPU, MGPU, N);
        cudaErrorCheck(__FILE__, __LINE__);
        
        moveBodies<<<GridSize, BlockSize>>>(PGPU, VGPU, FGPU, MGPU, Damp, dt, firstStep, N);
        cudaErrorCheck(__FILE__, __LINE__);
        
        firstStep = 0;

        if(drawCount == DRAW_RATE) 
        {
            if(DrawFlag)
            {
                cudaMemcpy(P, PGPU, N*sizeof(float3), cudaMemcpyDeviceToHost);
                cudaErrorCheck(__FILE__, __LINE__);
                drawPicture();
            }
            drawCount = 0;
        }
        
        time += dt;
        drawCount++;
    }
    
    cudaDeviceSynchronize();
    cudaErrorCheck(__FILE__, __LINE__);
    
    cudaMemcpy(P, PGPU, N*sizeof(float3), cudaMemcpyDeviceToHost);
    cudaErrorCheck(__FILE__, __LINE__);
}

int main(int argc, char** argv)
{
	if( argc < 3)
	{
		printf("\n You need to enter the number of bodies (an int)"); 
		printf("\n and if you want to draw the bodies as they move (1 draw, 0 don't draw),");
		printf("\n on the comand line.\n"); 
		exit(0);
	}
	else
	{
		N = atoi(argv[1]);
		DrawFlag = atoi(argv[2]);
	}

	if(N < 1 || MAX_BODIES < N)
    {
        printf("\n The number of bodies must be between 1 and %d for this one block version.\n", MAX_BODIES);
        exit(0);
    }
	
	setup();
	
	int XWindowSize = 1000;
	int YWindowSize = 1000;
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("nBody Test");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutKeyboardFunc(keyPressed);
	glutDisplayFunc(drawPicture);
	
	float3 eye = {0.0f, 0.0f, 2.0f*GlobeRadius};
	float near = 0.2;
	float far = 5.0*GlobeRadius;
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, near, far);
	glMatrixMode(GL_MODELVIEW);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluLookAt(eye.x, eye.y, eye.z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	
	glutMainLoop();
	return 0;
}





