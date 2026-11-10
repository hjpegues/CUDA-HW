// Name: Hayden Pegues
// Creating a n = whatever from an n <= 1024 nBody GPU code. 
// nvcc HW22_Hayden.cu -o HW22 -lglut -lm -lGLU -lGL

/*
 What to do:
 This is some lean nBody code that runs on the GPU. But the number of bodies it can simulated is limited to 1024
 so it can run on one block. Extend this code so it can simulation as many bodies as the user wants (within reason).
 Keep the same general format. But you will need to change a few major things in the code.
 Note: The code takes two arguments as inputs:
 1. The number of bodies to simulate.
 2. Whether to draw sub-arrangements of the bodies during the simulation (1), or only the first and last arrangements (0).
*/

/*
 Purpose:
 To learn how to extend an Nbody simulation from one block to many blocks.
*/

/*
 Explain what you did to fix the code:
 
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

// Threads per block. Must be <= 1024. The shared memory tiles are sized off of this.
#define BLOCK_SIZE 1024

// This is to create a Lennard-Jones type function G/(r^p) - H(r^q). (p < q) p has to be less than q.
// In this code we will keep it a p = 2 and q = 4 problem. The diameter of a body is found using the general
// case so it will be more robust but in the code leaving it as a set 2, 4 problem make the coding much easier.
#define G 10.0f
#define H 10.0f
#define LJP  2.0
#define LJQ  4.0

#define DT 0.0001
#define RUN_TIME 1.0

// Globals
int N, DrawFlag;
float3 *P, *V, *F;
float *M; 
float3 *PGPU, *VGPU, *FGPU;
float *MGPU;
float GlobeRadius, Diameter, Radius;
float Damp;
dim3 BlockSize;
dim3 GridSize;

// Function prototypes
void cudaErrorCheck(const char *, int);
void keyPressed(unsigned char, int, int);
long elaspedTime(struct timeval, struct timeval);
void drawPicture();
void timer();
void setup();
//__global__ void leapFrog(float3 *, float3 *, float3 *, float *, float, float, float, float, float, int);
__global__ void getForces(float3 *, float3 *, float *, float, float, int);
__global__ void moveBodies(float3 *, float3 *, float3 *, float *, float, float, float, int);
void nBody();
int main(int, char**);

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

void keyPressed(unsigned char key, int x, int y)
{
	if(key == 's')
	{
		printf("\n The simulation is running.\n");
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
    int tessellation;
    
    glClear(GL_COLOR_BUFFER_BIT);
    glClear(GL_DEPTH_BUFFER_BIT);
    
    cudaMemcpy(P, PGPU, N*sizeof(float3), cudaMemcpyDeviceToHost);
    cudaErrorCheck(__FILE__, __LINE__);
    
    if(N <= 1024)      tessellation = 20;
    else if(N <= 8192) tessellation = 10;
    else               tessellation = 5;
    
    glColor3d(1.0,1.0,0.5);
    for(i=0; i<N; i++)
    {
        glPushMatrix();
        glTranslatef(P[i].x, P[i].y, P[i].z);
        glutSolidSphere(Radius,tessellation,tessellation);
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
    		cudaDeviceSynchronize();
		cudaErrorCheck(__FILE__, __LINE__);
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

    	BlockSize.x = BLOCK_SIZE;
    	BlockSize.y = 1;
    	BlockSize.z = 1;

    	GridSize.x = (N - 1)/BLOCK_SIZE + 1;
    	GridSize.y = 1;
    	GridSize.z = 1;

    	Damp = 0.5;
        
        M = (float*)malloc(N*sizeof(float));
        P = (float3*)malloc(N*sizeof(float3));
        V = (float3*)malloc(N*sizeof(float3));
        F = (float3*)malloc(N*sizeof(float3));
        
        cudaMalloc(&MGPU,N*sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc(&PGPU,N*sizeof(float3));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc(&VGPU,N*sizeof(float3));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc(&FGPU,N*sizeof(float3));
    cudaErrorCheck(__FILE__, __LINE__);
        
    
    Diameter = pow(H/G, 1.0/(LJQ - LJP)); // This is the value where the force is zero for the L-J type force.
    Radius = Diameter/2.0;
    
    // Using the radius of a body and a 68% packing ratio to find the radius of a global sphere that should hold all the bodies.
    // Then we double this radius just so we can get all the bodies setup with no problems. 
    float totalVolume = float(N)*(4.0/3.0)*PI*Radius*Radius*Radius;
    totalVolume /= 0.68;
    float totalRadius = pow(3.0*totalVolume/(4.0*PI), 1.0/3.0);
    GlobeRadius = 2.0*totalRadius;
    
    // Randomly setting these bodies in the glaobal sphere and setting the initial velosity, inotial force, and mass.
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

	cudaMemcpyAsync(PGPU, P, N*sizeof(float3), cudaMemcpyHostToDevice);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMemcpyAsync(VGPU, V, N*sizeof(float3), cudaMemcpyHostToDevice);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMemcpyAsync(FGPU, F, N*sizeof(float3), cudaMemcpyHostToDevice);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMemcpyAsync(MGPU, M, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaErrorCheck(__FILE__, __LINE__);

	printf("\n To start timing type s.\n");

}

/*
__global__ void leapFrog(float3 *p, float3 *v, float3 *f, float *m, float g, float h, float damp, float dt, float t, int n)
{
	float dx, dy, dz,d,d2;
	float force_mag;
	
	int i = threadIdx.x;
	
	f[i].x = 0.0f;
	f[i].y = 0.0f;
	f[i].z = 0.0f;

	for(int j = 0; j < n; j++)
	{
		if(i != j)
		{
			dx = p[j].x-p[i].x;
			dy = p[j].y-p[i].y;
			dz = p[j].z-p[i].z;
			d2 = dx*dx + dy*dy + dz*dz;
			d  = sqrt(d2);
			
			force_mag  = (g*m[i]*m[j])/(d2) - (h*m[i]*m[j])/(d2*d2);
			f[i].x += force_mag*dx/d;
			f[i].y += force_mag*dy/d;
			f[i].z += force_mag*dz/d;
		}
	}
	__syncthreads();
	
	if(t == 0.0f)
	{
		v[i].x += ((f[i].x-damp*v[i].x)/m[i])*dt/2.0f;
		v[i].y += ((f[i].y-damp*v[i].y)/m[i])*dt/2.0f;
		v[i].z += ((f[i].z-damp*v[i].z)/m[i])*dt/2.0f;
	}
	else
	{
		v[i].x += ((f[i].x-damp*v[i].x)/m[i])*dt;
		v[i].y += ((f[i].y-damp*v[i].y)/m[i])*dt;
		v[i].z += ((f[i].z-damp*v[i].z)/m[i])*dt;
	}

	p[i].x += v[i].x*dt;
	p[i].y += v[i].y*dt;
	p[i].z += v[i].z*dt;
	__syncthreads();
}
*/

__global__ void getForces(float3 *p, float3 *f, float *m, float g, float h, int n)
{
    __shared__ float3 sharedP[BLOCK_SIZE];
    __shared__ float  sharedM[BLOCK_SIZE];

    float dx, dy, dz, d, d2;
    float force_mag;

    // Global body index for this thread.
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    // Accumulate in registers, write to global memory once at the end.
    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;

    // My own position and mass. The leftover threads (i >= n) still have to march through the
    // tile loop and hit the __syncthreads() calls, so give them harmless values instead of returning.
    float3 myP;
    float  myM;
    if(i < n)
    {
        myP = p[i];
        myM = m[i];
    }
    else
    {
        myP = make_float3(0.0f, 0.0f, 0.0f);
        myM = 0.0f;
    }
    
    for(int tile = 0; tile < gridDim.x; tile++)
    {
        int j = tile*blockDim.x + threadIdx.x; // Which body this thread loads into the tile.
        
        if(j < n)
        {
            sharedP[threadIdx.x] = p[j];
            sharedM[threadIdx.x] = m[j];
        }
        else
        {
            sharedP[threadIdx.x] = make_float3(0.0f, 0.0f, 0.0f);
            sharedM[threadIdx.x] = 0.0f; // Padding adds no force.
        }
        __syncthreads(); // Tile is loaded, safe to read.
        
        if(i < n)
        {
            for(int k = 0; k < blockDim.x; k++)
            {
                int jj = tile*blockDim.x + k; // Global index of the body in slot k.
                
                if(jj < n && jj != i)
                {
                    dx = sharedP[k].x - myP.x;
                    dy = sharedP[k].y - myP.y;
                    dz = sharedP[k].z - myP.z;
                    d2 = dx*dx + dy*dy + dz*dz;
                    d  = sqrtf(d2);
                    
                    force_mag = (g*myM*sharedM[k])/(d2) - (h*myM*sharedM[k])/(d2*d2);
                    fx += force_mag*dx/d;
                    fy += force_mag*dy/d;
                    fz += force_mag*dz/d;
                }
            }
        }
        __syncthreads(); // Everybody is done with this tile before it gets overwritten.
    }
    
    if(i < n)
    {
        f[i].x = fx;
        f[i].y = fy;
        f[i].z = fz;
    }
}

// Kernel 2: every thread leap-frogs its own body.
// This being a separate kernel launch is the grid-wide barrier that __syncthreads() could not give us.
__global__ void moveBodies(float3 *p, float3 *v, float3 *f, float *m, float damp, float dt, float t, int n)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    
    if(i < n)
    {
        if(t == 0.0f)
        {
            v[i].x += ((f[i].x-damp*v[i].x)/m[i])*dt/2.0f;
            v[i].y += ((f[i].y-damp*v[i].y)/m[i])*dt/2.0f;
            v[i].z += ((f[i].z-damp*v[i].z)/m[i])*dt/2.0f;
        }
        else
        {
            v[i].x += ((f[i].x-damp*v[i].x)/m[i])*dt;
            v[i].y += ((f[i].y-damp*v[i].y)/m[i])*dt;
            v[i].z += ((f[i].z-damp*v[i].z)/m[i])*dt;
        }

        p[i].x += v[i].x*dt;
        p[i].y += v[i].y*dt;
        p[i].z += v[i].z*dt;
    }
}

void nBody()
{
    int    drawCount = 0; 
    float  t = 0.0;
    float dt = 0.0001;

    while(t < RUN_TIME)
    {
        // Two launches per time step. All forces are finished before any body moves.
        getForces<<<GridSize,BlockSize>>>(PGPU, FGPU, MGPU, G, H, N);
        cudaErrorCheck(__FILE__, __LINE__);
        
        moveBodies<<<GridSize,BlockSize>>>(PGPU, VGPU, FGPU, MGPU, Damp, dt, t, N);
        cudaErrorCheck(__FILE__, __LINE__);
        
        if(drawCount == DRAW_RATE) 
        {
            if(DrawFlag) 
            {	
                drawPicture();
            }
            drawCount = 0;
        }
        
        t += dt;
        drawCount++;
    }
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
        
        if(N < 1)
        {
            printf("\n The number of bodies must be at least 1.\n");
            exit(0);
        }
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





