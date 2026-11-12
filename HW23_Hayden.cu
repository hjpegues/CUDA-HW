// Name: Hayden Pegues
// nBody code on multiple GPUs. 
// nvcc HW23_Hayden.cu -o HW23 -lglut -lm -lGLU -lGL

/*
 What to do:
 This is some robust N-body code that runs on the GPU with all the bells and whistles removed. 
 Modify it so it runs on two GPUs. Don't use unified memory, physically copy the memory around
 to get an appreciation for unified memory and to see how you might want to do it this way for
 speed.
*/

/*
 Purpose:
 We know how to run on two GPUs to add vectors. Wow big deal! See it you can do it with something for complicated.
*/

/*
 Explain what you did to fix the code:
 Under Defines: Defined the number of GPUs
 Under Globals: Edited the GPU pointers so that they're arrays of NUM_GPU size, also added two more variables
 Under Function Prototypes: Defined three new function prototypes
 Inside setUpDevices: This makes sure we have enough devices, tells the GPUs how much they're doing, and calculates block and grid size
 Inside copyPositionsToHost: Makes the normal position array using the slices from the GPUs
 Inside exchangePositions: Makes the GPUs see the updated positions of every body
 Inside drawPicture: Replaces the cudaMemcpy with copyPositionToHost
 Inside Setup: Calls setupDevices instead of calculating block and grid size itself
 Inside getForces and moveBodies: If statements go through offset + n now
 Inside nBody: Iterate through both GPUs when calculating forces and position
 
*/

// Include files
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Defines
#define BLOCK_SIZE 128
#define PI 3.14159265359
#define DRAW_RATE 10

// This is to create a Lennard-Jones type function G/(r^p) - H(r^q). (p < q) p has to be less than q.
// In this code we will keep it a p = 2 and q = 4 problem. The diameter of a body is found using the general
// case so it will be more robust but in the code leaving it as a set 2, 4 problem make the coding much easier.
#define G 10.0f
#define H 10.0f
#define LJP  2.0
#define LJQ  4.0

#define DT 0.0001
#define RUN_TIME 10.0

#define NUM_GPU 2

// Globals
int N;
float3 *P, *V, *F;
float *M; 
float3 *PGPU[NUM_GPU], *VGPU[NUM_GPU], *FGPU[NUM_GPU];
float *MGPU[NUM_GPU];
int NGPU[NUM_GPU]; //Bodies this GPU has
int Offset[NUM_GPU];  //Index of the first body
float GlobeRadius, Diameter, Radius;
float Damp;
dim3 BlockSize[NUM_GPU];
dim3 GridSize[NUM_GPU];

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void copyPositionsToHost();
void exchangePositions();
void drawPicture();
void setup();
__global__ void getForces(float3 *, float3 *, float3 *, float *, float, float, int, int, int);
__global__ void moveBodies(float3 *, float3 *, float3 *, float *, float, float, float, int, int);
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

void setUpDevices()
{
    int gpuCount;
    cudaGetDeviceCount(&gpuCount);
    cudaErrorCheck(__FILE__, __LINE__);
    
    printf("\n You have %d GPUs available.\n", gpuCount);
    if(gpuCount < NUM_GPU)
    {
        printf("\n This code needs %d GPUs and you have %d. I'd say buy more to boost Jensens' wifes' alimony check.\n", NUM_GPU, gpuCount);
        exit(0);
    }
   
	cudaDeviceProp prop;

	for(int i = 0; i < gpuCount; i++)
	{
		cudaGetDeviceProperties(&prop, i);
		printf("GPU %d: %s\n", i, prop.name);	
	}

    NGPU[0] = (N + 1)/2;
    NGPU[1] = N - NGPU[0];
    Offset[0] = 0;
    Offset[1] = NGPU[0];
    
    for(int dev = 0; dev < NUM_GPU; dev++)
    {
        BlockSize[dev].x = BLOCK_SIZE;
        BlockSize[dev].y = 1;
        BlockSize[dev].z = 1;
        
        GridSize[dev].x = (NGPU[dev] - 1)/BlockSize[dev].x + 1;
        GridSize[dev].y = 1;
        GridSize[dev].z = 1;
    }
}

void copyPositionsToHost()
{
    for(int dev = 0; dev < NUM_GPU; dev++)
    {
        cudaSetDevice(dev);
        cudaErrorCheck(__FILE__, __LINE__);
        cudaMemcpy(P + Offset[dev], PGPU[dev] + Offset[dev], NGPU[dev]*sizeof(float3), cudaMemcpyDeviceToHost);
        cudaErrorCheck(__FILE__, __LINE__);
    }
}

void exchangePositions()
{
    // Barrier on BOTH cards first, or we copy halves that are still being written.
    for(int dev = 0; dev < NUM_GPU; dev++)
    {
        cudaSetDevice(dev);
        cudaDeviceSynchronize();
        cudaErrorCheck(__FILE__, __LINE__);
    }
    
    // Owned slices down to the host.
    copyPositionsToHost();
    
    // Whole array back up to both cards. Only positions move, nothing else needs to.
    for(int dev = 0; dev < NUM_GPU; dev++)
    {
        cudaSetDevice(dev);
        cudaErrorCheck(__FILE__, __LINE__);
        cudaMemcpy(PGPU[dev], P, N*sizeof(float3), cudaMemcpyHostToDevice);
        cudaErrorCheck(__FILE__, __LINE__);
    }
}

void drawPicture()
{
	int i;
	
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	copyPositionsToHost();
	
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

void setup()
{
    	float randomAngle1, randomAngle2, randomRadius;
    	float d, dx, dy, dz;
    	int test;
    	
    	N = 1000;
	
		setUpDevices();

    	Damp = 0.5;
    	
    	M = (float*)malloc(N*sizeof(float));
    	P = (float3*)malloc(N*sizeof(float3));
    	V = (float3*)malloc(N*sizeof(float3));
    	F = (float3*)malloc(N*sizeof(float3));
    	
    	for(int dev = 0; dev < NUM_GPU; dev++)
        {
            cudaSetDevice(dev);
            cudaErrorCheck(__FILE__, __LINE__);
            
            cudaMalloc(&MGPU[dev], N*sizeof(float));
            cudaErrorCheck(__FILE__, __LINE__);
            cudaMalloc(&PGPU[dev], N*sizeof(float3));
            cudaErrorCheck(__FILE__, __LINE__);
            cudaMalloc(&VGPU[dev], N*sizeof(float3));
            cudaErrorCheck(__FILE__, __LINE__);
            cudaMalloc(&FGPU[dev], N*sizeof(float3));
            cudaErrorCheck(__FILE__, __LINE__);
        }
    	
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
	
	for(int dev = 0; dev < NUM_GPU; dev++)
    {
        cudaSetDevice(dev);
        cudaErrorCheck(__FILE__, __LINE__);
        
        cudaMemcpy(PGPU[dev], P, N*sizeof(float3), cudaMemcpyHostToDevice);
        cudaErrorCheck(__FILE__, __LINE__);
        cudaMemcpy(VGPU[dev], V, N*sizeof(float3), cudaMemcpyHostToDevice);
        cudaErrorCheck(__FILE__, __LINE__);
        cudaMemcpy(FGPU[dev], F, N*sizeof(float3), cudaMemcpyHostToDevice);
        cudaErrorCheck(__FILE__, __LINE__);
        cudaMemcpy(MGPU[dev], M, N*sizeof(float), cudaMemcpyHostToDevice);
        cudaErrorCheck(__FILE__, __LINE__);
    }
}

__global__ void getForces(float3 *p, float3 *v, float3 *f, float *m, float g, float h, int n, int offset, int nTotal)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	
	if(i < n)
	{
		float dx, dy, dz, d, d2;
    float force_mag;
    
    int i = threadIdx.x + blockDim.x*blockIdx.x + offset;
    
    if(i < offset + n)
    {
        f[i].x = 0.0f;
        f[i].y = 0.0f;
        f[i].z = 0.0f;
        
        for(int j = 0; j < nTotal; j++)
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
    }
}

}

__global__ void moveBodies(float3 *p, float3 *v, float3 *f, float *m, float damp, float dt, float t, int n, int offset)
{	
	int i = threadIdx.x + blockDim.x*blockIdx.x + offset;
    
    if(i < offset + n)
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
    
    timeval start, end;
    gettimeofday(&start, NULL);

    while(t < RUN_TIME)
    {
        // Launch both cards. These are async, so GPU0 and GPU1 work at the same time.
        for(int dev = 0; dev < NUM_GPU; dev++)
        {
            cudaSetDevice(dev);
            getForces<<<GridSize[dev],BlockSize[dev]>>>(PGPU[dev], VGPU[dev], FGPU[dev], MGPU[dev], G, H, NGPU[dev], Offset[dev], N);
            cudaErrorCheck(__FILE__, __LINE__);
        }
        
        for(int dev = 0; dev < NUM_GPU; dev++)
        {
            cudaSetDevice(dev);
            moveBodies<<<GridSize[dev],BlockSize[dev]>>>(PGPU[dev], VGPU[dev], FGPU[dev], MGPU[dev], Damp, dt, t, NGPU[dev], Offset[dev]);
            cudaErrorCheck(__FILE__, __LINE__);
        }
        
        // Both cards must see all N new positions before the next getForces.
        exchangePositions();
        
        if(drawCount == DRAW_RATE) 
        {	
            drawPicture();
            drawCount = 0;
        }
        
        t += dt;
        drawCount++;
    }
    
    gettimeofday(&end, NULL);
    float time = (end.tv_sec*1000000.0 + end.tv_usec) - (start.tv_sec*1000000.0 + start.tv_usec);
    printf("\n Run time on %d GPUs = %.3f milliseconds.\n", NUM_GPU, time/1000.0);
}

int main(int argc, char** argv)
{
	setup();
    
    int XWindowSize = 1000;
    int YWindowSize = 1000;
    
    glutInit(&argc,argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
    glutInitWindowSize(XWindowSize,YWindowSize);
    glutInitWindowPosition(0,0);
    glutCreateWindow("Nbody Two GPUs");
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
    glutDisplayFunc(drawPicture);
    glutIdleFunc(nBody);
    
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

