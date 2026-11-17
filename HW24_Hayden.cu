// Name: Hayden Pegues
// nBody run on all available GPUs. 
// nvcc HW24_Hayden.cu -o HW24 -lglut -lm -lGLU -lGL

/*
 What to do:
 This is some robust N-body code with all the bells and whistles removed. 
 It automatically detects the number of available GPUs on the machine and runs using all of them.
 Rewrite the code using CUDA unified memory to make it simpler and use cudaMemPrefetchAsyn to make
 the memory movement faster.
*/

/*
 Purpose:
 To see how Nbody code is run on many GPUs and learn how to simplify it with unified memory then speed it up 
 with prefectched memory.
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
#define RUN_TIME 1.0

// Globals
int N;
int NPerGPU; // Amount of vector on each GPU.
int NumberOfGpus;
int ConcurrentManagedAccess; // Can the CPU read managed memory while a kernel runs?
// One managed allocation each. No host copies, no per-GPU pointer arrays.
float3 *P, *V, *F;
float  *M;
float GlobeRadius, Diameter, Radius;
float Damp;
dim3 BlockSize;
dim3 GridSize;

// Function prototypes
void cudaErrorCheck(const char *, int);
int  sliceCount(int);
void setupGpus();
void adviseMemory();
void prefetchAll();
void prefetchPositions();
void syncAllGpus();
void drawPicture();
void setup();
__global__ void getForces(float3 *, float3 *, float3 *, float *, float, float, int, int, int);
__global__ void moveBodies(float3 *, float3 *, float3 *, float *, float, float, float, int, int, int);
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

// How many bodies device "device" actually owns (the last device may own fewer, or none).
int sliceCount(int device)
{
    int start = NPerGPU*device;
    if(start >= N) return 0;
    int count = N - start;
    if(count > NPerGPU) count = NPerGPU;
    return count;
}

void setupGpus()
{
    cudaGetDeviceCount(&NumberOfGpus);
    cudaErrorCheck(__FILE__, __LINE__);

    if(NumberOfGpus == 0)
    {
        printf("\n Dude, you don't even have a GPU. Sorry, you can't play with us. Call NVIDIA and buy a GPU - loser!\n");
        exit(0);
    }
    printf("\n You will be running on %d GPU(s)\n", NumberOfGpus);

    // Every GPU has to support concurrent managed access or we skip the advise/prefetch calls.
    ConcurrentManagedAccess = 1;
    for(int i = 0; i < NumberOfGpus; i++)
    {
        int flag = 0;
        cudaDeviceGetAttribute(&flag, cudaDevAttrConcurrentManagedAccess, i);
        cudaErrorCheck(__FILE__, __LINE__);
        if(flag == 0) ConcurrentManagedAccess = 0;
    }
	printf(" Concurrent managed access: %s\n", ConcurrentManagedAccess ? "yes" : "no");

    // Turn on peer access both directions between every pair that supports it.
    for(int i = 0; i < NumberOfGpus; i++)
    {
        cudaSetDevice(i);
        cudaErrorCheck(__FILE__, __LINE__);
        for(int j = 0; j < NumberOfGpus; j++)
        {
            if(i == j) continue;
            int canAccess = 0;
            cudaDeviceCanAccessPeer(&canAccess, i, j);
            cudaErrorCheck(__FILE__, __LINE__);
            if(canAccess)
            {
                cudaDeviceEnablePeerAccess(j, 0);
                // "Already enabled" is not a real error, so just clear it and move on.
                cudaGetLastError();
            }
        }
    }
    cudaSetDevice(0);
    cudaErrorCheck(__FILE__, __LINE__);
}

// Pin the layout: every GPU keeps its own slice, and everybody (plus the CPU) gets a mapping
// to all of P because getForces needs all N positions.
void adviseMemory()
{
    if(ConcurrentManagedAccess == 0) return;

    for(int i = 0; i < NumberOfGpus; i++)
    {
        int count  = sliceCount(i);
        if(count == 0) continue;
        int offset = NPerGPU*i;

        cudaMemAdvise(&P[offset], count*sizeof(float3), cudaMemAdviseSetPreferredLocation, i);
        cudaErrorCheck(__FILE__, __LINE__);
        cudaMemAdvise(&V[offset], count*sizeof(float3), cudaMemAdviseSetPreferredLocation, i);
        cudaErrorCheck(__FILE__, __LINE__);
        cudaMemAdvise(&F[offset], count*sizeof(float3), cudaMemAdviseSetPreferredLocation, i);
        cudaErrorCheck(__FILE__, __LINE__);
        cudaMemAdvise(&M[offset], count*sizeof(float),  cudaMemAdviseSetPreferredLocation, i);
        cudaErrorCheck(__FILE__, __LINE__);
    }

    // Positions and masses are read by every GPU, so map them everywhere.
    for(int i = 0; i < NumberOfGpus; i++)
    {
        cudaMemAdvise(P, N*sizeof(float3), cudaMemAdviseSetAccessedBy, i);
        cudaErrorCheck(__FILE__, __LINE__);
        cudaMemAdvise(M, N*sizeof(float),  cudaMemAdviseSetAccessedBy, i);
        cudaErrorCheck(__FILE__, __LINE__);
    }

    // drawPicture reads P on the host; map it so those reads do not drag pages off the GPUs.
    cudaMemAdvise(P, N*sizeof(float3), cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
    cudaErrorCheck(__FILE__, __LINE__);
}

void prefetchAll()
{
    if(ConcurrentManagedAccess == 0) return;

    for(int i = 0; i < NumberOfGpus; i++)
    {
        int count  = sliceCount(i);
        if(count == 0) continue;
        int offset = NPerGPU*i;

        cudaSetDevice(i);
        cudaMemPrefetchAsync(&P[offset], count*sizeof(float3), i, 0);
        cudaErrorCheck(__FILE__, __LINE__);
        cudaMemPrefetchAsync(&V[offset], count*sizeof(float3), i, 0);
        cudaErrorCheck(__FILE__, __LINE__);
        cudaMemPrefetchAsync(&F[offset], count*sizeof(float3), i, 0);
        cudaErrorCheck(__FILE__, __LINE__);
        cudaMemPrefetchAsync(&M[offset], count*sizeof(float),  i, 0);
        cudaErrorCheck(__FILE__, __LINE__);
    }

    for(int i = 0; i < NumberOfGpus; i++)
    {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
        cudaErrorCheck(__FILE__, __LINE__);
    }
    cudaSetDevice(0);
}

void prefetchPositions()
{
    if(ConcurrentManagedAccess == 0) return;

    for(int i = 0; i < NumberOfGpus; i++)
    {
        int count = sliceCount(i);
        if(count == 0) continue;
        cudaSetDevice(i);
        cudaMemPrefetchAsync(&P[NPerGPU*i], count*sizeof(float3), i, 0);
        cudaErrorCheck(__FILE__, __LINE__);
    }
}

void syncAllGpus()
{
    for(int i = 0; i < NumberOfGpus; i++)
    {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
        cudaErrorCheck(__FILE__, __LINE__);
    }
}

void drawPicture()
{
	glClear(GL_COLOR_BUFFER_BIT);
    glClear(GL_DEPTH_BUFFER_BIT);

    // No memcpy needed. Just make sure every GPU is done writing P before we read it.
    syncAllGpus();

    glColor3d(1.0,1.0,0.5);

    for(int i=0; i<N; i++)
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

    N = 1001;

    setupGpus();

    // Ceiling divide so the slices actually cover all N bodies.
    NPerGPU = (N - 1)/NumberOfGpus + 1;

    BlockSize.x = BLOCK_SIZE;
    BlockSize.y = 1;
    BlockSize.z = 1;

    GridSize.x = (NPerGPU - 1)/BlockSize.x + 1;
    GridSize.y = 1;
    GridSize.z = 1;

    Damp = 0.5;

    // Pad up to a whole number of slices so no device ever addresses past the end.
    int nn = NumberOfGpus*NPerGPU;

    // One managed allocation per vector, shared by the CPU and every GPU.
    cudaMallocManaged(&M, nn*sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMallocManaged(&P, nn*sizeof(float3));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMallocManaged(&V, nn*sizeof(float3));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMallocManaged(&F, nn*sizeof(float3));
    cudaErrorCheck(__FILE__, __LINE__);

    Diameter = pow(H/G, 1.0/(LJQ - LJP)); // This is the value where the force is zero for the L-J type force.
    Radius = Diameter/2.0;

    // Using the radius of a body and a 68% packing ratio to find the radius of a global sphere that should hold all the bodies.
    // Then we double this radius just so we can get all the bodies setup with no problems. 
    float totalVolume = float(N)*(4.0/3.0)*PI*Radius*Radius*Radius;
    totalVolume /= 0.68;
    float totalRadius = pow(3.0*totalVolume/(4.0*PI), 1.0/3.0);
    GlobeRadius = 2.0*totalRadius;

    // Randomly setting these bodies in the global sphere and setting the initial velocity, initial force, and mass.
    // This writes straight into managed memory, so there is no host-to-device copy afterward.
    for(int i = 0; i < N; i++)
    {
        test = 0;
        while(test == 0)
        {
            randomAngle1 = ((float)rand()/(float)RAND_MAX)*2.0*PI;
            randomAngle2 = ((float)rand()/(float)RAND_MAX)*PI;
            randomRadius = ((float)rand()/(float)RAND_MAX)*GlobeRadius;
            P[i].x = randomRadius*cos(randomAngle1)*sin(randomAngle2);
            P[i].y = randomRadius*sin(randomAngle1)*sin(randomAngle2);
            P[i].z = randomRadius*cos(randomAngle2);

            // Making sure the body centers are at least a diameter apart.
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

    // Hint the layout, then push everything to the GPU that owns it before the first kernel runs.
    adviseMemory();
    prefetchAll();

    printf("\n Setup finished.\n");
}

__global__ void getForces(float3 *p, float3 *v, float3 *f, float *m, float g, float h, int nPerGPU, int n, int device)
{
	float dx, dy, dz, d, d2;
    float force_mag;

    int offset = nPerGPU*device;
    int i = threadIdx.x + blockDim.x*blockIdx.x + offset;

    if(i < n)
    {
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
    }
}

__global__ void moveBodies(float3 *p, float3 *v, float3 *f, float *m, float damp, float dt, float t, int nPerGPU, int n, int device)
{
	int offset = nPerGPU*device;	
	int i = threadIdx.x + blockDim.x*blockIdx.x + offset;
	
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
	int   drawCount = 0;
    float t = 0.0;
    float dt = DT;

    printf("\n Simulation is running with %d bodies.\n", N);

    while(t < RUN_TIME)
    {
        // Make sure each GPU has its own position slice resident before it writes to it.
        prefetchPositions();

        // Adjusting bodies. Each GPU works only on its own slice but reads all of P and M.
        for(int i = 0; i < NumberOfGpus; i++)
        {
            cudaSetDevice(i);
            getForces<<<GridSize,BlockSize>>>(P, V, F, M, G, H, NPerGPU, N, i);
            cudaErrorCheck(__FILE__, __LINE__);
            moveBodies<<<GridSize,BlockSize>>>(P, V, F, M, Damp, dt, t, NPerGPU, N, i);
            cudaErrorCheck(__FILE__, __LINE__);
        }

        // Syncing CPU with GPUs. Unified memory makes the new positions visible to every GPU,
        // so the old device-to-device copy block is no longer needed.
        syncAllGpus();

        if(drawCount == DRAW_RATE)
        {
            drawPicture();
            drawCount = 0;
        }

        t += dt;
        drawCount++;
    }

    cudaSetDevice(0);
    printf("\n Simulation finished.\n");
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
	glutCreateWindow("Nbody");
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

