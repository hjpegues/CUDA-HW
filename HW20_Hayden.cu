// Name: Hayden Pegues
// Two body problem
// nvcc HW20_Hayden.cu -o HW20 -lglut -lGLU -lGL
//To stop hit "control c" in the window you launched it from.

/*
 What to do:
 This is some crude code that moves two bodies around in a box, attracted by gravity and 
 repelled when they hit each other. Take this from a two-body problem to an N-body problem, where 
 NUMBER_OF_SPHERES is a #define that you can change. Also clean it up a bit so it is more user friendly.
*/

/*
 Purpose:
 To learn about Nbody code.
*/

/*
 Explain what you did to fix the code:
 
*/

// Include files
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

// Defines
#define XWindowSize 1000
#define YWindowSize 1000
#define STOP_TIME 10000.0
#define DT        0.0001
#define GRAVITY 0.1 
#define NUMBER_OF_SPHERES 25
#define MASS 10.0  	
#define DIAMETER 1.0
#define SPHERE_PUSH_BACK_STRENGTH 50.0
#define PUSH_BACK_REDUCTION 0.1
#define DAMP 0.01
#define DRAW 100
#define LENGTH_OF_BOX 6.0
#define MAX_VELOCITY 5.0
#define MAX_PLACEMENT_ATTEMPTS 100000

// Globals
const float XMax = (LENGTH_OF_BOX/2.0);
const float YMax = (LENGTH_OF_BOX/2.0);
const float ZMax = (LENGTH_OF_BOX/2.0);
const float XMin = -(LENGTH_OF_BOX/2.0);
const float YMin = -(LENGTH_OF_BOX/2.0);
const float ZMin = -(LENGTH_OF_BOX/2.0);

float Px[NUMBER_OF_SPHERES], Py[NUMBER_OF_SPHERES], Pz[NUMBER_OF_SPHERES];
float Vx[NUMBER_OF_SPHERES], Vy[NUMBER_OF_SPHERES], Vz[NUMBER_OF_SPHERES];
float Fx[NUMBER_OF_SPHERES], Fy[NUMBER_OF_SPHERES], Fz[NUMBER_OF_SPHERES];
float Mass[NUMBER_OF_SPHERES];
float Color[NUMBER_OF_SPHERES][3];

float RunTime = 0.0;
int Finished = 0;
int Paused = 0;

// Function prototypes
float randomBetween(float, float);
void set_initail_conditions();
void Drawwirebox();
void draw_picture();
void keep_in_box();
void get_forces();
void move_bodies(float);
void nbody();
void Display(void);
void reshape(int, int);
int main(int, char**);
void idle();
void keyboard(unsigned char, int, int);

float randomBetween(float low, float high)
{
    return low + (high - low)*((float)rand()/(float)RAND_MAX);
}

void set_initail_conditions()
{ 
	int i, j, attempts, placed;
    float dx, dy, dz, separation;
    float halfSpan = (LENGTH_OF_BOX - DIAMETER)/2.0;

    for(i = 0; i < NUMBER_OF_SPHERES; i++)
    {
        // Keep drawing a candidate position until it clears every body
        // that has already been placed.
        placed = 0;
        attempts = 0;
        while(placed == 0)
        {
            Px[i] = randomBetween(-halfSpan, halfSpan);
            Py[i] = randomBetween(-halfSpan, halfSpan);
            Pz[i] = randomBetween(-halfSpan, halfSpan);

            placed = 1;
            for(j = 0; j < i; j++)
            {
                dx = Px[i] - Px[j];
                dy = Py[i] - Py[j];
                dz = Pz[i] - Pz[j];
                separation = sqrt(dx*dx + dy*dy + dz*dz);
                if(separation < DIAMETER)
                {
                    placed = 0;
                    break;
                }
            }

            attempts++;
            if(attempts > MAX_PLACEMENT_ATTEMPTS)
            {
                printf("\n Could not place body %d without overlap.\n", i);
                printf(" Lower NUMBER_OF_SPHERES or raise LENGTH_OF_BOX.\n\n");
                exit(1);
            }
        }

        Vx[i] = randomBetween(-MAX_VELOCITY, MAX_VELOCITY);
        Vy[i] = randomBetween(-MAX_VELOCITY, MAX_VELOCITY);
        Vz[i] = randomBetween(-MAX_VELOCITY, MAX_VELOCITY);

        Mass[i] = MASS;

        // Give each body a distinct color so they can be tracked by eye.
        Color[i][0] = randomBetween(0.3, 1.0);
        Color[i][1] = randomBetween(0.3, 1.0);
        Color[i][2] = randomBetween(0.3, 1.0);
    }

}

void Drawwirebox()
{		
	glColor3f (5.0,1.0,1.0);
	glBegin(GL_LINE_STRIP);
		glVertex3f(XMax,YMax,ZMax);
		glVertex3f(XMax,YMax,ZMin);	
		glVertex3f(XMax,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMax);
		glVertex3f(XMax,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		glVertex3f(XMin,YMax,ZMin);	
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMin,YMax,ZMax);	
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMax,YMin,ZMax);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMin);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMax,ZMin);
		glVertex3f(XMax,YMax,ZMin);		
	glEnd();
	
}

void draw_picture()
{
	float radius = DIAMETER/2.0;
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	Drawwirebox();

	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
    {
        glColor3f(Color[i][0], Color[i][1], Color[i][2]);
        glPushMatrix();
            glTranslatef(Px[i], Py[i], Pz[i]);
            glutSolidSphere(radius, 20, 20);
        glPopMatrix();
    }
	
	glutSwapBuffers();
}

void keep_in_box()
{
	float halfBoxLength = (LENGTH_OF_BOX - DIAMETER)/2.0;
	
	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
    {
        if(Px[i] > halfBoxLength)
        {
            Px[i] =  2.0*halfBoxLength - Px[i];
            Vx[i] = -Vx[i];
        }
        else if(Px[i] < -halfBoxLength)
        {
            Px[i] = -2.0*halfBoxLength - Px[i];
            Vx[i] = -Vx[i];
        }

        if(Py[i] > halfBoxLength)
        {
            Py[i] =  2.0*halfBoxLength - Py[i];
            Vy[i] = -Vy[i];
        }
        else if(Py[i] < -halfBoxLength)
        {
            Py[i] = -2.0*halfBoxLength - Py[i];
            Vy[i] = -Vy[i];
        }

        if(Pz[i] > halfBoxLength)
        {
            Pz[i] =  2.0*halfBoxLength - Pz[i];
            Vz[i] = -Vz[i];
        }
        else if(Pz[i] < -halfBoxLength)
        {
            Pz[i] = -2.0*halfBoxLength - Pz[i];
            Vz[i] = -Vz[i];
        }
    }
}

void get_forces()
{
	int i, j;
    float dx, dy, dz, r, r2;
    float dvx, dvy, dvz, forceMag, inout;
    float fx, fy, fz;

    // Clear the accumulators before summing pair contributions.
    for(i = 0; i < NUMBER_OF_SPHERES; i++)
    {
        Fx[i] = 0.0;
        Fy[i] = 0.0;
        Fz[i] = 0.0;
    }

    // Every unique pair is visited once. The force on j is the negative of
    // the force on i, so we get both out of one evaluation.
    for(i = 0; i < NUMBER_OF_SPHERES; i++)
    {
        for(j = i + 1; j < NUMBER_OF_SPHERES; j++)
        {
            dx = Px[j] - Px[i];
            dy = Py[j] - Py[i];
            dz = Pz[j] - Pz[i];

            r2 = dx*dx + dy*dy + dz*dz;
            r  = sqrt(r2);

            // Two bodies at the exact same point would divide by zero.
            if(r < 0.00001) continue;

            forceMag = Mass[i]*Mass[j]*GRAVITY/r2;

            if(r < DIAMETER)
            {
                dvx = Vx[j] - Vx[i];
                dvy = Vy[j] - Vy[i];
                dvz = Vz[j] - Vz[i];

                // Positive inout means they are separating already, so the
                // push back is reduced to bleed energy out of the bounce.
                inout = dx*dvx + dy*dvy + dz*dvz;
                if(inout <= 0.0)
                {
                    forceMag += SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
                }
                else
                {
                    forceMag += PUSH_BACK_REDUCTION*SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
                }
            }

            fx = forceMag*dx/r;
            fy = forceMag*dy/r;
            fz = forceMag*dz/r;

            Fx[i] += fx;
            Fy[i] += fy;
            Fz[i] += fz;

            Fx[j] -= fx;
            Fy[j] -= fy;
            Fz[j] -= fz;
        }
    }
}

void move_bodies(float time)
{
	int i;

    // Leap frog needs a half kick on the very first step to stagger the
    // velocities half a step behind the positions.
    float dtv = (time == 0.0) ? 0.5*DT : DT;

    for(i = 0; i < NUMBER_OF_SPHERES; i++)
    {
        Vx[i] += dtv*(Fx[i] - DAMP*Vx[i])/Mass[i];
        Vy[i] += dtv*(Fy[i] - DAMP*Vy[i])/Mass[i];
        Vz[i] += dtv*(Fz[i] - DAMP*Vz[i])/Mass[i];
    }

    for(i = 0; i < NUMBER_OF_SPHERES; i++)
    {
        Px[i] += DT*Vx[i];
        Py[i] += DT*Vy[i];
        Pz[i] += DT*Vz[i];
    }
	
	keep_in_box();
}

/*
void nbody()
{	
	int    tdraw = 0;
	float  time = 0.0;

	set_initail_conditions();
	
	draw_picture();
	
	while(time < STOP_TIME)
	{
		get_forces();
	
		move_bodies(time);
	
		tdraw++;
		if(tdraw == DRAW) 
		{
			draw_picture(); 
			tdraw = 0;
		}
		
		time += DT;
	}
	printf("\n DONE \n");
	while(1);
}
*/

void Display(void)
{
	draw_picture();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, 50.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0.0, 0.0, 10.0,  0.0, 0.0, 0.0,  0.0, 1.0, 0.0);
}

void idle()
{
	if(Paused || Finished) return;

    for(int step = 0; step < DRAW; step++)
    {
        get_forces();
        move_bodies(RunTime);
        RunTime += DT;
        if(RunTime >= STOP_TIME)
        {
            Finished = 1;
            printf(" Reached STOP_TIME. Press r to restart or q to quit.\n");
            break;
        }
    }
    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
    switch(key)
    {
        case 27:
            exit(0);
    }
}

int main(int argc, char** argv)
{
	time_t t;
    srand((unsigned) time(&t));

	set_initail_conditions();

	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("N Body 3D");

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

	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutIdleFunc(idle);
	
	glutMainLoop();
	return 0;
}