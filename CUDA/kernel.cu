// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include<fstream>
#include<iostream>
#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/freeglut.h>

// includes, cuda
#include <cuda_runtime.h>
#include<device_launch_parameters.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>

#include <tchar.h>
////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width = 800;
const unsigned int window_height = 600;

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -50.0;

cudaEvent_t start_time, stop_time;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
float avgFPS = 0.0f;
unsigned int frameCount = 0;
char fps[256];
#define MAX(a,b) ((a > b) ? a : b)


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv);
void cleanup();
// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
	unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void idle();
void reshape(int width, int height);
// Cuda functionality
void runCuda(struct cudaGraphicsResource **vbo_resource);

const char *sSDKsample = "simpleGL (VBO)";


/////////////////////////
void loadvertic(char*filename);
void loadweight(char*filename);
float *Handmodel_weights;
float4 *Handmodel_vertices;
int Handmodel_vertices_num;

float *d_weight;
float4* d_vertices;
float *d_globalMatrix;

__global__ void simple_vbo_kernel(float4 *pos, float4* vertices,int verticesNum)
{
	unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;

	if (index >= verticesNum) return;
	// write output vertex
	pos[index].x = vertices[index].x;
	pos[index].y = vertices[index].y;
	pos[index].z = vertices[index].z;
	pos[index].w = vertices[index].w;

}

void launch_kernel(float4 *pos, int verticesNum)
{
	// execute the kernel
	//dim3 block(8, 8, 1);
	//dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
	simple_vbo_kernel << < (verticesNum+512-1)/512, 512 >> >(pos, d_vertices, verticesNum);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{

	printf("%s starting...\n", sSDKsample);
	loadvertic(".\\model\\handverts.txt");
	loadweight(".\\model\\newWeight.txt");
	cudaMalloc(&d_weight, sizeof(float)*Handmodel_vertices_num *23);
	cudaMalloc(&d_globalMatrix, sizeof(float) * 16 * 23);
	cudaMalloc(&d_vertices, sizeof(float) * 4 * Handmodel_vertices_num);

	cudaMemcpy(d_weight, Handmodel_weights, sizeof(float)*23* Handmodel_vertices_num, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vertices, Handmodel_vertices, sizeof(float)*4* Handmodel_vertices_num, cudaMemcpyHostToDevice);

	runTest(argc, argv);



	cudaFree(d_weight);
	cudaFree(d_globalMatrix);
	cudaFree(d_vertices);
	return 0;
}


void computeFPS()
{
	frameCount++;
	fpsCount++;
	float time = 0;
	cudaEventElapsedTime(&time, start_time, stop_time);
	//printf("time is : %f\n", time);
	if (fpsCount == fpsLimit)
	{
		avgFPS = 1.f / (time / 1000.f);
		fpsCount = 0;
		fpsLimit = (int)MAX(avgFPS, 1.f);
	}
	sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz),%f(z)", avgFPS, translate_z);
	glutSetWindowTitle(fps);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Cuda GL Interop (VBO)");
	glewInit();                    //这个函数是必须的，不然后续的GLbindbuffer会出错；

								   // default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	//// viewport
	GLfloat fieldOfView = 120;
	glViewport(0, 0, (GLsizei)window_width, (GLsizei)window_height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fieldOfView, (GLfloat)window_width / (GLfloat)window_height, 0.1, 1000.0);

	SDK_CHECK_ERROR_GL();

	return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char **argv)
{
	// Create the CUTIL timer
	cudaEventCreate(&start_time);
	cudaEventCreate(&stop_time);

	// First initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	if (false == initGL(&argc, argv))
	{
		return false;
	}

	// register callbacks
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutIdleFunc(idle);
	glutReshapeFunc(reshape);
	glutCloseFunc(cleanup);

	// create VBO
	createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

	// run the cuda part
	//runCuda(&cuda_vbo_resource);

	// start rendering mainloop
	glutMainLoop();

	return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource)
{
	// map OpenGL buffer object for writing from CUDA
	float4 *dptr;
	checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
		*vbo_resource));
	//printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

	// execute the kernel
	launch_kernel(dptr, Handmodel_vertices_num);
	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}


////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
	unsigned int vbo_res_flags)
{
	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	unsigned int size = Handmodel_vertices_num * 4 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

	SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

	// unregister this buffer object with CUDA
	checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
	cudaEventRecord(start_time);

	// run CUDA kernel to generate vertex positions

	runCuda(&cuda_vbo_resource);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 1.0, 0.0);

	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(1.0, 0.0, 0.0);
	glPointSize(2);
	//glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
	glDrawArrays(GL_POINTS, 0, Handmodel_vertices_num);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();

	cudaEventRecord(stop_time);
	cudaEventSynchronize(stop_time);
	computeFPS();
}

void cleanup()
{
	if (vbo)
	{
		deleteVBO(&vbo, cuda_vbo_resource);
	}
}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key)
	{
	case (27):
#if defined(__APPLE__) || defined(MACOSX)
		exit(EXIT_SUCCESS);
#else
		glutDestroyWindow(glutGetWindow());
		return;
#endif
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1 << button;
	}
	else if (state == GLUT_UP)
	{
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1)
	{
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	}
	else if (mouse_buttons & 4)
	{
		translate_z += dy * 1.0f;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void idle()
{
	glutPostRedisplay();
}

void reshape(int width, int height) {

	GLfloat fieldOfView = 120;
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fieldOfView, (GLfloat)width / (GLfloat)height, 0.1, 1000.0);
}
void loadvertic(char*filename)
{
	std::ifstream f;
	f.open(filename, std::ios::in);
	//int number;
	f >> Handmodel_vertices_num;
	Handmodel_vertices = new float4[Handmodel_vertices_num];

	for (int i = 0; i < Handmodel_vertices_num; i++) {
		f >> Handmodel_vertices[i].x >> Handmodel_vertices[i].y >> Handmodel_vertices[i].z;
		Handmodel_vertices[i].x = Handmodel_vertices[i].x - 79.446518;      //这里加的是handbone相对于wrist的偏移，由于BVH文件最初是以Handbone为基准的，vertices也是以此为基准的，这里相当于对所有的点改变基准坐标
		Handmodel_vertices[i].y = Handmodel_vertices[i].y + 5.274386;
		Handmodel_vertices[i].z = Handmodel_vertices[i].z - 13.494767;
		Handmodel_vertices[i].w = 1;          //w参数必须设置为1，不能不赋值，不然手模是不可见的（这里猜测w代表透明度吧）
											  //cout<< vertices_(i,0)<<" " << vertices_(i,1)<<" "<<vertices_(i,2)<<endl;
	}
	f.close();
	printf("Load vertices succeed!!!\n");
	std::cout << "num of vertices is: " << Handmodel_vertices_num << std::endl;
}

void loadweight(char*filename)
{
	std::ifstream f;
	f.open(filename, std::ios::in);
	//int number;
	Handmodel_weights = new float[Handmodel_vertices_num * 23];

	for (int i = 0; i < Handmodel_vertices_num; i++) {
		for (int j = 0; j < 23; j++)
		{
			f >> Handmodel_weights[i * 23 + j];
		}
	}
	f.close();
	printf("Load weight succeed!!!\n");
}