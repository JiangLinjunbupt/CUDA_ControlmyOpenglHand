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
//共享内存的相关定义
HANDLE hMapFile;
LPCTSTR pBuf;
#define BUF_SIZE 1024
TCHAR szName[] = TEXT("Global\\MyFileMappingObject");    //指向同一块共享内存的名字
float *GetSharedMemeryPtr;

#define PI 3.1415926
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
float *d_finalMatrix;
float Handinf[27] = { 0 };
struct Pose
{
	float x;
	float y;
	float z;
	Pose(float xx, float yy, float zz) :x(xx), y(yy), z(zz) {}
	Pose() {}
};
const int NUMofJoints = 22;
//手模的世界坐标系下的关节点，通过原始的BVH文件计算得来。
const float4 HandJointsInitGlobalPosition[NUMofJoints] = {
	make_float4(0.0f,0.0f,0.0f,1), /*0----->wrist*/
	make_float4(-79.446f,5.274f,-13.494f,1), /* 1 ----> handbone*/
	make_float4(-107.056f,32.489f,-3.463f,1),/* 2 --->indexlower*/
	make_float4(-143.095f,39.646f,-3.701f,1),/* 3 --->indexmiddle*/
	make_float4(-177.646f,43.533f,-4.376f,1),/* 4 ---->indextop*/
	make_float4(-197.879f,45.443f,-6.490f,1),/* 5 --->indexsite*/
	make_float4(-112.519f,10.000f,-4.371f,1),/* 6 ---->middlelower*/
	make_float4(-154.123f,9.109f,-5.786f,1),/*7 --->middlemiddle*/
	make_float4(-187.057f,7.916f,-5.241f,1),/* 8 --->middletop*/
	make_float4(-206.370f,7.722f,-8.007f,1),/* 9 --->middlesite*/
	make_float4(-96.014f,-32.055f,-0.181f,1),/* 10 --->pinkeylower*/
	make_float4(-127.971f,-44.581f,0.423f,1),/* 11 --->pinkeymiddle*/
	make_float4(-152.207f,-54.548f,-0.005f,1),/* 12 ---->pinkeytop*/
	make_float4(-168.034f,-61.510f,-0.219f,1),/* 13---->pinkeysite*/
	make_float4(-107.198f,-12.481f,-4.071f,1),/* 14 ---->ringlower*/
	make_float4(-145.032f,-18.135f,-8.160f,1),/*15 ----->ringmiddle*/
	make_float4(-178.386f,-23.843f,-9.498f,1),/* 16 ---->ringtop*/
	make_float4(-197.783f,-26.518f,-11.079f,1),/* 17 ---->ringSite*/
	make_float4(-32.043f,47.464f,-1.782f,1),/* 18 --->thumblower*/
	make_float4(-59.707f,69.457f,5.811f,1),/* 19 ---->thumbmiddle*/
	make_float4(-89.150f,96.657f,9.811f,1),/*20 ---->thumbtop*/
	make_float4(-102.633f,114.594f,14.026f,1),/* 21 ----->thumbSite*/
};

//固定的
//矩阵引索方式
//    0   1   2   3
//    4   5   6   7 
//    8   9   10  11
//    12  13  14  15
float JointsLocalMatrix_inverse[16 * NUMofJoints] = {
	/* 0 -->wrist*/
	-0.983774f , 0.0653119f ,-0.167104f ,0,
	0.0662433f , 0.997803f  ,0          ,0,
	0.166737f  ,-0.0110695f ,-0.985939f ,0,
	0          , 0          , 0         ,1,
	/*1 --->handbone*/
	-0.954981f , 0.136453f  ,0.263423f  ,-73.0348f,
	0.141449f  , 0.989946f  ,0          ,6.01625f,
	-0.260775  , 0.0372609f ,-0.96468f  ,-33.9323f,
	0          ,0           ,0          ,1,
	/* 2 -->indexlow*/
	-0.980827f , 0.194771f  ,-0.0064709f,-111.355f,
	0.194775f  , 0.980848f  ,0          ,-11.0155f,
	0.00634699f,-0.00126037f,-0.999979f ,-2.7425f,
	0          , 0          , 0         , 1,
	/* 3 --->indexmiddle*/
	-0.993542f , 0.111789f  ,-0.019443f ,-146.676f,
	0.11181f   , 0.99373f   ,0          ,-23.3981f,
	0.0193207f ,-0.00217389f,-0.999811f ,-0.849176f,
	0          , 0          , 0         , 1,
	/*4 --->indextop*/
	-0.990234f , 0.0934805f ,-0.103434f ,-180.434f,
	0.0939846f , 0.995574f  ,0          ,-26.6451f,
	0.102976f  ,-0.00972122f,-0.994636f , 14.3632f,
	0          , 0          , 0         ,  1,
	/*5 ---->indexsite*/
	-0.990234f ,0.0934805f  ,-0.103434f ,-200.866f,
	0.0939846f ,0.995574f   , 0         ,-26.6451f,
	0.102976f  ,-0.00972122f,-0.994636f , 14.3632f,
	0          , 0          , 0         ,  1,
	/*6 ----->middlelow*/
	-0.999194  ,-0.0213804f ,-0.033981f ,-112.363f,
	-0.021393f , 0.999771f  , 0         ,-12.4048f,
	0.0339732f , 0.00072695f,-0.999422f ,-0.554011f,
	0          , 0          , 0         ,  1,
	/*7 ----->middlemiddle*/
	-0.999207f ,-0.0362107f , 0.0165399f,-153.576f,
	-0.036216f , 0.999344f  , 0         ,-14.6855f,
	-0.0165291f,-0.000599f  ,-0.999863f ,-8.3281f,
	0          , 0          , 0         , 1,
	/* 8 ----->middletop*/
	-0.989849f ,-0.00993543f,-0.141772f ,-185.823f,
	-0.010037f , 0.99995f   , 0         ,-9.79334f,
	0.141765f  , 0.00142294f,-0.989899f , 21.3182f,
	0          , 0          , 0         , 1,
	/* 9 ----->middlesite*/
	-0.989849f ,-0.00993543f,-0.141772f ,-205.334f,
	-0.0100368f, 0.99995f   , 0         ,-9.79334f,
	0.141765f  , 0.00142294f,-0.989899f , 21.3182f,
	0          , 0          , 0         ,  1,
	/* 10 ---->pinkeylow*/
	-0.930889f ,-0.364876f  , 0.0176157f,-101.072f,
	-0.364933f , 0.931034f  , 0         ,-5.19403f,
	-0.016401f ,-0.00642857f,-0.999845f ,-1.96223f,
	0          , 0          , 0         , 1,
	/*11 ----->pinkeymiddle*/
	-0.924721f ,-0.380293f  ,-0.0163675f,-135.285f,
	-0.380344f , 0.924845f  , 0         ,-7.44211f,
	0.0151374f , 0.00622528f,-0.999866f , 2.63792f,
	0          , 0          , 0         , 1,
	/* 12 --->pinkeytop*/
	-0.915294f ,-0.402597f  ,-0.0123518f,-161.276f,
	-0.402627f , 0.915364f  , 0         ,-11.351f,
	0.0113064f , 0.0049732f ,-0.999924f , 1.9865f,
	0          , 0          , 0         , 1,
	/* 13 ---->pinkeysite*/
	-0.915294f ,-0.402597f  ,-0.0123518f,-178.568f,
	-0.402627f , 0.915364f  , 0         ,-11.351f,
	0.0113064f , 0.004973f  ,-0.999924f , 1.9865f,
	0          , 0          , 0         , 1,
	/* 14 ----->ringlow*/
	-0.983417f ,-0.146944f  ,-0.10629f  ,-107.687f,
	-0.147781f , 0.98902f   , 0         ,-3.49722f,
	0.105123f  , 0.0157077f ,-0.994335f , 7.41678f,
	0          , 0          , 0         , 1,
	/* 15 ---->ringmiddle*/
	-0.9849f   ,-0.168555f  ,-0.0395108f,-146.222f,
	-0.168687f , 0.98567f   , 0         ,-6.59002f,
	0.0389446f , 0.00666495f,-0.999219f ,-2.38513f,
	0          , 0          , 0         , 1,
	/* 16 ----->ringtop*/
	-0.987408f ,-0.136191f  ,-0.0804793f,-180.152f,
	-0.136635f , 0.990622f  , 0         ,-0.754242f,
	0.0797245f , 0.0109963f ,-0.996756f , 5.01612f,
	0          , 0          , 0         , 1,
	/* 17 ---->ringsite*/
	-0.987408f ,-0.136191f  ,-0.0804793f,-199.796f,
	-0.136635f , 0.990622f  , 0         ,-0.754242f,
	0.0797245f , 0.0109963f ,-0.996756f , 5.01612f,
	0          , 0          , 0         , 1,
	/* 18 ---->thumblow*/
	-0.765308f , 0.608419f  , 0.210073f ,-53.0269f,
	0.622305f  , 0.782775f  , 0         ,-17.2129f,
	-0.164439f , 0.130729f  , -0.977686f,-13.217f,
	0          , 0          , 0         , 1,
	/* 19 ---->thumbmiddle*/
	-0.730889f , 0.675231f  , 0.099319f ,-91.1163f,
	0.678586f  , 0.734521f  , 0         ,-10.501f,
	-0.07295f  , 0.0673964f ,-0.995056f ,-3.2546f,
	0          , 0          , 0         , 1,
	/*20 ----->thumbtop*/
	-0.590548f , 0.785604f  , 0.184606f ,-130.393f,
	0.799343f  , 0.600876f  , 0         , 13.1823f,
	-0.110925f , 0.147563f  ,-0.982813f ,-14.5088f,
	0          , 0          , 0         , 1,
	/* 21 ----->thumbsite*/
	-0.590548f , 0.785604f  , 0.184606f ,-153.226f,
	0.799343f  , 0.600876f  , 0         , 13.1823f,
	-0.110925f , 0.147563f  ,-0.982813f ,-14.5088f,
	0          , 0          , 0         , 1

};

//固定的
float JointsTransMatrix[16 * NUMofJoints] = {
	/* 0 -->wrist*/
	-0.983774f  ,0.0662433f  ,0.166737f  ,  0,
	0.0653119f  ,0.997803f   ,-0.0110695f,  0,
	-0.167104f  ,0           ,-0.985939f ,  0,
	0           ,0           ,0          ,  1,
	/*1 --->handbone*/
	0.904378f   ,-0.0744983f ,0.420178f  ,  80.7569f,
	0.072892f   ,0.997141f   ,0.0199045f ,  0,
	-0.42046f   ,0.0126265f  ,0.907223f  ,  0,
	0           ,0           ,0          ,  1,
	/* 2 -->indexlow*/
	0.961544f   ,-0.0521669f ,-0.269651f ,  32.7234f,
	0.0540758f  ,0.998537f   ,-0.00035f  ,  23.0363f,
	0.269275f   ,-0.01425f   , 0.962958f ,  -1.46334f,
	0           ,0           , 0         ,   1,
	/* 3 --->indexmiddle*/
	0.996392f   ,0.0838828f  ,-0.012904f ,  36.7436f,
	-0.0838687f ,0.996475f   , 0.0016309f,  0,
	0.0129953f  ,-0.0005428f , 0.999915f ,  0,
	0           , 0          , 0         ,  1,
	/*4 --->indextop*/
	0.9963f     ,0.0179168f  ,-0.0840596f,  34.7753f,
	-0.0178241f ,0.999839f   , 0.0018536f,  0,
	0.0840793f  ,-0.0003484f , 0.996459f ,  0,
	0           , 0          , 0         ,  1,
	/*5 ---->indexsite*/
	1           , 0          , 0         ,  20.4324f,
	0           , 1          , 0         ,  0,
	0           , 0          , 1         ,  0,
	0           , 0          , 0         ,  1,
	/*6 ----->middlelow*/
	0.942342f   , 0.156851f  ,-0.295616f ,  34.6319f,
	-0.1625f    , 0.986693f  , 0.005525f ,  0,
	0.292549f   , 0.042831f  , 0.955291f ,  0,
	0           , 0          , 0         ,  1,
	/*7 ----->middlemiddle*/
	0.998614f   , 0.0148201f , 0.0505049f,  41.6381f,
	-0.0148266f , 0.99989f   ,-0.0002453f,  0,
	-0.050503f  ,-0.00050f   , 0.998724f ,  0,
	0           , 0          , 0         ,  1,
	/* 8 ----->middletop*/
	0.98708f    ,-0.02618f   ,-0.158077f ,  32.9596f,
	0.0259191f  , 0.999657f  ,-0.003712f ,  0,
	0.15812f    ,-0.0004331f , 0.98742f  ,  0,
	0           , 0          , 0         ,  1,
	/* 9 ----->middlesite*/
	1           , 0          , 0         ,  19.5109f,
	0           , 1          , 0         ,  0,
	0           , 0          , 1         ,  0,
	0           , 0          , 0         ,  1,
	/* 10 ---->pinkeylow*/
	0.843834f   ,0.475546f   ,-0.248597f ,  14.2354f,
	-0.492881f  ,0.870053f   ,-0.0086838f, -39.2982f,
	0.212163f   ,0.129856f   , 0.968568f , -9.9135f,
	0           ,0           ,0          ,  1,
	/*11 ----->pinkeymiddle*/
	0.999285f   ,0.0166041f  ,-0.0339761f,  34.3297f,
	-0.0166045f ,0.999862f   , 0.0002718f,  0,
	0.0339759f  ,0.000292539f, 0.999423f ,  0,
	0           ,0           , 0         ,  1,
	/* 12 --->pinkeytop*/
	0.999699f   ,0.0242113f  , 0.0040198f,  26.2084f,
	-0.0242127f ,0.999707f   , 0.0002991f,  0,
	-0.0040113f ,-0.00039633f, 0.999992f ,  0,
	0           , 0          , 0         ,  1,
	/* 13 ---->pinkeysite*/
	1           , 0          , 0         ,  17.292f,
	0           , 1          , 0         ,  0,
	0           , 0          , 1         ,  0,
	0           , 0          , 0         ,  1,
	/* 14 ----->ringlow*/
	0.891095f   , 0.276083f  ,-0.360178f ,  26.5616f,
	-0.28457f   , 0.958173f  , 0.0304193f,  -21.5029f,
	0.353511f   , 0.0753893  , 0.932387f ,  -2.51533f,
	0           , 0          , 0         ,  1,
	/* 15 ---->ringmiddle*/
	0.997536f   , 0.0210516f , 0.066929f ,  38.4729f,
	-0.021155f  , 0.999776f  , 0.0008365f,  0,
	-0.0668964f ,-0.00225032f, 0.997757f ,  0,
	0           , 0          , 0         ,  1,
	/* 16 ----->ringtop*/
	0.998634f   ,-0.0324031f ,-0.0409916f,  33.8647f,
	0.0323232f  , 0.999474f  ,-0.0026098f,  0,
	0.0410546f  , 0.0012813f , 0.999156f ,  0,
	0           , 0          , 0         ,  1,
	/* 17 ---->ringsite*/
	1           , 0          , 0         ,  19.6446f,
	0           , 1          , 0         ,  0,
	0           , 0          , 1         ,  0,
	0           , 0          , 0         ,  1,
	/* 18 ---->thumblow*/
	0.869213f   ,-0.487478f  ,-0.082670f ,  -36.4267f,
	0.49405f    , 0.862929f  , 0.106155f ,   48.4708f,
	0.0195901f  ,-0.133115f  , 0.990907f ,  -22.088f,
	0           , 0          , 0         ,  1,
	/* 19 ---->thumbmiddle*/
	0.991043f   ,-0.0724302f ,-0.112198f ,  36.1474f,
	0.0737171f  , 0.997252f  , 0.0073578f,  0,
	0.111357f   ,-0.0155629f , 0.993659f ,  0,
	0           , 0          , 0         ,  1,
	/*20 ----->thumbtop*/
	0.980424f   ,-0.178501f  , 0.0831013f,  40.2832f,
	0.176305f   , 0.983778f  , 0.0331161f,  0,
	-0.0876645f ,-0.0178166f , 0.995991f ,  0,
	0           , 0          , 0         ,  1,
	/* 21 ----->thumbsite*/
	1           , 0          , 0         ,  22.8323f,
	0           , 1          , 0         ,  0,
	0           , 0          , 1         ,  0,
	0           , 0          , 0         ,  1

};

//根据输入手模参数计算得到
float JointsRotationMatrix[16 * NUMofJoints] = {
	1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
	1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
	1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
	1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
	1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
	1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
	1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
	1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
	1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
	1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
	1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
	1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
	1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
	1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
	1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
	1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
	1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
	1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
	1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
	1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
	1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
	1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1
};

//根据JointsRotationMatrix计算得到
float JointsGlobalMatrix[16 * NUMofJoints] = { 0 };

float *finalMatrix;

void ComputeRotation(float *joints_rotation_matrix, Pose p, int index);
void ComputeHandRotation(float *joints_rotation_matrix, Pose p);
void ComputeJointsRotationMatrix(float *joints_rotation_matrix, float *handinf);
void MatrixProduct(float *out, int StartIndex_out, float *a, int StartIndex_a, float *b, int SatrtIndex_b);
void ComputeJointsGlobalMatrix(float *joints_global_matrix, float *joints_rotation_matrix, float *joints_trans_matrix, float *joints_localmatrix_inv);
void ComputeFinalMatrix(float *joints_global_matrix, float *weight,float *finalM);

__global__ void simple_vbo_kernel(float4 *pos, float4* vertices,int verticesNum, float *finalM)
{
	unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= verticesNum) return;

	float x = vertices[index].x;
	float y = vertices[index].y;
	float z = vertices[index].z;

	// write output vertex
	pos[index].x = finalM[index * 16 + 0] * x + finalM[index * 16 + 1] * y + finalM[index * 16 + 2] * z + finalM[index * 16 + 3];
	pos[index].y = finalM[index * 16 + 4] * x + finalM[index * 16 + 5] * y + finalM[index * 16 + 6] * z + finalM[index * 16 + 7];
	pos[index].z = finalM[index * 16 + 8] * x + finalM[index * 16 + 9] * y + finalM[index * 16 + 10] * z + finalM[index * 16 + 11];
	pos[index].w = 1;

}

void launch_kernel(float4 *pos, int verticesNum)
{
	// execute the kernel
	//dim3 block(8, 8, 1);
	//dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
	for (int i = 0; i < 24; i++)
	{
		Handinf[i] = GetSharedMemeryPtr[i];
	}
	ComputeJointsRotationMatrix(JointsRotationMatrix, Handinf);
	ComputeJointsGlobalMatrix(JointsGlobalMatrix, JointsRotationMatrix, JointsTransMatrix, JointsLocalMatrix_inverse);
	ComputeFinalMatrix(JointsGlobalMatrix, Handmodel_weights, finalMatrix);
	//cudaMemcpy(d_globalMatrix, JointsGlobalMatrix, sizeof(float) * 16 * 23, cudaMemcpyHostToDevice);
	cudaMemcpy(d_finalMatrix, finalMatrix, sizeof(float) * 16 * verticesNum, cudaMemcpyHostToDevice);
	simple_vbo_kernel << < (verticesNum+512-1)/512, 512 >> >(pos, d_vertices, verticesNum,d_finalMatrix);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
#pragma region SharedMemery
	hMapFile = CreateFileMapping(
		INVALID_HANDLE_VALUE,    // use paging file
		NULL,                    // default security
		PAGE_READWRITE,          // read/write access
		0,                       // maximum object size (high-order DWORD)
		BUF_SIZE,                // maximum object size (low-order DWORD)
		szName);                 // name of mapping object

	if (hMapFile == NULL)
	{
		_tprintf(TEXT("Could not create file mapping object (%d).\n"),
			GetLastError());
		return 1;
	}
	pBuf = (LPTSTR)MapViewOfFile(hMapFile,   // handle to map object
		FILE_MAP_ALL_ACCESS, // read/write permission
		0,
		0,
		BUF_SIZE);

	if (pBuf == NULL)
	{
		_tprintf(TEXT("Could not map view of file (%d).\n"),
			GetLastError());

		CloseHandle(hMapFile);

		return 1;
	}

	GetSharedMemeryPtr = (float*)pBuf;
#pragma endregion SharedMemery


	printf("%s starting...\n", sSDKsample);
	loadvertic(".\\model\\newVertices.txt");
	loadweight(".\\model\\newWeights.txt");
	//cudaMalloc(&d_weight, sizeof(float)*Handmodel_vertices_num *23);
	//cudaMalloc(&d_globalMatrix, sizeof(float) * 16 * 23);
	cudaMalloc(&d_vertices, sizeof(float) * 4 * Handmodel_vertices_num);
	cudaMalloc(&d_finalMatrix, sizeof(float)*Handmodel_vertices_num * 16);
	//cudaMemcpy(d_weight, Handmodel_weights, sizeof(float)*23* Handmodel_vertices_num, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vertices, Handmodel_vertices, sizeof(float)*4* Handmodel_vertices_num, cudaMemcpyHostToDevice);

	runTest(argc, argv);



	//cudaFree(d_weight);
	//cudaFree(d_globalMatrix);
	cudaFree(d_vertices);
	cudaFree(d_finalMatrix);
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
	//cudaEventRecord(start_time);

	// run CUDA kernel to generate vertex positions
	//runCuda(&cuda_vbo_resource);

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
	glPointSize(5);
	//glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
	glDrawArrays(GL_POINTS, 0, Handmodel_vertices_num);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();

	/*cudaEventRecord(stop_time);
	cudaEventSynchronize(stop_time);
	computeFPS();*/
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
	cudaEventRecord(start_time);
	runCuda(&cuda_vbo_resource);
	cudaEventRecord(stop_time);
	cudaEventSynchronize(stop_time);
	computeFPS();
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
	finalMatrix = new float[Handmodel_vertices_num * 16];
	for (int i = 0; i < Handmodel_vertices_num; i++) {
		f >> Handmodel_vertices[i].x >> Handmodel_vertices[i].y >> Handmodel_vertices[i].z;
		Handmodel_vertices[i].w = 1;          //w参数必须设置为1，不能不赋值，不然手模是不可见的（这里猜测w代表透明度吧）,
		                                      //但是在kernel中对于Opengl交互的点的w强行赋值为1，所以这里也没所谓了
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
	Handmodel_weights = new float[Handmodel_vertices_num * NUMofJoints];

	for (int i = 0; i < Handmodel_vertices_num; i++) {
		for (int j = 0; j < NUMofJoints; j++)
		{
			f >> Handmodel_weights[i * NUMofJoints + j];
		}
	}
	f.close();
	printf("Load weight succeed!!!\n");
}

void ComputeRotation(float *joints_rotation_matrix, Pose p, int index)
{
	float cx, sx;
	float cy, sy;
	float cz, sz;
	if (index == 18)
	{
		cx = cosf(-p.y / 180.0f*PI);  sx = sinf(-p.y / 180.0f*PI);
		cy = cosf(p.x / 180.0f*PI);   sy = sinf(p.x / 180.0f*PI);
		cz = cosf(p.z / 180.0f*PI);   sz = sinf(p.z / 180.0f*PI);
	}
	else
	{
		cx = cosf(p.x / 180.0f*PI);   sx = sinf(p.x / 180.0f*PI);
		cy = cosf(p.y / 180.0f*PI);   sy = sinf(p.y / 180.0f*PI);
		cz = cosf(p.z / 180.0f*PI);   sz = sinf(p.z / 180.0f*PI);
	}
	int StatPose = index * 16;
	joints_rotation_matrix[StatPose] = cy*cz;
	joints_rotation_matrix[StatPose + 1] = -cy*sz;
	joints_rotation_matrix[StatPose + 2] = sy;
	joints_rotation_matrix[StatPose + 3] = 0;
	joints_rotation_matrix[StatPose + 4] = sx*sy*cz + cx*sz;
	joints_rotation_matrix[StatPose + 5] = -sx*sy*sz + cx*cz;
	joints_rotation_matrix[StatPose + 6] = -sx*cy;
	joints_rotation_matrix[StatPose + 7] = 0;
	joints_rotation_matrix[StatPose + 8] = -cx*sy*cz + sx*sz;
	joints_rotation_matrix[StatPose + 9] = cx*sy*sz + sx*cz;
	joints_rotation_matrix[StatPose + 10] = cx*cy;
	joints_rotation_matrix[StatPose + 11] = 0;
	joints_rotation_matrix[StatPose + 12] = 0;
	joints_rotation_matrix[StatPose + 13] = 0;
	joints_rotation_matrix[StatPose + 14] = 0;
	joints_rotation_matrix[StatPose + 15] = 1;
}

void ComputeHandRotation(float *joints_rotation_matrix, Pose p)
{
	float cx, sx;
	float cy, sy;
	float cz, sz;

	float cz0, sz0;

	cx = cosf(p.x / 180.0f*PI);   sx = sinf(p.x / 180.0f*PI);
	cy = cosf(-p.y / 180.0f*PI);   sy = sinf(-p.y / 180.0f*PI);
	cz = cosf(-p.z / 180.0f*PI);   sz = sinf(-p.z / 180.0f*PI);

	cz0 = cosf(-90.0f / 180.0f*PI); sz0 = sinf(-90.0f / 180.0f*PI);

	int StatPose = 0 * 16;
	joints_rotation_matrix[StatPose] = cy*cz*cz0 - cy*sz*sz0;
	joints_rotation_matrix[StatPose + 1] = -cy*cz*sz0 - cy*sz*cz0;
	joints_rotation_matrix[StatPose + 2] = sy;
	joints_rotation_matrix[StatPose + 3] = 0;
	joints_rotation_matrix[StatPose + 4] = (sx*sy*cz + cx*sz)*cz0 + (-sx*sy*sz + cx*cz)*sz0;
	joints_rotation_matrix[StatPose + 5] = -(sx*sy*cz + cx*sz)*sz0 + (-sx*sy*sz + cx*cz)*cz0;
	joints_rotation_matrix[StatPose + 6] = -sx*cy;
	joints_rotation_matrix[StatPose + 7] = 0;
	joints_rotation_matrix[StatPose + 8] = (-cx*sy*cz + sx*sz)*cz0 + (cx*sy*sz + sx*cz)*sz0;
	joints_rotation_matrix[StatPose + 9] = -(-cx*sy*cz + sx*sz)*sz0 + (cx*sy*sz + sx*cz)*cz0;
	joints_rotation_matrix[StatPose + 10] = cx*cy;
	joints_rotation_matrix[StatPose + 11] = 0;
	joints_rotation_matrix[StatPose + 12] = 0;
	joints_rotation_matrix[StatPose + 13] = 0;
	joints_rotation_matrix[StatPose + 14] = 0;
	joints_rotation_matrix[StatPose + 15] = 1;
}

void ComputeJointsRotationMatrix(float *joints_rotation_matrix, float *handinf)
{
	Pose p_hand(handinf[15], handinf[16], handinf[17]);
	ComputeHandRotation(joints_rotation_matrix, p_hand);

	//thumb
	Pose p_thumb_lower(handinf[12], handinf[18], handinf[13]);
	Pose p_thumb_middle(0, handinf[19], 0);
	Pose p_thumb_top(0, handinf[14], 0);
	ComputeRotation(joints_rotation_matrix, p_thumb_lower, 18);
	ComputeRotation(joints_rotation_matrix, p_thumb_middle, 19);
	ComputeRotation(joints_rotation_matrix, p_thumb_top, 20);

	//pinkey
	Pose p_pinkey_lower(0, handinf[0], handinf[1]);
	Pose p_pinkey_middle(0, handinf[2], 0);
	Pose p_pinkey_top(0, handinf[20], 0);
	ComputeRotation(joints_rotation_matrix, p_pinkey_lower, 10);
	ComputeRotation(joints_rotation_matrix, p_pinkey_middle, 11);
	ComputeRotation(joints_rotation_matrix, p_pinkey_top, 12);

	//ring
	Pose p_ring_lower(0, handinf[3], handinf[4]);
	Pose p_ring_middle(0, handinf[5], 0);
	Pose p_ring_top(0, handinf[21], 0);
	ComputeRotation(joints_rotation_matrix, p_ring_lower, 14);
	ComputeRotation(joints_rotation_matrix, p_ring_middle, 15);
	ComputeRotation(joints_rotation_matrix, p_ring_top, 16);

	//middle
	Pose p_middle_lower(0, handinf[6], handinf[7]);
	Pose p_middle_middle(0, handinf[8], 0);
	Pose p_middle_top(0, handinf[22], 0);
	ComputeRotation(joints_rotation_matrix, p_middle_lower, 6);
	ComputeRotation(joints_rotation_matrix, p_middle_middle, 7);
	ComputeRotation(joints_rotation_matrix, p_middle_top, 8);

	//index
	Pose p_index_lower(0, handinf[9], handinf[10]);
	Pose p_index_middle(0, handinf[11], 0);
	Pose p_index_top(0, handinf[23], 0);
	ComputeRotation(joints_rotation_matrix, p_index_lower, 2);
	ComputeRotation(joints_rotation_matrix, p_index_middle, 3);
	ComputeRotation(joints_rotation_matrix, p_index_top, 4);

}

void MatrixProduct(float *out, int StartIndex_out, float *a, int StartIndex_a, float *b, int SatrtIndex_b)
{
	float a00 = a[StartIndex_a * 16 + 0]; float a01 = a[StartIndex_a * 16 + 1]; float a02 = a[StartIndex_a * 16 + 2]; float a03 = a[StartIndex_a * 16 + 3];
	float a10 = a[StartIndex_a * 16 + 4]; float a11 = a[StartIndex_a * 16 + 5]; float a12 = a[StartIndex_a * 16 + 6]; float a13 = a[StartIndex_a * 16 + 7];
	float a20 = a[StartIndex_a * 16 + 8]; float a21 = a[StartIndex_a * 16 + 9]; float a22 = a[StartIndex_a * 16 + 10]; float a23 = a[StartIndex_a * 16 + 11];
	float a30 = a[StartIndex_a * 16 + 12]; float a31 = a[StartIndex_a * 16 + 13]; float a32 = a[StartIndex_a * 16 + 14]; float a33 = a[StartIndex_a * 16 + 15];

	float b00 = b[SatrtIndex_b * 16 + 0]; float b01 = b[SatrtIndex_b * 16 + 1]; float b02 = b[SatrtIndex_b * 16 + 2]; float b03 = b[SatrtIndex_b * 16 + 3];
	float b10 = b[SatrtIndex_b * 16 + 4]; float b11 = b[SatrtIndex_b * 16 + 5]; float b12 = b[SatrtIndex_b * 16 + 6]; float b13 = b[SatrtIndex_b * 16 + 7];
	float b20 = b[SatrtIndex_b * 16 + 8]; float b21 = b[SatrtIndex_b * 16 + 9]; float b22 = b[SatrtIndex_b * 16 + 10]; float b23 = b[SatrtIndex_b * 16 + 11];
	float b30 = b[SatrtIndex_b * 16 + 12]; float b31 = b[SatrtIndex_b * 16 + 13]; float b32 = b[SatrtIndex_b * 16 + 14]; float b33 = b[SatrtIndex_b * 16 + 15];

	out[StartIndex_out * 16 + 0] = a00 * b00 + a01 * b10 + a02 * b20 + a03 * b30;
	out[StartIndex_out * 16 + 1] = a00 * b01 + a01 * b11 + a02 * b21 + a03 * b31;
	out[StartIndex_out * 16 + 2] = a00 * b02 + a01 * b12 + a02 * b22 + a03 * b32;
	out[StartIndex_out * 16 + 3] = a00 * b03 + a01 * b13 + a02 * b23 + a03 * b33;

	out[StartIndex_out * 16 + 4] = a10 * b00 + a11 * b10 + a12 * b20 + a13 * b30;
	out[StartIndex_out * 16 + 5] = a10 * b01 + a11 * b11 + a12 * b21 + a13 * b31;
	out[StartIndex_out * 16 + 6] = a10 * b02 + a11 * b12 + a12 * b22 + a13 * b32;
	out[StartIndex_out * 16 + 7] = a10 * b03 + a11 * b13 + a12 * b23 + a13 * b33;

	out[StartIndex_out * 16 + 8] = a20 * b00 + a21 * b10 + a22 * b20 + a23 * b30;
	out[StartIndex_out * 16 + 9] = a20 * b01 + a21 * b11 + a22 * b21 + a23 * b31;
	out[StartIndex_out * 16 + 10] = a20 * b02 + a21 * b12 + a22 * b22 + a23 * b32;
	out[StartIndex_out * 16 + 11] = a20 * b03 + a21 * b13 + a22 * b23 + a23 * b33;

	out[StartIndex_out * 16 + 12] = a30 * b00 + a31 * b10 + a32 * b20 + a33 * b30;
	out[StartIndex_out * 16 + 13] = a30 * b01 + a31 * b11 + a32 * b21 + a33 * b31;
	out[StartIndex_out * 16 + 14] = a30 * b02 + a31 * b12 + a32 * b22 + a33 * b32;
	out[StartIndex_out * 16 + 15] = a30 * b03 + a31 * b13 + a32 * b23 + a33 * b33;
}

void ComputeJointsGlobalMatrix(float *joints_global_matrix, float *joints_rotation_matrix, float *joints_trans_matrix, float *joints_localmatrix_inv)
{
	for (int i = 0; i < 16; i++)
	{
		joints_global_matrix[i] = joints_trans_matrix[i];
	}

	//handbone
	MatrixProduct(joints_global_matrix, 1, joints_global_matrix, 0, joints_trans_matrix, 1);
	MatrixProduct(joints_global_matrix, 1, joints_global_matrix, 1, joints_rotation_matrix, 1);

	//indexlower
	MatrixProduct(joints_global_matrix, 2, joints_global_matrix, 1, joints_trans_matrix, 2);
	MatrixProduct(joints_global_matrix, 2, joints_global_matrix, 2, joints_rotation_matrix, 2);
	//indexmiddle
	MatrixProduct(joints_global_matrix, 3, joints_global_matrix, 2, joints_trans_matrix, 3);
	MatrixProduct(joints_global_matrix, 3, joints_global_matrix, 3, joints_rotation_matrix, 3);
	//indextop
	MatrixProduct(joints_global_matrix, 4, joints_global_matrix, 3, joints_trans_matrix, 4);
	MatrixProduct(joints_global_matrix, 4, joints_global_matrix, 4, joints_rotation_matrix, 4);
	//indexSite
	MatrixProduct(joints_global_matrix, 5, joints_global_matrix, 4, joints_trans_matrix, 5);
	MatrixProduct(joints_global_matrix, 5, joints_global_matrix, 5, joints_rotation_matrix, 5);

	//Middlelow
	MatrixProduct(joints_global_matrix, 6, joints_global_matrix, 1, joints_trans_matrix, 6);
	MatrixProduct(joints_global_matrix, 6, joints_global_matrix, 6, joints_rotation_matrix, 6);
	//Middlemiddle
	MatrixProduct(joints_global_matrix, 7, joints_global_matrix, 6, joints_trans_matrix, 7);
	MatrixProduct(joints_global_matrix, 7, joints_global_matrix, 7, joints_rotation_matrix, 7);
	//Middletop
	MatrixProduct(joints_global_matrix, 8, joints_global_matrix, 7, joints_trans_matrix, 8);
	MatrixProduct(joints_global_matrix, 8, joints_global_matrix, 8, joints_rotation_matrix, 8);
	//Middlesite
	MatrixProduct(joints_global_matrix, 9, joints_global_matrix, 8, joints_trans_matrix, 9);
	MatrixProduct(joints_global_matrix, 9, joints_global_matrix, 9, joints_rotation_matrix, 9);

	//Pinkeylow
	MatrixProduct(joints_global_matrix, 10, joints_global_matrix, 1, joints_trans_matrix, 10);
	MatrixProduct(joints_global_matrix, 10, joints_global_matrix, 10, joints_rotation_matrix, 10);
	//PinkeyMiddle
	MatrixProduct(joints_global_matrix, 11, joints_global_matrix, 10, joints_trans_matrix, 11);
	MatrixProduct(joints_global_matrix, 11, joints_global_matrix, 11, joints_rotation_matrix, 11);
	//Pinkeytop
	MatrixProduct(joints_global_matrix, 12, joints_global_matrix, 11, joints_trans_matrix, 12);
	MatrixProduct(joints_global_matrix, 12, joints_global_matrix, 12, joints_rotation_matrix, 12);
	//PinkeySite
	MatrixProduct(joints_global_matrix, 13, joints_global_matrix, 12, joints_trans_matrix, 13);
	MatrixProduct(joints_global_matrix, 13, joints_global_matrix, 13, joints_rotation_matrix, 13);

	//Ringlow
	MatrixProduct(joints_global_matrix, 14, joints_global_matrix, 1, joints_trans_matrix, 14);
	MatrixProduct(joints_global_matrix, 14, joints_global_matrix, 14, joints_rotation_matrix, 14);
	//RingMiddle
	MatrixProduct(joints_global_matrix, 15, joints_global_matrix, 14, joints_trans_matrix, 15);
	MatrixProduct(joints_global_matrix, 15, joints_global_matrix, 15, joints_rotation_matrix, 15);
	//Ringtop
	MatrixProduct(joints_global_matrix, 16, joints_global_matrix, 15, joints_trans_matrix, 16);
	MatrixProduct(joints_global_matrix, 16, joints_global_matrix, 16, joints_rotation_matrix, 16);
	//RingSite
	MatrixProduct(joints_global_matrix, 17, joints_global_matrix, 16, joints_trans_matrix, 17);
	MatrixProduct(joints_global_matrix, 17, joints_global_matrix, 17, joints_rotation_matrix, 17);

	//Thumblow
	MatrixProduct(joints_global_matrix, 18, joints_global_matrix, 1, joints_trans_matrix, 18);
	MatrixProduct(joints_global_matrix, 18, joints_global_matrix, 18, joints_rotation_matrix, 18);
	//Thumbmiddle
	MatrixProduct(joints_global_matrix, 19, joints_global_matrix, 18, joints_trans_matrix, 19);
	MatrixProduct(joints_global_matrix, 19, joints_global_matrix, 19, joints_rotation_matrix, 19);
	//Thumbtop
	MatrixProduct(joints_global_matrix, 20, joints_global_matrix, 19, joints_trans_matrix, 20);
	MatrixProduct(joints_global_matrix, 20, joints_global_matrix, 20, joints_rotation_matrix, 20);
	//ThumbSite
	MatrixProduct(joints_global_matrix, 21, joints_global_matrix, 20, joints_trans_matrix, 21);
	MatrixProduct(joints_global_matrix, 21, joints_global_matrix, 21, joints_rotation_matrix, 21);


	for (int i = 0; i < NUMofJoints; i++)
	{
		MatrixProduct(joints_global_matrix, i, joints_rotation_matrix, 0, joints_global_matrix, i);
	}


	for (int i = 0; i < NUMofJoints; i++)
	{
		MatrixProduct(joints_global_matrix, i, joints_global_matrix, i, joints_localmatrix_inv, i);
	}
	//printf("Compute GlobalMatrix succeed!!!\n");
}

void ComputeFinalMatrix(float *joints_global_matrix, float *weight, float *finalM)
{
	for (int i = 0; i < Handmodel_vertices_num; i++)
	{
		for (int k = 0; k < NUMofJoints; k++)
		{
			if (k == 0)
			{
				for (int j = 0; j < 16; j++)
				{
					finalM[i * 16 + j] = weight[i * NUMofJoints + k] * joints_global_matrix[k * 16 + j];
				}
			}
			else
			{
				for (int j = 0; j < 16; j++)
				{
					finalM[i * 16 + j] += weight[i * NUMofJoints + k] * joints_global_matrix[k * 16 + j];
				}
			}
		}
	}
}