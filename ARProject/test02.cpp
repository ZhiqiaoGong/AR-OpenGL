#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include<glm-master/glm/glm.hpp>
#include<glm-master/glm/gtc/matrix_transform.hpp>
#include<glm-master/glm/gtc/type_ptr.hpp>

#include <opencv2\core.hpp>
#include <opencv2\core\opengl.hpp>
#include <opencv2\core\cuda.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\aruco.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\calib3d.hpp>

#include <iostream>
#include "shader.h"

using namespace cv;
using namespace std;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// settings
const unsigned int SCR_WIDTH = 640;
const unsigned int SCR_HEIGHT = 480;


Mat camera_matrix, distortion_coefficients;
Ptr<aruco::Dictionary> dictionary;
vector< vector<Point2f> > markerCorners, rejectedImgPoints;
vector<int> markerIds;
Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
float markerLength = 1.75;
Mat viewMatrix = Mat::zeros(4, 4, CV_32F);
Mat proMatrix = Mat::zeros(4, 4, CV_32F);
glm::mat4 modelMatrix = glm::mat4(1.0f);
//static Mat_<float> proMatrix;

bool marked = false;

//读取视频
VideoCapture cap(0);

glm::mat4 cvtoglm(cv::Mat m) {
	glm::mat4 t = glm::mat4(1.0f);
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			t[i][j] = m.at<float>(i, j);
		}
	}
	return t;
}


int main() {
	//从aruco预定义库中选择一个6x6的marker用于识别
    dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);

	//读取相机参数
	FileStorage fs("camera.yml", FileStorage::READ);
	fs["camera_matrix"] >> camera_matrix;
	fs["distortion_coefficients"] >> distortion_coefficients;


	//opengl init
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	// glfw window creation
	// --------------------
	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "myBox", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	//shader
	//用于绘制相机录制到的图像
	Shader texShader("texturev.txt", "texturef.txt");
	float tex_vertices[] = {
		// positions          // colors           // texture coords
		1.0f,  1.0f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f, 
		1.0f, -1.0f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, 
		-1.0f, -1.0f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f,
		-1.0f,  1.0f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f  
	};
	unsigned int indices[] = {
		0, 1, 3, // first triangle
		1, 2, 3  // second triangle
	};
	unsigned int TVBO, TVAO, TEBO;
	glGenVertexArrays(1, &TVAO);
	glGenBuffers(1, &TVBO);
	glGenBuffers(1, &TEBO);

	glBindVertexArray(TVAO);

	glBindBuffer(GL_ARRAY_BUFFER, TVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(tex_vertices), tex_vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, TEBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// color attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
	// texture coord attribute
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);
	//enable depth test
	glEnable(GL_DEPTH_TEST);

	//用于绘制虚拟物体
	Shader ourShader("vshader.txt", "fshader.txt");
	// set up vertex data (and buffer(s)) and configure vertex attributes
	// ------------------------------------------------------------------
	float vertices[] = {

		0.0f, 0.0f, 0.0f,   
		1.0f, 0.0f, 0.0f,   
		1.0f, 1.0f, 0.0f,   
		1.0f, 1.0f, 0.0f,   
		0.0f, 1.0f, 0.0f,   
		0.0f, 0.0f, 0.0f,   

		0.0f, 0.0f, 1.0f,   
		1.0f, 0.0f, 1.0f,   
		1.0f, 1.0f, 1.0f,   
		1.0f, 1.0f, 1.0f,   
		0.0f, 1.0f, 1.0f,   
		0.0f, 0.0f, 1.0f,   

		0.0f, 1.0f, 1.0f,   
		0.0f, 1.0f, 0.0f,   
		0.0f, 0.0f, 0.0f,   
		0.0f, 0.0f, 0.0f,   
		0.0f, 0.0f, 1.0f,   
		0.0f, 1.0f, 1.0f,   

		1.0f, 1.0f, 1.0f,   
		1.0f, 1.0f, 0.0f,   
		1.0f, 0.0f, 0.0f,   
		1.0f, 0.0f, 0.0f,   
		1.0f, 0.0f, 1.0f,   
		1.0f, 1.0f, 1.0f,   

		0.0f, 0.0f, 0.0f,   
		1.0f, 0.0f, 0.0f,   
		1.0f, 0.0f, 1.0f,   
		1.0f, 0.0f, 1.0f,   
		0.0f, 0.0f, 1.0f,   
		0.0f, 0.0f, 0.0f,   

		0.0f, 1.0f, 0.0f,   
		1.0f, 1.0f, 0.0f,   
		1.0f, 1.0f, 1.0f,   
		1.0f, 1.0f, 1.0f,   
		0.0f, 1.0f, 1.0f,   
		0.0f, 1.0f, 0.0f, 

	};
	unsigned int VBO, VAO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	//计算 projection Matrixs
	float f_x = camera_matrix.at<double>(0, 0);
	float f_y = camera_matrix.at<double>(1, 1);
	float c_x = camera_matrix.at<double>(0, 2);
	float c_y = camera_matrix.at<double>(1, 2);

	proMatrix.at<float>(0, 0) = 2 * f_x / (float)SCR_WIDTH;
	proMatrix.at<float>(1, 1) = 2 * f_y / (float)SCR_HEIGHT;
	proMatrix.at<float>(2, 0) = 1.0f - 2 * c_x / (float)SCR_WIDTH;
	proMatrix.at<float>(2, 1) = 2 * c_y / (float)SCR_HEIGHT - 1.0f;
	proMatrix.at<float>(2, 2) = -(1000.0f + 0.01f) / (1000.0f - 0.01f);
	proMatrix.at<float>(2, 3) = -1.0f;
	proMatrix.at<float>(3, 2) = -2.0f * 1000.0f * 0.01f / (1000.0f - 0.01f); //参考老师的ppt

	Mat frame;
	vector< Vec3d > rvecs, tvecs;
	Mat rotation;
	Mat gl = Mat::zeros(4, 4, CV_32F);
	gl.at<float>(0, 0) = 1.0f;
	gl.at<float>(1, 1) = -1.0f; // 注意OpenCV y和z轴与OpenGL相反
	gl.at<float>(2, 2) = -1.0f;
	gl.at<float>(3, 3) = 1.0f;

	double start = 0, end = 0;

	while (!glfwWindowShouldClose(window)) {

		processInput(window);

		cap >> frame;

		start = clock();

		//检测marker
		aruco::detectMarkers(frame, dictionary, markerCorners, markerIds, detectorParams, rejectedImgPoints);

		//检测marker位姿
		if (markerIds.size() > 0) {
			aruco::drawDetectedMarkers(frame, markerCorners, markerIds);

			aruco::estimatePoseSingleMarkers(markerCorners, markerLength, camera_matrix, distortion_coefficients, rvecs, tvecs);

			//计算view Matrix
			for (unsigned int i = 0; i < markerIds.size(); i++) {
				viewMatrix = Mat::zeros(4, 4, CV_32F);
				Rodrigues(rvecs[i], rotation); //将旋转向量转换为旋转矩阵
				for (unsigned int row = 0; row < 3; ++row)
				{
					for (unsigned int col = 0; col < 3; ++col)
					{
						viewMatrix.at<float>(row, col) = (float)rotation.at<double>(row, col);
					}
					viewMatrix.at<float>(row, 3) = (float)tvecs[i][row];
				}
				viewMatrix.at<float>(3, 3) = 1.0f; //4*4的齐次旋转矩阵 左上方3*3是旋转矩阵 右上是3*1的平移向量 左下是0 右下角是1

				viewMatrix = gl * viewMatrix; //相乘 反向
				cv::transpose(viewMatrix, viewMatrix); //矩阵转置

				//画出坐标轴
				aruco::drawAxis(frame, camera_matrix, distortion_coefficients, rvecs[i], tvecs[i], 0.5 * markerLength);
			}

			marked = true;
		}
		else {
			marked = false;
		}		

		// 将opencv读取到的图像作为texture
		unsigned int texture;
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture); 
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		cv::flip(frame, frame, 0);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.cols, frame.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, frame.data);
		glGenerateMipmap(GL_TEXTURE_2D);

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glBindTexture(GL_TEXTURE_2D, texture);
	
		texShader.use();
		glBindVertexArray(TVAO);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
		glClear(GL_DEPTH_BUFFER_BIT);

		ourShader.use();
		ourShader.setMat4("modelMatrix", modelMatrix);
		ourShader.setMat4("viewMatrix", cvtoglm(viewMatrix));
		ourShader.setMat4("proMatrix", cvtoglm(proMatrix));

		if (marked) {
			glBindVertexArray(VAO);
			glDrawArrays(GL_TRIANGLES, 0, 36);

			end = clock();
			cout << "FPS:" << int(1.0 / ((end - start) / CLOCKS_PER_SEC)) << endl;
		}	
		

		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		// -------------------------------------------------------------------------------
		glfwSwapBuffers(window);
		glfwPollEvents();
	}	

	glDeleteVertexArrays(1, &TVAO);
	glDeleteBuffers(1, &TVBO);
	glDeleteBuffers(1, &TEBO);

	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);

	// glfw: terminate, clearing all previously allocated GLFW resources.
	// ------------------------------------------------------------------
	glfwTerminate();
	return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}