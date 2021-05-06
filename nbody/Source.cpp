#include <glad/glad.h>
#include <glfw3.h>
#include <ctime>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cuda_gl_interop.h>

#include "Shader.h"
#include "Constant.h"
#include "kernel.cuh"
#include "Camera.h"

float deltaTime = 0.0;
float lastframe = 0.0;
float lastX = WIDTH / 2.0;
float lastY = HEIGHT / 2.0;
bool firstMouse = true;
Camera camera = Camera(glm::vec3(0.0f, 0.0f, 3.0f));
__device__ float3 *d_speed, *d_accs;
enum MODE
{
	CPU, CUDA, INTEROP
};

glm::vec3 bodyInteraction(glm::vec3 loc1, glm::vec3 loc2, glm::vec3 acc)
{
	float dx = loc2.x - loc1.x;
	float dy = loc2.y - loc1.y;
	float dz = loc2.z - loc1.z;
	float distSqrt = dx * dx + dy * dy + dz * dz + EPS2;
	float distCube = distSqrt * distSqrt * distSqrt;
	float invDist = G / sqrtf(distCube);
	acc.x += dx * invDist;
	acc.y += dy * invDist;
	acc.z += dz * invDist;
	acc *= SOFTEN;
	return acc;
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
	{
		glfwSetWindowShouldClose(window, true);
	}

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		camera.ProcessKeyboard(FORWARD, deltaTime);
	}

	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		camera.ProcessKeyboard(BACKWARD, deltaTime);
	}

	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		camera.ProcessKeyboard(LEFT, deltaTime);
	}

	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		camera.ProcessKeyboard(RIGHT, deltaTime);
	}
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	camera.ProcessMouseScroll(yoffset);
}

void mouse_callback(GLFWwindow* window, double xPos, double yPos)
{
	if (firstMouse)
	{
		lastX = xPos;
		lastY = yPos;
		firstMouse = false;
	}

	float xoffset = xPos - lastX;
	float yoffset = lastY - yPos;
	lastX = xPos;
	lastY = yPos;
	camera.ProcessMouseMovement(xoffset, yoffset);
}

void cpuStandard(glm::vec3 t1[], glm::vec3 t2[])
{
	for (size_t i = 0; i < NUM_OF_ENTITY; i++)
	{
		if (t1[i].x != t2[i].x || t1[i].y != t2[i].y || t1[i].z != t2[i].z)
		{
			std::cout << "t1[" << i << "]: x = " << t1[i].x << ", " << "y = " << t1[i].y << ", " << "z = " << t1[i].z << std::endl;
			std::cout << "t2[" << i << "]: x = " << t2[i].x << ", " << "y = " << t2[i].y << ", " << "z = " << t2[i].z << std::endl;
		}
	}
	std::cout << "======= END =======" << std::endl;
}

float random()
{
	return float(rand()) / float(RAND_MAX) * 2.0 - 1.0;
}

void showFPS(GLFWwindow *window, double fps)
{
	std::stringstream ss;
	ss << TITLE << " || FPS: " << fps << " || " << "NUM: " << NUM_OF_ENTITY << " || " << "BLOCK: " << TILE;
	glfwSetWindowTitle(window, ss.str().c_str());
}

void interop_mode(GLFWwindow *window, Shader ourShader, glm::mat4 view, glm::mat4 projection, unsigned int VAO, cudaGraphicsResource *cuda_vbo_resource)
{
	ourShader.use();
	float curFrame, startTime = glfwGetTime();
	unsigned int count = 0;

	while (!glfwWindowShouldClose(window))
	{
		processInput(window);
		count++;
		curFrame = glfwGetTime();
		if (curFrame - startTime >= 1.0)
		{
			double fps = count / (curFrame - startTime);
			showFPS(window, fps);
			startTime = glfwGetTime();
			count = 0;
		}
		deltaTime = curFrame - lastframe;
		lastframe = curFrame;

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		projection = glm::perspective(glm::radians(camera.Zoom), (float)(WIDTH / HEIGHT), 0.1f, 100.0f);
		view = camera.GetViewMatrix();
		ourShader.setMat4("view", glm::value_ptr(view));
		ourShader.setMat4("projection", glm::value_ptr(projection));

		glBindVertexArray(VAO);

		glDrawArrays(GL_POINTS, 0, NUM_OF_ENTITY);
		interop_caller(&cuda_vbo_resource, d_speed, d_accs, deltaTime);
		
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}

void pure_cuda_mode(GLFWwindow *window, Shader ourShader, unsigned int VAO, glm::mat4 view, glm::mat4 projection, glm::vec3 location[], glm::vec3 speed_v[], glm::vec3 accs_v[])
{
	float curFrame, startTime = glfwGetTime();
	unsigned int count = 0;
	ourShader.use();

	while (!glfwWindowShouldClose(window))
	{
		processInput(window);
		count++;
		curFrame = glfwGetTime();
		if (curFrame - startTime >= 1.0)
		{
			double fps = count / (curFrame - startTime);
			showFPS(window, fps);
			startTime = glfwGetTime();
			count = 0;
		}
		
		deltaTime = curFrame - lastframe;
		lastframe = curFrame;
		float3 *accsf3;

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		projection = glm::perspective(glm::radians(camera.Zoom), (float)(WIDTH / HEIGHT), 0.1f, 100.0f);
		view = camera.GetViewMatrix();
		ourShader.setMat4("view", glm::value_ptr(view));
		ourShader.setMat4("projection", glm::value_ptr(projection));

		glBindVertexArray(VAO);

		accsf3 = cuda_caller(location, accs_v);

		for (unsigned int i = 0; i < NUM_OF_ENTITY; i++)
		{
			ourShader.setVec3("translate", glm::value_ptr(location[i]));

			glDrawArrays(GL_POINTS, 0, 3);

			accs_v[i].x = accsf3[i].x;
			accs_v[i].y = accsf3[i].y;
			accs_v[i].z = accsf3[i].z;

			location[i] += speed_v[i] * deltaTime + accs_v[i] * (deltaTime * deltaTime / 2);
			speed_v[i] += accs_v[i] * deltaTime;
		}

		//cpuStandard(accs, accs_v);

		free(accsf3);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}

void cpu_mode(GLFWwindow *window, Shader ourShader, unsigned int VAO, glm::mat4 view, glm::mat4 projection, glm::vec3 location[], glm::vec3 speed_v[], glm::vec3 accs_v[])
{
	float curFrame, startTime = glfwGetTime();
	unsigned int count = 0;
	ourShader.use();

	while (!glfwWindowShouldClose(window))
	{
		processInput(window);
		count++;
		curFrame = glfwGetTime();
		if (curFrame - startTime >= 1.0)
		{
			double fps = count / (curFrame - startTime);
			showFPS(window, fps);
			startTime = glfwGetTime();
			count = 0;
		}
		
		deltaTime = curFrame - lastframe;
		lastframe = curFrame;
		float3 *accsf3;

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		projection = glm::perspective(glm::radians(camera.Zoom), (float)(WIDTH / HEIGHT), 0.1f, 100.0f);
		view = camera.GetViewMatrix();
		ourShader.setMat4("view", glm::value_ptr(view));
		ourShader.setMat4("projection", glm::value_ptr(projection));

		glBindVertexArray(VAO);

		for (unsigned int i = 0; i < NUM_OF_ENTITY; i++)
		{
			ourShader.setVec3("translate", glm::value_ptr(location[i]));

			glDrawArrays(GL_POINTS, 0, 3);
			
			for (unsigned int j = 0; j < NUM_OF_ENTITY; j++)
			{
				accs_v[i] = bodyInteraction(location[i], location[j], accs_v[i]);
			}

			location[i] += speed_v[i] * deltaTime + accs_v[i] * (deltaTime * deltaTime / 2);
			speed_v[i] += accs_v[i] * deltaTime;
		}

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}

/*
Input
	0: CPU mode
	1: CUDA mode
	2: INTEROP mode
*/
int main(int argc, char **argv)
{
	if (argc < 2)
	{
		std::cout << "0: CPU mode(default), 1 : CUDA mode, 2 : INTEROP mode" << std::endl;
	}
	
	unsigned int VBO, VAO;
	cudaGraphicsResource *cuda_vbo_resource;

	MODE mode = CPU;

	GLfloat vertices1[] = { 0.0f, 0.0f, 0.0f };

	GLfloat vertices2[NUM_OF_ENTITY * 3];
	float3 speed_f[NUM_OF_ENTITY], accs_f[NUM_OF_ENTITY];

	glm::vec3 location[NUM_OF_ENTITY], speed_v[NUM_OF_ENTITY], accs_v[NUM_OF_ENTITY];
	glm::vec3 accs_cpu[NUM_OF_ENTITY];

	if (argc > 1)
	{
		if (strncmp(argv[1], "0", 1) == 0)
		{
			mode = CPU;
			std::cout << "CPU MODE" << std::endl;
		}
		else if (strncmp(argv[1], "1", 1) == 0)
		{
			mode = CUDA;
			std::cout << "CUDA MODE" << std::endl;
		}
		else if (strncmp(argv[1], "2", 1) == 0)
		{
			mode = INTEROP;
			std::cout << "INTEROP MODE" << std::endl;
		}
		else
		{
			std::cout << argv[1] << std::endl;
		}
	}

	srand((unsigned int)time(NULL));

	if (mode == INTEROP)
	{
		for (size_t i = 0; i < NUM_OF_ENTITY; i++)
		{
			vertices2[i * 3] = random();
			vertices2[i * 3 + 1] = random();
			vertices2[i * 3 + 2] = random();
			speed_f[i] = make_float3(0.0f, 0.0f, 0.0f);
			accs_f[i] = make_float3(0.0f, 0.0f, 0.0f);
		}
	}
	else
	{
		for (size_t i = 0; i < NUM_OF_ENTITY; i++)
		{
			location[i] = glm::vec3(random(), random(), random());
			speed_v[i] = glm::vec3(0.0f);
			accs_v[i] = glm::vec3(0.0f);
			accs_v[i] = glm::vec3(0.0f);
		}
	}

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

	GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, TITLE, NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// set up vertex data
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	if (mode == INTEROP)
	{
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices2), vertices2, GL_DYNAMIC_DRAW);
	}
	else
	{
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices1), vertices1, GL_STATIC_DRAW);
	}

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	if (mode == INTEROP)
	{
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
	}
	else
	{
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	}
	
	glEnableVertexAttribArray(0);

	Shader ourShader("shader.vert", "shader.frag");
	ourShader.use();
	ourShader.setBool("isInterop", mode == INTEROP);
	glEnable(GL_BLEND);
	glPointSize(POINTSIZE);

	glm::mat4 view = glm::mat4(1.0f);
	glm::mat4 projection = glm::mat4(1.0f);

	if (mode == INTEROP)
	{
		cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, VBO, cudaGraphicsMapFlagsWriteDiscard);
		cudaMalloc(&d_speed, sizeof(float3) * NUM_OF_ENTITY);
		cudaMalloc(&d_accs, sizeof(float3) * NUM_OF_ENTITY);
		cudaMemcpy(d_speed, &speed_f[0], sizeof(speed_f), cudaMemcpyHostToDevice);
		cudaMemcpy(d_accs, &accs_f[0], sizeof(accs_f), cudaMemcpyHostToDevice);

		interop_mode(window, ourShader, view, projection, VAO, cuda_vbo_resource);

		cudaFree(d_accs);
		cudaFree(d_speed);
	}
	else if (mode == CUDA)
	{
		pure_cuda_mode(window, ourShader, VAO, view, projection, location, speed_v, accs_v);
	}
	else
	{
		cpu_mode(window, ourShader, VAO, view, projection, location, speed_v, accs_v);
	}

	glfwTerminate();
	
	return 0;
}