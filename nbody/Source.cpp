#include <glad/glad.h>
#include <glfw3.h>
#include <ctime>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

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

int main()
{
	unsigned int VBO, VAO;
	GLfloat vertices[] = { 0.0f, 0.0f, 0.0f };
	glm::vec3 location[NUM_OF_ENTITY], speed[NUM_OF_ENTITY], accs[NUM_OF_ENTITY];
	//glm::vec3 accs_v[NUM_OF_ENTITY];

	srand((unsigned int)time(NULL));

	for (size_t i = 0; i < NUM_OF_ENTITY; i++)
	{
		location[i] = glm::vec3(float(rand()) / float(RAND_MAX) * 2.0 - 1.0, float(rand()) / float(RAND_MAX) * 2.0 - 1.0, float(rand()) / float(RAND_MAX) * 2.0 - 1.0);
		speed[i] = glm::vec3(0.0f);
		accs[i] = glm::vec3(0.0f);
		//accs_v[i] = glm::vec3(0.0f);
	}
	

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

	GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "N-Body", NULL, NULL);
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

	Shader ourShader("shader.vert", "shader.frag");

	// set up vertex data
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(0);

	ourShader.use();
	glEnable(GL_BLEND);
	glPointSize(POINTSIZE);

	glm::mat4 view;
	glm::mat4 projection = glm::mat4(1.0f);
	float curFrame;

	while (!glfwWindowShouldClose(window))
	{
		processInput(window);
		
		curFrame = glfwGetTime();
		deltaTime = curFrame - lastframe;
		lastframe = curFrame;
		
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		glBindVertexArray(VAO);
		
		float3 *accsf3 = cuda_handler(location, accs);

		projection = glm::perspective(glm::radians(camera.Zoom), (float)(WIDTH / HEIGHT), 0.1f, 100.0f);
		view = camera.GetViewMatrix();
		ourShader.setMat4("view", glm::value_ptr(view));
		ourShader.setMat4("projection", glm::value_ptr(projection));
		
		for (unsigned int i = 0; i < NUM_OF_ENTITY; i++)
		{
			ourShader.setVec3("translate", glm::value_ptr(location[i]));
			
			glDrawArrays(GL_POINTS, 0, 3);

			/*for (unsigned int j = 0; j < NUM_OF_ENTITY; j++)
			{
				accs_v[i] = bodyInteraction(location[i], location[j], accs_v[i]);
			}*/
			accs[i].x = accsf3[i].x;
			accs[i].y = accsf3[i].y;
			accs[i].z = accsf3[i].z;
			location[i] += speed[i] * deltaTime + accs[i] * (deltaTime * deltaTime / 2);
			speed[i] += accs[i] * deltaTime;
		}
		
		//cpuStandard(accs, accs_v);
		
		free(accsf3);
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();
	return 0;
}