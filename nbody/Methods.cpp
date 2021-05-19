#include "Methods.h"

float deltaTime = 0.0;
float lastframe = 0.0;

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

bool withinCutoff(glm::vec3 n1, glm::vec3 n2, int cutoff)
{
	return (n1.x - n2.x)*(n1.x - n2.x) + (n1.y - n2.y)*(n1.y - n2.y) + (n1.z - n2.z)*(n1.z - n2.z) <= (cutoff * cutoff);
}

void assign_bin(glm::vec4 bin[][BIN_SIZE], glm::vec3 location[], int count[], std::vector<glm::vec3> &overflow)
{
	glm::vec3 entity;
	int x, y, z, idx;
	for (size_t i = 0; i < NUM_OF_ENTITY; i++)
	{
		entity = location[i];
		x = (entity.x + OFFSET) / (POS_MAX - POS_MIN) * BIN_DIVISION;
		x = (x >= BIN_DIVISION) ? BIN_DIVISION - 1 : x;

		y = (entity.y + OFFSET) / (POS_MAX - POS_MIN) * BIN_DIVISION;
		y = (y >= BIN_DIVISION) ? BIN_DIVISION - 1 : y;

		z = (entity.z + OFFSET) / (POS_MAX - POS_MIN) * BIN_DIVISION;
		z = (z >= BIN_DIVISION) ? BIN_DIVISION - 1 : z;

		int to1d = to1D(x, y, z, BIN_DIVISION, BIN_DIVISION, BIN_DIVISION);
		idx = count[to1d];

		if (idx >= BIN_SIZE)
		{
			overflow.push_back(entity);
		}
		else
		{
			bin[to1d][idx] = glm::vec4(entity, 1.0);
			count[to1d]++;
		}
	}
}

void clean_bin(glm::vec4 bin[][BIN_SIZE], int count[], std::vector<glm::vec3> &overflow)
{
	overflow.clear();
	std::fill_n(count, BIN_DIVISION*BIN_DIVISION*BIN_DIVISION, 0);

	for (size_t i = 0; i < BIN_DIVISION; i++)
	{
		for (size_t j = 0; j < BIN_DIVISION; j++)
		{
			for (size_t k = 0; k < BIN_DIVISION; k++)
			{
				int idx = to1D(i, j, k, BIN_DIVISION, BIN_DIVISION, BIN_DIVISION);
				std::fill_n(bin[idx], BIN_SIZE, glm::vec4(0.0));
			}
		}
	}
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

float random(float min, float max)
{
	return float(rand()) / float(RAND_MAX) * (max - min) + min;
}

void showFPS(GLFWwindow *window, double fps)
{
	std::stringstream ss;
	ss << TITLE << " || FPS: " << fps << " || " << "NUM: " << NUM_OF_ENTITY << " || " << "BLOCK: " << TILE;
	glfwSetWindowTitle(window, ss.str().c_str());
}

int cpu_mode(GLFWwindow* window)
{
	unsigned int VBO, VAO;
	
	glm::vec3 location[NUM_OF_ENTITY], speed[NUM_OF_ENTITY], accs[NUM_OF_ENTITY];
	glm::vec3 accs_cpu[NUM_OF_ENTITY];

	GLfloat vertices[] = { 0.0f, 0.0f, 0.0f };

	srand((unsigned int)time(NULL));

	for (size_t i = 0; i < NUM_OF_ENTITY; i++)
	{
		location[i] = glm::vec3(random(POS_MIN, POS_MAX), random(POS_MIN, POS_MAX), random(POS_MIN, POS_MAX));
		speed[i] = glm::vec3(0.0f);
		accs[i] = glm::vec3(0.0f);
	}

	// set up vertex data
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(0);

	Shader ourShader("shader.vert", "shader.frag");
	ourShader.use();
	ourShader.setBool("isInterop", false);
	glEnable(GL_BLEND);
	glPointSize(POINTSIZE);

	glm::mat4 view = glm::mat4(1.0f);
	glm::mat4 projection = glm::mat4(1.0f);

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
				accs[i] = bodyInteraction(location[i], location[j], accs[i]);
			}

			location[i] += speed[i] * deltaTime + accs[i] * (deltaTime * deltaTime / 2);
			speed[i] += accs[i] * deltaTime;
		}

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();

	return 0;
}

int cuda_mode(GLFWwindow* window)
{
	unsigned int VBO, VAO;

	glm::vec3 location[NUM_OF_ENTITY], speed[NUM_OF_ENTITY], accs[NUM_OF_ENTITY];
	glm::vec3 accs_cpu[NUM_OF_ENTITY];

	GLfloat vertices[] = { 0.0f, 0.0f, 0.0f };

	srand((unsigned int)time(NULL));

	for (size_t i = 0; i < NUM_OF_ENTITY; i++)
	{
		location[i] = glm::vec3(random(POS_MIN, POS_MAX), random(POS_MIN, POS_MAX), random(POS_MIN, POS_MAX));
		speed[i] = glm::vec3(0.0f);
		accs[i] = glm::vec3(0.0f);
	}

	// set up vertex data
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(0);

	glEnable(GL_BLEND);
	glPointSize(POINTSIZE);

	Shader ourShader("shader.vert", "shader.frag");
	ourShader.use();
	ourShader.setBool("isInterop", false);

	glm::mat4 view = glm::mat4(1.0f);
	glm::mat4 projection = glm::mat4(1.0f);

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
		float3 *accsf3;

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		projection = glm::perspective(glm::radians(camera.Zoom), (float)(WIDTH / HEIGHT), 0.1f, 100.0f);
		view = camera.GetViewMatrix();
		ourShader.setMat4("view", glm::value_ptr(view));
		ourShader.setMat4("projection", glm::value_ptr(projection));

		glBindVertexArray(VAO);

		accsf3 = cuda_caller(location, accs);

		for (unsigned int i = 0; i < NUM_OF_ENTITY; i++)
		{
			ourShader.setVec3("translate", glm::value_ptr(location[i]));

			glDrawArrays(GL_POINTS, 0, 3);

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

int interop_mode(GLFWwindow* window)
{
	unsigned int VBO, VAO;
	cudaGraphicsResource *cuda_vbo_resource;

	GLfloat location[NUM_OF_ENTITY * 3];
	float3 speed[NUM_OF_ENTITY], accs[NUM_OF_ENTITY];
	__device__ float3 *d_speed, *d_accs;

	srand((unsigned int)time(NULL));
	for (size_t i = 0; i < NUM_OF_ENTITY; i++)
	{
		location[i * 3] = random(POS_MIN, POS_MAX);
		location[i * 3 + 1] = random(POS_MIN, POS_MAX);
		location[i * 3 + 2] = random(POS_MIN, POS_MAX);
		speed[i] = make_float3(0.0f, 0.0f, 0.0f);
		accs[i] = make_float3(0.0f, 0.0f, 0.0f);
	}

	// set up vertex data
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(location), location, GL_STATIC_DRAW);

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(0);

	glEnable(GL_BLEND);
	glPointSize(POINTSIZE);

	Shader ourShader("shader.vert", "shader.frag");
	ourShader.use();
	ourShader.setBool("isInterop", true);

	glm::mat4 view = glm::mat4(1.0f);
	glm::mat4 projection = glm::mat4(1.0f);

	cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, VBO, cudaGraphicsMapFlagsWriteDiscard);
	cudaMalloc(&d_speed, sizeof(float3) * NUM_OF_ENTITY);
	cudaMalloc(&d_accs, sizeof(float3) * NUM_OF_ENTITY);
	cudaMemcpy(d_speed, &speed[0], sizeof(speed), cudaMemcpyHostToDevice);
	cudaMemcpy(d_accs, &accs[0], sizeof(accs), cudaMemcpyHostToDevice);

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

	cudaFree(d_accs);
	cudaFree(d_speed);

	glfwTerminate();

	return 0;
}

int bin_mode(GLFWwindow* window)
{
	unsigned int VBO, VAO;
	GLfloat vertices[] = { 0.0f, 0.0f, 0.0f };
	glm::vec3 location[NUM_OF_ENTITY], speed[NUM_OF_ENTITY], accs[NUM_OF_ENTITY];
	//glm::vec3 accs_cpu[NUM_OF_ENTITY];

	srand((unsigned int)time(NULL));

	for (size_t i = 0; i < NUM_OF_ENTITY; i++)
	{
		location[i] = glm::vec3(random(POS_MIN, POS_MAX), random(POS_MIN, POS_MAX), random(POS_MIN, POS_MAX));
		speed[i] = glm::vec3(0.0f);
		accs[i] = glm::vec3(0.0f);
	}

	// bin
	const double BIN_WIDTH = 2.0 / BIN_DIVISION;
	const int STEP = std::ceil((CUTOFF - BIN_WIDTH / 2.0) / BIN_WIDTH);
	const int LENGTH = STEP * 2 + 1;

	glm::vec4 bin[BIN_DIVISION * BIN_DIVISION * BIN_DIVISION][BIN_SIZE];
	int capacity[BIN_DIVISION * BIN_DIVISION * BIN_DIVISION];
	int3 *offsets = (int3 *)malloc(sizeof(int3) * LENGTH * LENGTH * LENGTH);
	std::vector<glm::vec3> overflow;

	size_t idx = 0;
	for (int i = 0 - STEP; i <= STEP; i++)
	{
		for (int j = 0 - STEP; j <= STEP; j++)
		{
			for (int k = 0 - STEP; k <= STEP; k++)
			{
				offsets[idx++] = make_int3(i, j, k);
			}
		}
	}

	// box
	unsigned int VBO1, VAO1;
	GLfloat box_vertices[] = {
		-WALL, -WALL, -WALL,    // left bottom back
		-WALL, WALL, -WALL,     // left up back

		-WALL, -WALL, -WALL,    // left bottom back
		-WALL, -WALL, WALL,     // left bottom front

		-WALL, -WALL, -WALL,    // left bottom back
		WALL, -WALL, -WALL,		// right bottom back

		WALL, -WALL, WALL,		// right bottom front
		-WALL, -WALL, WALL,     // left bottom front

		WALL, -WALL, WALL,		// right bottom front
		WALL, -WALL, -WALL,		// right bottom back

		WALL, -WALL, WALL,		// right bottom front
		WALL, WALL, WALL,		// right up front

		-WALL, WALL, WALL,		// left up front
		-WALL, WALL, -WALL,     // left up back

		-WALL, WALL, WALL,		// left up front
		-WALL, -WALL, WALL,     // left bottom front

		-WALL, WALL, WALL,		// left up front
		WALL, WALL, WALL,		// right up front

		WALL, WALL, -WALL,		// right up back
		WALL, -WALL, -WALL,		// right bottom back

		WALL, WALL, -WALL,		// right up back
		-WALL, WALL, -WALL,     // left up back

		WALL, WALL, -WALL,		// right up back
		WALL, WALL, WALL,		// right up front
	};

	// set up vertex data
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(0);

	// set up box
	glGenBuffers(1, &VBO1);
	glBindBuffer(GL_ARRAY_BUFFER, VBO1);
	glBufferData(GL_ARRAY_BUFFER, sizeof(box_vertices), box_vertices, GL_STATIC_DRAW);

	glGenVertexArrays(1, &VAO1);
	glBindVertexArray(VAO1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
	glEnableVertexAttribArray(1);
	// end setting up box

	glEnable(GL_BLEND);
	glPointSize(POINTSIZE);

	Shader ourShader("shader.vert", "shader.frag");
	ourShader.use();
	ourShader.setBool("isInterop", false);

	glm::mat4 view = glm::mat4(1.0f);
	glm::mat4 projection = glm::mat4(1.0f);

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
		float3 *accsf3;

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		projection = glm::perspective(glm::radians(camera.Zoom), (float)(WIDTH / HEIGHT), 0.1f, 100.0f);
		view = camera.GetViewMatrix();
		ourShader.setMat4("view", glm::value_ptr(view));
		ourShader.setMat4("projection", glm::value_ptr(projection));

		// draw box
		ourShader.setBool("isBox", true);
		glBindVertexArray(VAO1);
		glDrawArrays(GL_LINES, 0, 24);
		ourShader.setBool("isBox", false);

		glBindVertexArray(VAO);

		clean_bin(bin, capacity, overflow);
		assign_bin(bin, location, capacity, overflow);
		std::cout << "overflow: " << overflow.size() << std::endl;
		accsf3 = cuda_bin_caller(bin, location, accs, offsets, LENGTH * LENGTH * LENGTH);

		for (unsigned int i = 0; i < NUM_OF_ENTITY; i++)
		{
			ourShader.setVec3("translate", glm::value_ptr(location[i]));

			glDrawArrays(GL_POINTS, 0, 3);

			accs[i].x = accsf3[i].x;
			accs[i].y = accsf3[i].y;
			accs[i].z = accsf3[i].z;

			for (auto other = overflow.begin(); other != overflow.end(); other++)
			{
				glm::vec3 adj = *other;
				if (withinCutoff(location[i], adj, CUTOFF))
				{
					accs[i] = bodyInteraction(location[i], adj, accs[i]);
				}
			}

			location[i] += speed[i] * deltaTime + accs[i] * (deltaTime * deltaTime / 2);
			speed[i] += accs[i] * deltaTime;

			if (location[i].x <= 0 - WALL)
			{
				speed[i].x *= -1.0;
				accs[i].x *= -1.0;
				location[i].x = 0 - WALL + 0.01;
			}
			else if (location[i].x >= WALL)
			{
				speed[i].x *= -1.0;
				accs[i].x *= -1.0;
				location[i].x = WALL - 0.01;
			}

			if (location[i].y <= 0 - WALL)
			{
				speed[i].y *= -1.0;
				accs[i].y *= -1.0;
				location[i].y = 0 - WALL + 0.01;
			}
			else if (location[i].y >= WALL)
			{
				speed[i].y *= -1.0;
				accs[i].y *= -1.0;
				location[i].y = WALL - 0.01;
			}
			if (location[i].z <= 0 - WALL)
			{
				speed[i].z *= -1.0;
				accs[i].z *= -1.0;
				location[i].z = 0 - WALL + 0.01;
			}
			else if (location[i].z >= WALL)
			{
				speed[i].z *= -1.0;
				accs[i].z *= -1.0;
				location[i].z = WALL - 0.01;
			}
		}

		//cpuStandard(accs, accs_v);

		free(accsf3);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	free(offsets);
	glfwTerminate();

	return 0;
}