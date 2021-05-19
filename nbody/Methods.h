#pragma once
#ifndef METHODS_H
#define METHODS_H

#include <glad/glad.h>
#include <glfw3.h>
#include <ctime>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cuda_gl_interop.h>
#include <vector>

#include "Shader.h"
#include "Constant.h"
#include "kernel.cuh"
#include "Camera.h"


int cpu_mode(GLFWwindow* window);
int cuda_mode(GLFWwindow* window);
int interop_mode(GLFWwindow* window);
int bin_mode(GLFWwindow* window);

extern Camera camera;

#endif // !METHODS_H