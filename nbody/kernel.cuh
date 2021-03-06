#pragma once
//#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <math.h>
#include <iostream>
#include <string>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Constant.h"
#include "device_functions.h"
#include <helper_cuda.h>

template<typename T> void print(T *arr, int num);
float3* cuda_caller(const glm::vec3 locations[], const glm::vec3 accs[]);
void interop_caller(cudaGraphicsResource **vbo_resource, float3 *d_speed, float3 *d_accs, float time);
float3* cuda_bin_caller(glm::vec4 bin[][BIN_SIZE], glm::vec3 locations[], glm::vec3 accs[], int3 *offset, size_t offset_len);
int to1D(int x, int y, int z, int width, int height, int depth);