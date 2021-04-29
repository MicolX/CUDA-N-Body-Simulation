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

template<typename T> void print(T *arr, int num);
float3* cuda_handler(const glm::vec3 locations[], const glm::vec3 accs[]);