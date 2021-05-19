#include "kernel.cuh"

__device__ 
float3 calculateAcceleration(float3 loc1, float3 loc2, float3 acc)
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
	acc.x *= SOFTEN;
	acc.y *= SOFTEN;
	acc.z *= SOFTEN;
	return acc;
}

__device__
float3 updateLocation(float3 loc, float time, float3 &acc, float3 &speed)
{
	loc.x += speed.x * time + acc.x * (time * time / 2);
	loc.y += speed.y * time + acc.y * (time * time / 2);
	loc.z += speed.z * time + acc.z * (time * time / 2);
	return loc;
}

__device__
float3 updateSpeed(float3 speed, float3 acc, float time)
{
	speed.x += acc.x * time;
	speed.y += acc.y * time;
	speed.z += acc.z * time;
	return speed;
}

__host__
__device__
int to1D(int x, int y, int z, int width, int height, int depth)
{
	return x + width * (y + depth * z);
}


__global__ 
void share_kernel(float3 *entities, float3 *accs, int length)
{
	__shared__ float3 share[TILE];
	unsigned int tx = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + tx;
	float3 acc;
	float3 entity;

	if (idx < length)
	{
		entity = entities[idx];
		acc = accs[idx];
	
		for (size_t i = 0; i < length; i += TILE)
		{
			if (i + tx < length) 
			{
				share[tx] = entities[i + tx];
			}
			 
			__syncthreads();

			for (size_t j = 0; j < blockDim.x; j++)
			{
				if (i + j < length)
				{
					acc = calculateAcceleration(entity, share[j], acc);
				}
			}
			__syncthreads();
		}

		accs[idx] = acc;
	}
}

__global__ 
void naive_kernel(float3 *entities, float3 *accs, int length)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float3 entity, acc;
	if (idx < length)
	{
		entity = entities[idx];
		acc = accs[idx];


		for (size_t i = 0; i < length; i++)
		{
			acc = calculateAcceleration(entity, entities[i], acc);
		}
		accs[idx] = acc;
	}
}

__global__
void naive_op_kernel(float3 *pos, float3 *d_speed, float3 *d_accs, float time, int num)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num)
	{
		float3 entity = pos[idx];
		float3 speed = d_speed[idx];
		float3 acc = d_accs[idx];

		for (size_t i = 0; i < num; i++)
		{
			acc = calculateAcceleration(entity, pos[i], acc);
		}

		d_speed[idx] = updateSpeed(speed, acc, time);
		pos[idx] = updateLocation(entity, time, acc, speed);
		d_accs[idx] = acc;
	}
}

__global__
void share_op_kernel(float3 *pos, float3 *d_speed, float3 *d_accs, float time, int num)
{
	__shared__ float3 share_pos[TILE];
	unsigned int thx = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + thx;
	float3 entity = pos[idx];
	float3 &speed = d_speed[idx];
	float3 &acc = d_accs[idx];

	if (idx < num)
	{
		for (size_t i = 0; i < num; i += TILE)
		{
			if (i + thx < num)
			{
				share_pos[thx] = pos[i + thx];
			}
			
			__syncthreads();

			for (size_t j = 0; j < TILE; j++)
			{
				if (i + j < num)
				{
					acc = calculateAcceleration(entity, share_pos[j], acc);
				}
			}

			__syncthreads();
		}

		pos[idx] = updateLocation(entity, time, acc, speed);
		d_accs[idx] = acc;
		d_speed[idx] = updateSpeed(speed, acc, time);
	}
}

__global__
void bin_kernel(float3 *pos, float3 *accs, float4 *bin, int3 *offsets, size_t offset_len)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float3 entity = pos[idx];
	float3 acc = accs[idx];
	int x, y, z;

	x = (entity.x + OFFSET) / (POS_MAX - POS_MIN) * BIN_DIVISION;
	x = (x >= BIN_DIVISION) ? BIN_DIVISION - 1 : x;
	y = (entity.y + OFFSET) / (POS_MAX - POS_MIN) * BIN_DIVISION;
	y = (y >= BIN_DIVISION) ? BIN_DIVISION - 1 : y;
	z = (entity.z + OFFSET) / (POS_MAX - POS_MIN) * BIN_DIVISION;
	z = (z >= BIN_DIVISION) ? BIN_DIVISION - 1 : z;
	
	// go through offset list
	for (size_t i = 0; i < offset_len; i++)
	{
		int x_off = offsets[i].x;
		int y_off = offsets[i].y;
		int z_off = offsets[i].z;
		if (x+x_off < 0 || x+x_off >= BIN_DIVISION || y + y_off < 0 || y + y_off >= BIN_DIVISION || z + z_off < 0 || z + z_off >= BIN_DIVISION)
		{
			continue;
		}

		int index = to1D(x+x_off, y+y_off, z+z_off, BIN_DIVISION, BIN_DIVISION, BIN_DIVISION);

		// go through bin
		for (size_t j = 0; j < BIN_SIZE; j++)
		{
			float4 other = bin[index + j];
			if (other.w == 0)
			{
				break;
			}
			acc = calculateAcceleration(entity, make_float3(other.x, other.y, other.z), acc);
		}
	}
	accs[idx] = acc;
}

template<typename T>
void print(T* arr, int num)
{
	std::cout << "===== Start printing =====" << std::endl;
	for (size_t i = 0; i < num; i++)
	{
		std::cout << arr[i].x << ", " << arr[i].y << ", " << arr[i].z << std::endl;
	}
	std::cout << "===== End printing =====" << std::endl;
}


float3* cuda_caller(const glm::vec3 locations[], const glm::vec3 accs[])
{
	size_t bytes = sizeof(float3);
	float3 *d_locations, *d_accs, *h_locations, *h_accs;

	h_locations = (float3 *)malloc(bytes * NUM_OF_ENTITY);
	h_accs = (float3 *)malloc(bytes * NUM_OF_ENTITY);

	for (size_t i = 0; i < NUM_OF_ENTITY; i++)
	{
		h_locations[i] = make_float3(locations[i].x, locations[i].y, locations[i].z);
		h_accs[i] = make_float3(accs[i].x, accs[i].y, accs[i].z);
	}
	

	cudaMalloc(&d_locations, bytes * NUM_OF_ENTITY);
	cudaMalloc(&d_accs, bytes * NUM_OF_ENTITY);
	cudaMemcpy(d_locations, h_locations, bytes * NUM_OF_ENTITY, cudaMemcpyHostToDevice);
	cudaMemcpy(d_accs, h_accs, bytes * NUM_OF_ENTITY, cudaMemcpyHostToDevice);

	dim3 dimBlock(TILE);
	dim3 dimGrid(ceil((double)NUM_OF_ENTITY / (double)TILE));
	
	naive_kernel <<< dimGrid, dimBlock >>> (d_locations, d_accs, NUM_OF_ENTITY);

	cudaMemcpy(h_accs, d_accs, bytes * NUM_OF_ENTITY, cudaMemcpyDeviceToHost);
	cudaFree(d_locations);
	cudaFree(d_accs);
	free(h_locations);

	return h_accs;
}


float3* cuda_bin_caller(glm::vec4 bin[][BIN_SIZE], glm::vec3 locations[], glm::vec3 accs[], int3 *offset, size_t offset_len)
{
	size_t bytes = sizeof(float3);
	float3 *d_locations, *d_accs, *h_locations, *h_accs;
	float4 *d_bin, *h_bin;
	int3 *d_offsets;

	h_locations = (float3 *)malloc(bytes * NUM_OF_ENTITY);
	h_accs = (float3 *)malloc(bytes * NUM_OF_ENTITY);
	h_bin = (float4 *)malloc(sizeof(float4) * BIN_DIVISION * BIN_DIVISION * BIN_DIVISION * BIN_SIZE);
	

	for (size_t i = 0; i < NUM_OF_ENTITY; i++)
	{
		h_locations[i] = make_float3(locations[i].x, locations[i].y, locations[i].z);
		h_accs[i] = make_float3(accs[i].x, accs[i].y, accs[i].z);
	}

	for (size_t i = 0; i < BIN_DIVISION; i++)
	{
		for (size_t j = 0; j < BIN_DIVISION; j++)
		{
			for (size_t k = 0; k < BIN_DIVISION; k++)
			{
				int idx = to1D(i, j, k, BIN_DIVISION, BIN_DIVISION, BIN_DIVISION);
				for (size_t n = 0; n < BIN_SIZE; n++)
				{
					glm::vec4 entity = bin[idx][n];
					h_bin[idx + n] = make_float4(entity.x, entity.y, entity.z, entity.w);
				}
			}
		}
	}

	cudaMalloc(&d_locations, bytes * NUM_OF_ENTITY);
	cudaMalloc(&d_accs, bytes * NUM_OF_ENTITY);
	cudaMalloc(&d_bin, BIN_DIVISION * BIN_DIVISION * BIN_DIVISION * BIN_SIZE * sizeof(float4));
	cudaMalloc(&d_offsets, offset_len * sizeof(int3));

	cudaMemcpy(d_locations, h_locations, bytes * NUM_OF_ENTITY, cudaMemcpyHostToDevice);
	cudaMemcpy(d_accs, h_accs, bytes * NUM_OF_ENTITY, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bin, h_bin, BIN_DIVISION*BIN_DIVISION*BIN_DIVISION*BIN_SIZE * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(d_offsets, offset, offset_len * sizeof(int3), cudaMemcpyHostToDevice);

	dim3 dimBlock(TILE);
	dim3 dimGrid(ceil((double)NUM_OF_ENTITY / (double)TILE));

	bin_kernel <<< dimGrid, dimBlock >>> (d_locations, d_accs, d_bin, d_offsets, offset_len);

	cudaMemcpy(h_accs, d_accs, bytes * NUM_OF_ENTITY, cudaMemcpyDeviceToHost);
	cudaFree(d_locations);
	cudaFree(d_accs);
	cudaFree(d_bin);
	cudaFree(d_offsets);
	free(h_locations);
	free(h_bin);
	
	return h_accs;
}

void interop_caller(cudaGraphicsResource **vbo_resource, float3 *d_speed, float3 *d_accs, float time)
{
	float3 *dptr;
	size_t bytes;
	checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &bytes, *vbo_resource));

	dim3 dimBlock(TILE);
	dim3 dimGrid(ceil((double)NUM_OF_ENTITY / (double)TILE));

	naive_op_kernel <<< dimGrid, dimBlock >>> (dptr, d_speed, d_accs, time, NUM_OF_ENTITY);
	
	checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

