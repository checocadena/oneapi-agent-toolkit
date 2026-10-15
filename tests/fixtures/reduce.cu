#include <cuda_runtime.h>
#include <cstdio>

__constant__ float scale[1];

__global__ void reduce(const float* in, float* out, int n) {
  float v = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    v += in[i] * scale[0];
  for (int off = warpSize / 2; off > 0; off /= 2)
    v += __shfl_down_sync(0xffffffff, v, off);
  if ((threadIdx.x & (warpSize - 1)) == 0) atomicAdd(out, v);
}

int main() {
  const int n = 1 << 20;
  float *d_in, *d_out;
  cudaMalloc(&d_in, n * sizeof(float));
  cudaMalloc(&d_out, sizeof(float));
  float s = 0.5f;
  cudaMemcpyToSymbol(scale, &s, sizeof(float));
  cudaMemset(d_out, 0, sizeof(float));
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start);
  reduce<<<256, 256>>>(d_in, d_out, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0; cudaEventElapsedTime(&ms, start, stop);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }
  float h = 0; cudaMemcpy(&h, d_out, sizeof(float), cudaMemcpyDeviceToHost);
  int dev; cudaGetDevice(&dev); cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
  printf("%s %f %f\n", prop.name, h, ms);
  cudaFree(d_in); cudaFree(d_out);
  return 0;
}
